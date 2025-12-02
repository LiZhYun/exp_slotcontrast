from typing import Optional, Tuple

import torch
from torch import nn

from slotcontrast.modules import networks
from slotcontrast.utils import make_build_fn


@make_build_fn(__name__, "grouper")
def build(config, name: str):
    pass  # No special module building needed


class SlotAttention(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        slot_dim: int,
        kvq_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        n_iters: int = 3,
        eps: float = 1e-8,
        use_gru: bool = True,
        use_mlp: bool = True,
        use_ttt: bool = False,
        use_ent: bool = False,
        use_gated: bool = False,
        frozen: bool = False,
    ):
        super().__init__()
        assert n_iters >= 1

        if kvq_dim is None:
            kvq_dim = slot_dim
        self.to_k = nn.Linear(inp_dim, kvq_dim, bias=False)
        self.to_v = nn.Linear(inp_dim, kvq_dim, bias=False)
        self.to_q = nn.Linear(slot_dim, kvq_dim, bias=False)

        if use_gru:
            self.gru = nn.GRUCell(input_size=kvq_dim, hidden_size=slot_dim)
        else:
            assert kvq_dim == slot_dim
            self.gru = None

        self.use_ttt = use_ttt
        self.use_ent = use_ent
        self.use_gated = use_gated

        if use_gated:
            self.gate_proj = nn.Linear(slot_dim, kvq_dim, bias=True)

        if hidden_dim is None:
            hidden_dim = 4 * slot_dim

        if use_mlp:
            self.mlp = networks.MLP(
                slot_dim, slot_dim, [hidden_dim], initial_layer_norm=True, residual=True
            )
        else:
            self.mlp = None

        self.norm_features = nn.LayerNorm(inp_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

        self.n_iters = n_iters
        self.eps = eps
        self.scale = kvq_dim**-0.5

        if frozen:
            for param in self.parameters():
                param.requires_grad = False

    def step(
        self, slots: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one iteration of slot attention."""
        slots = self.norm_slots(slots)
        queries = self.to_q(slots)

        dots = torch.einsum("bsd, bfd -> bsf", queries, keys) * self.scale
        pre_norm_attn = torch.softmax(dots, dim=1)
        attn = pre_norm_attn + self.eps
        attn = attn / attn.sum(-1, keepdim=True)

        updates = torch.einsum("bsf, bfd -> bsd", attn, values)

        if self.gru:
            updated_slots = self.gru(updates.flatten(0, 1), slots.flatten(0, 1))
            slots = updated_slots.unflatten(0, slots.shape[:2])
        elif self.use_ttt:
            # TTT: Test-Time Training with attention-based adaptive updates
            # Following TTT3R: use attention BEFORE softmax (dots) for adaptive updates
            # Higher attention logits = more relevant = larger update weight
            slot_relevance = attn.mean(dim=-1, keepdim=True)  # Average attention logits across features [b, s, 1]
            update_weight = torch.sigmoid(slot_relevance)  # Scale to [0, 1]
            slots = updates * update_weight + slots * (1 - update_weight)
        elif self.use_gated:
            # Gated Attention: Y ⊙ σ(XWθ) - head-specific sigmoid gate
            gate_logits = self.gate_proj(slots)
            gate_scores = torch.sigmoid(gate_logits)
            gated_updates = updates * gate_scores
            slots = slots + gated_updates
        else:
            slots = slots + updates

        if self.mlp is not None:
            slots = self.mlp(slots)

        return slots, pre_norm_attn

    def forward(self, slots: torch.Tensor, features: torch.Tensor, n_iters: Optional[int] = None):
        features = self.norm_features(features)
        keys = self.to_k(features)
        values = self.to_v(features)

        if n_iters is None:
            n_iters = self.n_iters

        entropy_loss = None
        for _ in range(n_iters):
            slots, pre_norm_attn = self.step(slots, keys, values)
            
            if self.use_ent:
                if entropy_loss is None:
                    entropy_loss = self.compute_entropy_loss(pre_norm_attn)
                else:
                    entropy_loss = entropy_loss + self.compute_entropy_loss(pre_norm_attn)

        result = {"slots": slots, "masks": pre_norm_attn}
        if entropy_loss is not None:
            result["entropy_loss"] = entropy_loss / n_iters
        
        return result
    
    def compute_entropy_loss(self, attn: torch.Tensor) -> torch.Tensor:
        entropy = -torch.sum(attn * torch.log(attn + self.eps), dim=1)
        return entropy.mean()
