from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from slotcontrast.utils import make_build_fn


@make_build_fn(__name__, "initializer")
def build(config, name: str):
    pass  # No special module building needed


def greedy_slot_initialization(
    features: torch.Tensor,
    n_slots: int,
    temperature: float = 0.1,
    saliency_mode: str = "norm",
    aggregate: bool = False,
    aggregate_threshold: float = 0.0,
) -> torch.Tensor:
    # Handle video input [B, T, N, D]
    if features.ndim == 4:
        B, T, N, D = features.shape
        features_flat = features.view(B * T, N, D)
        slots = greedy_slot_initialization(features_flat, n_slots, temperature, saliency_mode, aggregate, aggregate_threshold)
        return slots.view(B, T, n_slots, D)
    
    B, N, D = features.shape
    device = features.device
    
    # Normalize features for cosine similarity
    features_norm = F.normalize(features, dim=-1)
    
    # Compute saliency scores
    if saliency_mode == "norm":
        saliency = features.norm(dim=-1)  # [B, N]
    elif saliency_mode == "entropy":
        sim = torch.bmm(features_norm, features_norm.transpose(1, 2)) / temperature
        attn = F.softmax(sim, dim=-1)
        entropy = -(attn * (attn + 1e-8).log()).sum(dim=-1)
        saliency = -entropy
    elif saliency_mode == "variance":
        mean_feat = features.mean(dim=1, keepdim=True)
        saliency = (features - mean_feat).norm(dim=-1)
    elif saliency_mode == "pca":
        # PCA-based saliency: patches with high projection on top PCs are salient
        mean_feat = features.mean(dim=1, keepdim=True)  # [B, 1, D]
        centered = features - mean_feat  # [B, N, D]
        # Covariance matrix [B, D, D]
        cov = torch.bmm(centered.transpose(1, 2), centered) / (N - 1)
        # Top-k eigenvectors (eigh returns ascending order)
        k = min(n_slots, D // 4, 32)
        _, eigenvectors = torch.linalg.eigh(cov)
        top_vecs = eigenvectors[:, :, -k:]  # [B, D, k]
        # Projection magnitude onto top PCs
        saliency = torch.bmm(centered, top_vecs).norm(dim=-1)  # [B, N]
    else:
        saliency = features.norm(dim=-1)
    
    # Greedy selection
    slots = []
    mask = torch.ones(B, N, device=device)
    
    for _ in range(n_slots):
        masked_saliency = saliency * mask
        idx = masked_saliency.argmax(dim=-1)  # [B]
        selected = features[torch.arange(B, device=device), idx]  # [B, D]
        selected_norm = F.normalize(selected, dim=-1).unsqueeze(1)  # [B, 1, D]
        similarity = (features_norm * selected_norm).sum(dim=-1)  # [B, N]
        
        if aggregate:
            # Aggregate similar patches into slot via weighted average
            agg_weights = (similarity * mask).clamp(min=0)  # [B, N]
            agg_weights = agg_weights * (similarity > aggregate_threshold).float()
            agg_weights = agg_weights / (agg_weights.sum(dim=-1, keepdim=True) + 1e-8)
            slot = torch.einsum("bn,bnd->bd", agg_weights, features)
        else:
            slot = selected
        slots.append(slot)
        
        # Suppress similar features
        suppression = 1 - similarity.clamp(0, 1)
        mask = mask * suppression
    
    return torch.stack(slots, dim=1)  # [B, n_slots, D]


class RandomInit(nn.Module):
    """Sampled random initialization for all slots."""

    def __init__(self, n_slots: int, dim: int, initial_std: Optional[float] = None, **kwargs):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        self.mean = nn.Parameter(torch.zeros(1, 1, dim))
        if initial_std is None:
            initial_std = dim**-0.5
        self.log_std = nn.Parameter(torch.log(torch.ones(1, 1, dim) * initial_std))

    def forward(self, batch_size: int, features: Optional[torch.Tensor] = None):
        noise = torch.randn(batch_size, self.n_slots, self.dim, device=self.mean.device)
        return self.mean + noise * self.log_std.exp()


class FixedLearnedInit(nn.Module):
    """Learned initialization with a fixed number of slots."""

    def __init__(
        self, n_slots: int, dim: int, initial_std: Optional[float] = None, frozen: bool = False,
        init_mode: str = "first_frame", **kwargs
    ):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        self.init_mode = init_mode
        if initial_std is None:
            initial_std = dim**-0.5
        self.slots = nn.Parameter(torch.randn(1, n_slots, dim) * initial_std)
        if frozen:
            self.slots.requires_grad_(False)

    def forward(self, batch_size: int, features: Optional[torch.Tensor] = None):
        slots = self.slots.expand(batch_size, -1, -1)  # [B, n_slots, D]
        
        # For video input with per_frame mode, expand to [B, T, n_slots, D]
        if features is not None and features.ndim == 4 and self.init_mode == "per_frame":
            T = features.shape[1]
            slots = slots.unsqueeze(1).expand(-1, T, -1, -1)
        
        return slots


class GreedyFeatureInit(nn.Module):
    def __init__(self, n_slots: int, dim: int, temperature: float = 0.1, 
                 saliency_mode: str = "norm", init_mode: str = "first_frame",
                 aggregate: bool = False, aggregate_threshold: float = 0.5):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        self.temperature = temperature
        self.saliency_mode = saliency_mode
        self.init_mode = init_mode
        self.aggregate = aggregate
        self.aggregate_threshold = aggregate_threshold
        self.fallback = nn.Parameter(torch.randn(1, n_slots, dim) * dim**-0.5)

    def forward(self, batch_size: int, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if features is None:
            return self.fallback.expand(batch_size, -1, -1)
        
        if features.ndim == 4:
            if self.init_mode == "first_frame":
                first_frame_feat = features[:, 0]
                return greedy_slot_initialization(first_frame_feat, self.n_slots, self.temperature, 
                                                   self.saliency_mode, self.aggregate, self.aggregate_threshold)
            else:
                return greedy_slot_initialization(features, self.n_slots, self.temperature, 
                                                   self.saliency_mode, self.aggregate, self.aggregate_threshold)
        
        return greedy_slot_initialization(features, self.n_slots, self.temperature, 
                                           self.saliency_mode, self.aggregate, self.aggregate_threshold)
