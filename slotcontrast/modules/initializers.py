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
) -> torch.Tensor:
    """Greedy feature-guided slot initialization based on cosine similarity.
    
    Args:
        features: Input features [B, N, D] or [B, T, N, D] for video
        n_slots: Number of slots to initialize
        temperature: Temperature for softmax saliency (used in 'entropy' mode)
        saliency_mode: How to compute saliency ('norm', 'entropy', 'variance')
    
    Returns:
        Selected slot initializations [B, n_slots, D] or [B, T, n_slots, D]
    """
    # Handle video input [B, T, N, D]
    if features.ndim == 4:
        B, T, N, D = features.shape
        features_flat = features.view(B * T, N, D)
        slots = greedy_slot_initialization(features_flat, n_slots, temperature, saliency_mode)
        return slots.view(B, T, n_slots, D)
    
    B, N, D = features.shape
    device = features.device
    
    # Normalize features for cosine similarity
    features_norm = F.normalize(features, dim=-1)
    
    # Compute saliency scores
    if saliency_mode == "norm":
        saliency = features.norm(dim=-1)  # [B, N]
    elif saliency_mode == "entropy":
        # Self-attention based saliency
        sim = torch.bmm(features_norm, features_norm.transpose(1, 2)) / temperature
        attn = F.softmax(sim, dim=-1)
        entropy = -(attn * (attn + 1e-8).log()).sum(dim=-1)
        saliency = -entropy  # Lower entropy = more salient
    elif saliency_mode == "variance":
        # Features that differ most from the mean
        mean_feat = features.mean(dim=1, keepdim=True)
        saliency = (features - mean_feat).norm(dim=-1)
    else:
        saliency = features.norm(dim=-1)
    
    # Greedy selection
    slots = []
    mask = torch.ones(B, N, device=device)  # Track available positions
    
    for _ in range(n_slots):
        # Apply mask to saliency
        masked_saliency = saliency * mask
        
        # Select highest saliency point
        idx = masked_saliency.argmax(dim=-1)  # [B]
        
        # Gather selected features
        selected = features[torch.arange(B, device=device), idx]  # [B, D]
        slots.append(selected)
        
        # Update mask: suppress similar features using cosine similarity
        selected_norm = F.normalize(selected, dim=-1).unsqueeze(1)  # [B, 1, D]
        similarity = (features_norm * selected_norm).sum(dim=-1)  # [B, N]
        
        # Suppress features with high similarity to selected
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
        self, n_slots: int, dim: int, initial_std: Optional[float] = None, frozen: bool = False, **kwargs
    ):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        if initial_std is None:
            initial_std = dim**-0.5
        self.slots = nn.Parameter(torch.randn(1, n_slots, dim) * initial_std)
        if frozen:
            self.slots.requires_grad_(False)

    def forward(self, batch_size: int, features: Optional[torch.Tensor] = None):
        return self.slots.expand(batch_size, -1, -1)


class GreedyFeatureInit(nn.Module):
    """Feature-guided greedy initialization. Selects diverse high-saliency features.
    
    Args:
        init_mode: 'first_frame' uses only first frame features, 'per_frame' initializes each frame independently
    """

    def __init__(self, n_slots: int, dim: int, temperature: float = 0.1, 
                 saliency_mode: str = "norm", init_mode: str = "first_frame"):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        self.temperature = temperature
        self.saliency_mode = saliency_mode
        self.init_mode = init_mode
        # Fallback params for when features not provided
        self.fallback = nn.Parameter(torch.randn(1, n_slots, dim) * dim**-0.5)

    def forward(self, batch_size: int, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if features is None:
            return self.fallback.expand(batch_size, -1, -1)
        
        # For video [B, T, N, D]
        if features.ndim == 4:
            if self.init_mode == "first_frame":
                # Only use first frame, return [B, n_slots, D]
                first_frame_feat = features[:, 0]  # [B, N, D]
                return greedy_slot_initialization(first_frame_feat, self.n_slots, self.temperature, self.saliency_mode)
            else:  # per_frame
                # Initialize each frame independently, return [B, T, n_slots, D]
                return greedy_slot_initialization(features, self.n_slots, self.temperature, self.saliency_mode)
        
        # For image [B, N, D]
        return greedy_slot_initialization(features, self.n_slots, self.temperature, self.saliency_mode)
