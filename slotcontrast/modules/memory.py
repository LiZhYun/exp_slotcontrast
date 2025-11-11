"""Memory components for temporal slot tracking inspired by SAM2."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from slotcontrast.utils import make_build_fn


@make_build_fn(__name__, "memory bank")
def build_memory_bank(config, name: str):
    pass  # No special module building needed


@make_build_fn(__name__, "memory encoder")
def build_memory_encoder(config, name: str):
    pass  # No special module building needed


class MemoryBank:
    """FIFO queue for storing temporal object memories."""

    def __init__(
        self,
        capacity: int = 7,
        mem_dim: int = 64,
        temporal_stride_eval: int = 1,
    ):
        """
        Args:
            capacity: Maximum number of memories to store (default: 7)
            mem_dim: Dimension of memory features
            temporal_stride_eval: Sampling stride during evaluation (default: 1)
        """
        self.capacity = capacity
        self.mem_dim = mem_dim
        self.temporal_stride_eval = temporal_stride_eval
        self.memories = {}  # {frame_idx: memory_dict}

    def push(
        self,
        frame_idx: int,
        slots: torch.Tensor,
        features: torch.Tensor,
        masks: torch.Tensor,
        memory_features: torch.Tensor,
        memory_pos_enc: torch.Tensor,
    ):
        """Add new memory to bank."""
        self.memories[frame_idx] = {
            "slots": slots,
            "features": features,
            "masks": masks,
            "memory_features": memory_features,
            "memory_pos_enc": memory_pos_enc,
        }

    def get_memories(
        self,
        current_frame_idx: int,
        training: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Retrieve relevant memories with FIFO logic.
        
        Args:
            current_frame_idx: Current frame index
            training: Whether in training mode
            
        Returns:
            memory_features: Concatenated memory features [B, N_mem, mem_dim] or None
            memory_pos_enc: Concatenated positional encodings [B, N_mem, mem_dim] or None
        """
        stride = 1 if training else self.temporal_stride_eval

        to_cat_memory = []
        to_cat_memory_pos = []

        # Retrieve last (capacity - 1) frames before current frame
        for t_pos in range(1, self.capacity):
            t_rel = self.capacity - t_pos  # How many frames before current frame

            if t_rel == 1:
                # Always include the immediate previous frame
                prev_idx = current_frame_idx - 1
            else:
                # For t_rel >= 2, sample with stride
                prev_idx = ((current_frame_idx - 2) // stride) * stride
                prev_idx = prev_idx - (t_rel - 2) * stride

            if prev_idx >= 0 and prev_idx in self.memories:
                mem = self.memories[prev_idx]
                to_cat_memory.append(mem["memory_features"])
                to_cat_memory_pos.append(mem["memory_pos_enc"])

        if len(to_cat_memory) == 0:
            return None, None

        # Concatenate along sequence dimension
        # Each memory has shape [B, n_slots, mem_dim]
        # Stack to [B, N_frames * n_slots, mem_dim]
        memory_features = torch.cat(to_cat_memory, dim=1)
        memory_pos_enc = torch.cat(to_cat_memory_pos, dim=1)

        return memory_features, memory_pos_enc

    def clear(self):
        """Clear all memories."""
        self.memories.clear()


class SlotMemoryEncoder(nn.Module):
    """
    Encode slots and encoder features into memory representation.
    
    Inspired by SAM2's MemoryEncoder, adapted for object-centric representations.
    Fuses slot representations with mask-weighted visual features.
    """

    def __init__(
        self,
        slot_dim: int = 64,
        feat_dim: int = 768,
        mem_dim: int = 64,
        n_slots: int = 7,
        fusion_blocks: int = 2,
        dropout: float = 0.0,
    ):
        """
        Args:
            slot_dim: Dimension of slot representations
            feat_dim: Dimension of encoder features
            mem_dim: Dimension of memory output
            n_slots: Number of slots
            fusion_blocks: Number of transformer blocks for fusion
            dropout: Dropout rate
        """
        super().__init__()

        self.slot_dim = slot_dim
        self.feat_dim = feat_dim
        self.mem_dim = mem_dim
        self.n_slots = n_slots

        # Project slots to memory dimension
        self.slot_proj = nn.Linear(slot_dim, mem_dim)

        # Project features to memory dimension
        self.feat_proj = nn.Linear(feat_dim, mem_dim)

        # Fusion layers (self-attention among slots for relational reasoning)
        self.fusion_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=mem_dim,
                    nhead=4,
                    dim_feedforward=mem_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(fusion_blocks)
            ]
        )

        self.norm = nn.LayerNorm(mem_dim)

        # Learnable temporal positional encoding for memory bank
        # Shape: [capacity, 1, mem_dim] to be added based on temporal position
        self.temporal_pos_enc = nn.Parameter(torch.zeros(7, 1, mem_dim))
        nn.init.trunc_normal_(self.temporal_pos_enc, std=0.02)

    def forward(
        self,
        slots: torch.Tensor,
        features: torch.Tensor,
        masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse slots and encoder features into memory representation.
        
        Args:
            slots: [B, n_slots, slot_dim] - slot representations from corrector
            features: [B, n_patches, feat_dim] - encoder features
            masks: [B, n_slots, n_patches] - attention masks from corrector
            
        Returns:
            Dictionary containing:
                - memory_features: [B, n_slots, mem_dim]
                - memory_pos_enc: [B, n_slots, mem_dim]
        """
        B, n_slots, _ = slots.shape
        _, n_patches, _ = features.shape

        # Project slots to memory dimension
        slot_feats = self.slot_proj(slots)  # [B, n_slots, mem_dim]

        # Aggregate features per slot using attention masks
        # Normalize masks to get proper weights
        masks_normalized = F.softmax(masks, dim=1)  # [B, n_slots, n_patches]

        # Project features
        feat_proj = self.feat_proj(features)  # [B, n_patches, mem_dim]

        # Weighted aggregation: sum over patches weighted by mask
        # [B, n_slots, n_patches] @ [B, n_patches, mem_dim] -> [B, n_slots, mem_dim]
        slot_features = torch.bmm(masks_normalized, feat_proj)

        # Fuse slot representations with aggregated features (element-wise add like SAM2)
        combined = slot_feats + slot_features  # [B, n_slots, mem_dim]

        # Apply fusion layers for relational reasoning among slots
        memory = combined
        for layer in self.fusion_layers:
            memory = layer(memory)

        memory = self.norm(memory)

        # Temporal positional encoding (will be set based on frame position in memory bank)
        # Use the first position (t=0) as placeholder; actual position will be added in bank
        pos_enc = self.temporal_pos_enc[0].unsqueeze(0).expand(B, n_slots, -1)

        return {
            "memory_features": memory,
            "memory_pos_enc": pos_enc,
        }
