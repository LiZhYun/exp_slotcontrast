from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from slotcontrast.utils import make_build_fn


@make_build_fn(__name__, "initializer")
def build(config, name: str):
    pass  # No special module building needed


def compute_neighbor_similarity(
    features_norm: torch.Tensor, patch_h: int, patch_w: int, radius: int = 1
) -> torch.Tensor:
    """Compute mean cosine similarity to spatial neighbors for each patch.
    
    Args:
        features_norm: [B, N, D] L2-normalized features
        patch_h, patch_w: Spatial dimensions of patch grid
        radius: Neighborhood radius (1 = 3x3, 2 = 5x5)
    
    Returns:
        local_sim: [B, N] mean similarity to neighbors
    """
    B, N, D = features_norm.shape
    
    # Reshape to spatial grid: [B, D, H, W]
    features_spatial = features_norm.view(B, patch_h, patch_w, D).permute(0, 3, 1, 2)
    
    # Use unfold to extract neighbor windows
    kernel_size = 2 * radius + 1
    # Unfold: [B, D, H, W] -> [B, D*k*k, H*W]
    unfolded = F.unfold(features_spatial, kernel_size=kernel_size, padding=radius)
    # Reshape to [B, D, k*k, N]
    unfolded = unfolded.view(B, D, kernel_size * kernel_size, N)
    
    # Center patch features: [B, D, N]
    center = features_spatial.view(B, D, N)
    
    # Dot product between center and all neighbors: [B, k*k, N]
    similarity = torch.einsum("bdn,bdkn->bkn", center, unfolded)
    
    # Mean over neighbors (k*k includes self, but that's fine)
    local_sim = similarity.mean(dim=1)  # [B, N]
    
    return local_sim


def compute_neighbor_similarity_stats(
    features_norm: torch.Tensor, patch_h: int, patch_w: int, radius: int = 1
) -> tuple:
    """Compute mean and variance of cosine similarity to spatial neighbors."""
    B, N, D = features_norm.shape
    features_spatial = features_norm.view(B, patch_h, patch_w, D).permute(0, 3, 1, 2)
    kernel_size = 2 * radius + 1
    unfolded = F.unfold(features_spatial, kernel_size=kernel_size, padding=radius)
    unfolded = unfolded.view(B, D, kernel_size * kernel_size, N)
    center = features_spatial.view(B, D, N)
    similarity = torch.einsum("bdn,bdkn->bkn", center, unfolded)  # [B, k*k, N]
    local_sim_mean = similarity.mean(dim=1)
    local_sim_var = similarity.var(dim=1)
    return local_sim_mean, local_sim_var


def compute_gaussian_neighbor_similarity(
    features_norm: torch.Tensor, patch_h: int, patch_w: int, radius: int = 1, sigma: float = 1.0
) -> torch.Tensor:
    """Compute distance-weighted (Gaussian) mean similarity to spatial neighbors."""
    B, N, D = features_norm.shape
    device = features_norm.device
    
    features_spatial = features_norm.view(B, patch_h, patch_w, D).permute(0, 3, 1, 2)
    kernel_size = 2 * radius + 1
    unfolded = F.unfold(features_spatial, kernel_size=kernel_size, padding=radius)
    unfolded = unfolded.view(B, D, kernel_size * kernel_size, N)
    center = features_spatial.view(B, D, N)
    similarity = torch.einsum("bdn,bdkn->bkn", center, unfolded)  # [B, k*k, N]
    
    # Create Gaussian weights based on distance from center
    offsets = torch.arange(-radius, radius + 1, device=device).float()
    grid_h, grid_w = torch.meshgrid(offsets, offsets, indexing='ij')
    dist_sq = (grid_h ** 2 + grid_w ** 2).reshape(-1)  # [k*k]
    weights = torch.exp(-dist_sq / (2 * sigma ** 2))  # [k*k]
    weights = weights / weights.sum()  # Normalize
    
    # Weighted mean
    local_sim = (similarity * weights.view(1, -1, 1)).sum(dim=1)  # [B, N]
    return local_sim


def compute_feature_density(features_norm: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Compute feature-space density: mean similarity to k-nearest neighbors in feature space."""
    B, N, D = features_norm.shape
    # [B, N, N] similarity matrix
    sim_matrix = torch.bmm(features_norm, features_norm.transpose(1, 2))
    # Top-k similarities (excluding self which is 1.0)
    topk_sim, _ = sim_matrix.topk(k + 1, dim=-1)  # [B, N, k+1]
    # Exclude the first one (self-similarity = 1)
    density = topk_sim[:, :, 1:].mean(dim=-1)  # [B, N]
    return density


def greedy_slot_initialization(
    features: torch.Tensor,
    n_slots: int,
    temperature: float = 0.1,
    saliency_mode: str = "norm",
    aggregate: bool = False,
    aggregate_threshold: float = 0.0,
    neighbor_radius: int = 1,
    saliency_smoothing: int = 0,
    selection_mode: str = "hard",
    soft_topk: int = 5,
    neighbor_avg_radius: int = 1,
    saliency_alpha: float = 1.0,
    spatial_suppression_radius: int = 0,
    spatial_suppression_strength: float = 0.5,
) -> torch.Tensor:
    # Handle video input [B, T, N, D]
    if features.ndim == 4:
        B, T, N, D = features.shape
        features_flat = features.view(B * T, N, D)
        slots = greedy_slot_initialization(
            features_flat, n_slots, temperature, saliency_mode, 
            aggregate, aggregate_threshold, neighbor_radius, saliency_smoothing,
            selection_mode, soft_topk, neighbor_avg_radius, saliency_alpha,
            spatial_suppression_radius, spatial_suppression_strength
        )
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
    elif saliency_mode == "local_consistency":
        # Local consistency: similar to neighbors, different from global
        # Selects object interiors rather than boundaries
        patch_hw = int(N ** 0.5)  # Assume square grid
        
        # Local similarity: mean cosine sim to spatial neighbors
        local_sim = compute_neighbor_similarity(features_norm, patch_hw, patch_hw, neighbor_radius)
        
        # Global similarity: cosine sim to global mean
        global_mean = F.normalize(features.mean(dim=1, keepdim=True), dim=-1)  # [B, 1, D]
        global_sim = (features_norm * global_mean).sum(dim=-1)  # [B, N]
        
        # High local_sim + low global_sim = interior of distinct object
        # saliency_alpha controls the weight of global_sim penalty
        saliency = local_sim - saliency_alpha * global_sim
    elif saliency_mode == "local_consistency_pca":
        # Combined: local consistency weighted by PCA distinctiveness
        # Selects interiors of DISTINCTIVE objects (not background interiors)
        patch_hw = int(N ** 0.5)
        
        # 1. Local consistency score
        local_sim = compute_neighbor_similarity(features_norm, patch_hw, patch_hw, neighbor_radius)
        global_mean = F.normalize(features.mean(dim=1, keepdim=True), dim=-1)
        global_sim = (features_norm * global_mean).sum(dim=-1)
        local_consistency = local_sim - global_sim  # [B, N]
        
        # 2. PCA distinctiveness score
        mean_feat = features.mean(dim=1, keepdim=True)
        centered = features - mean_feat
        cov = torch.bmm(centered.transpose(1, 2), centered) / (N - 1)
        k = min(n_slots, D // 4, 32)
        _, eigenvectors = torch.linalg.eigh(cov)
        top_vecs = eigenvectors[:, :, -k:]
        pca_score = torch.bmm(centered, top_vecs).norm(dim=-1)  # [B, N]
        
        # Normalize PCA score to [0, 1] range
        pca_min = pca_score.min(dim=-1, keepdim=True)[0]
        pca_max = pca_score.max(dim=-1, keepdim=True)[0]
        pca_normalized = (pca_score - pca_min) / (pca_max - pca_min + 1e-8)
        
        # Combine: local_consistency * (1 + pca_normalized)
        # High when: interior (local_consistency > 0) AND distinctive (high pca)
        saliency = local_consistency * (1 + pca_normalized)
    elif saliency_mode == "local_consistency_ms":
        # Multi-scale local consistency: consistent at both small and large scales
        patch_hw = int(N ** 0.5)
        local_sim_r1 = compute_neighbor_similarity(features_norm, patch_hw, patch_hw, radius=1)
        local_sim_r2 = compute_neighbor_similarity(features_norm, patch_hw, patch_hw, radius=2)
        global_mean = F.normalize(features.mean(dim=1, keepdim=True), dim=-1)
        global_sim = (features_norm * global_mean).sum(dim=-1)
        saliency = (local_sim_r1 + local_sim_r2) / 2 - global_sim
    elif saliency_mode == "local_consistency_uniform":
        # Local consistency with uniformity: penalize high variance (edge of interior)
        patch_hw = int(N ** 0.5)
        local_sim, local_var = compute_neighbor_similarity_stats(features_norm, patch_hw, patch_hw, neighbor_radius)
        global_mean = F.normalize(features.mean(dim=1, keepdim=True), dim=-1)
        global_sim = (features_norm * global_mean).sum(dim=-1)
        # Normalize variance to [0, 1] for balanced weighting
        var_max = local_var.max(dim=-1, keepdim=True)[0] + 1e-8
        local_var_norm = local_var / var_max
        saliency = local_sim - global_sim - 0.5 * local_var_norm
    elif saliency_mode == "local_consistency_centroid":
        # Regional centroid: prefer patches near center of coherent regions
        patch_hw = int(N ** 0.5)
        local_sim = compute_neighbor_similarity(features_norm, patch_hw, patch_hw, neighbor_radius)
        global_mean = F.normalize(features.mean(dim=1, keepdim=True), dim=-1)
        global_sim = (features_norm * global_mean).sum(dim=-1)
        local_consistency = local_sim - global_sim
        # Spatial coordinates
        coords_h = torch.arange(patch_hw, device=device).float()
        coords_w = torch.arange(patch_hw, device=device).float()
        grid_h, grid_w = torch.meshgrid(coords_h, coords_w, indexing='ij')
        # Compute weighted centroid distance using local consistency as weights
        lc_pos = local_consistency.clamp(min=0).view(B, 1, patch_hw, patch_hw)
        weights_sum = F.avg_pool2d(lc_pos, 2 * neighbor_radius + 1, stride=1, padding=neighbor_radius)
        weighted_h = F.avg_pool2d(lc_pos * grid_h.view(1, 1, patch_hw, patch_hw), 2 * neighbor_radius + 1, stride=1, padding=neighbor_radius)
        weighted_w = F.avg_pool2d(lc_pos * grid_w.view(1, 1, patch_hw, patch_hw), 2 * neighbor_radius + 1, stride=1, padding=neighbor_radius)
        centroid_h = weighted_h / (weights_sum + 1e-8)
        centroid_w = weighted_w / (weights_sum + 1e-8)
        dist_to_centroid = ((grid_h.view(1, 1, patch_hw, patch_hw) - centroid_h) ** 2 +
                           (grid_w.view(1, 1, patch_hw, patch_hw) - centroid_w) ** 2).sqrt()
        dist_to_centroid = dist_to_centroid.view(B, N)
        # Normalize distance
        dist_max = dist_to_centroid.max(dim=-1, keepdim=True)[0] + 1e-8
        dist_norm = dist_to_centroid / dist_max
        saliency = local_consistency - 0.3 * dist_norm
    elif saliency_mode == "local_consistency_balanced":
        # Boundary-balanced: avoid patches too deep in interior
        patch_hw = int(N ** 0.5)
        local_sim = compute_neighbor_similarity(features_norm, patch_hw, patch_hw, neighbor_radius)
        global_mean = F.normalize(features.mean(dim=1, keepdim=True), dim=-1)
        global_sim = (features_norm * global_mean).sum(dim=-1)
        local_consistency = local_sim - global_sim
        # Approximate distance to boundary using iterative max-pool of boundary indicator
        boundary_indicator = (1.0 - local_sim).view(B, 1, patch_hw, patch_hw)
        dist_to_boundary = boundary_indicator
        for _ in range(3):
            dist_to_boundary = F.max_pool2d(dist_to_boundary, 3, stride=1, padding=1)
        dist_to_boundary = dist_to_boundary.view(B, N)
        # Normalize and penalize extreme distances (want moderate interior, not too deep)
        dist_min = dist_to_boundary.min(dim=-1, keepdim=True)[0]
        dist_max = dist_to_boundary.max(dim=-1, keepdim=True)[0]
        dist_norm = (dist_to_boundary - dist_min) / (dist_max - dist_min + 1e-8)
        optimal_dist = 0.4  # Sweet spot: moderately interior
        dist_penalty = (dist_norm - optimal_dist).abs()
        saliency = local_consistency - 0.3 * dist_penalty
    elif saliency_mode == "local_consistency_soft":
        # Soft global: compare to local background instead of global mean
        patch_hw = int(N ** 0.5)
        local_sim = compute_neighbor_similarity(features_norm, patch_hw, patch_hw, neighbor_radius)
        # Local background: large-scale average (excludes immediate neighborhood)
        features_spatial = features_norm.view(B, patch_hw, patch_hw, D).permute(0, 3, 1, 2)
        large_radius = max(3, patch_hw // 4)
        local_bg = F.avg_pool2d(features_spatial, 2 * large_radius + 1, stride=1, padding=large_radius)
        local_bg = F.normalize(local_bg.permute(0, 2, 3, 1).reshape(B, N, D), dim=-1)
        bg_sim = (features_norm * local_bg).sum(dim=-1)
        saliency = local_sim - bg_sim
    elif saliency_mode == "local_consistency_density":
        # Local consistency weighted by feature-space density
        # High density = patch is in a dense feature cluster (object interior)
        patch_hw = int(N ** 0.5)
        local_sim = compute_neighbor_similarity(features_norm, patch_hw, patch_hw, neighbor_radius)
        global_mean = F.normalize(features.mean(dim=1, keepdim=True), dim=-1)
        global_sim = (features_norm * global_mean).sum(dim=-1)
        local_consistency = local_sim - saliency_alpha * global_sim
        # Feature-space density
        density = compute_feature_density(features_norm, k=10)
        # Normalize density to [0, 1]
        d_min, d_max = density.min(dim=-1, keepdim=True)[0], density.max(dim=-1, keepdim=True)[0]
        density_norm = (density - d_min) / (d_max - d_min + 1e-8)
        # Combine: local_consistency * (1 + density_norm)
        saliency = local_consistency * (1 + density_norm)
    elif saliency_mode == "local_consistency_second":
        # Second-order: prefer patches whose neighbors also have high local consistency
        patch_hw = int(N ** 0.5)
        local_sim = compute_neighbor_similarity(features_norm, patch_hw, patch_hw, neighbor_radius)
        global_mean = F.normalize(features.mean(dim=1, keepdim=True), dim=-1)
        global_sim = (features_norm * global_mean).sum(dim=-1)
        local_consistency = local_sim - saliency_alpha * global_sim  # [B, N]
        # Second-order: average local_consistency of spatial neighbors
        lc_spatial = local_consistency.view(B, 1, patch_hw, patch_hw)
        kernel_size = 2 * neighbor_radius + 1
        lc_second = F.avg_pool2d(lc_spatial, kernel_size, stride=1, padding=neighbor_radius)
        lc_second = lc_second.view(B, N)
        # Combine first and second order
        saliency = 0.5 * local_consistency + 0.5 * lc_second
    elif saliency_mode == "local_consistency_gaussian":
        # Distance-weighted (Gaussian) local consistency
        patch_hw = int(N ** 0.5)
        # Use Gaussian-weighted neighbor similarity (closer neighbors matter more)
        local_sim = compute_gaussian_neighbor_similarity(
            features_norm, patch_hw, patch_hw, neighbor_radius, sigma=neighbor_radius / 2
        )
        global_mean = F.normalize(features.mean(dim=1, keepdim=True), dim=-1)
        global_sim = (features_norm * global_mean).sum(dim=-1)
        saliency = local_sim - saliency_alpha * global_sim
    else:
        saliency = features.norm(dim=-1)
    
    # Optional: Spatial smoothing of saliency (shifts peaks toward region centers)
    if saliency_smoothing > 0:
        patch_hw = int(N ** 0.5)  # Assume square grid
        saliency_spatial = saliency.view(B, 1, patch_hw, patch_hw)
        kernel_size = 2 * saliency_smoothing + 1  # smoothing=1 -> 3x3, smoothing=2 -> 5x5
        saliency_smoothed = F.avg_pool2d(
            saliency_spatial, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=saliency_smoothing
        )
        saliency = saliency_smoothed.view(B, N)
    
    # Greedy selection
    slots = []
    mask = torch.ones(B, N, device=device)
    patch_hw = int(N ** 0.5)  # For spatial operations
    
    for _ in range(n_slots):
        masked_saliency = saliency * mask
        
        if selection_mode == "soft":
            # Soft selection: weighted average of top-k patches
            topk_values, topk_indices = masked_saliency.topk(soft_topk, dim=-1)  # [B, k]
            topk_weights = F.softmax(topk_values / temperature, dim=-1)  # [B, k]
            topk_features = features.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, D))  # [B, k, D]
            slot = (topk_weights.unsqueeze(-1) * topk_features).sum(dim=1)  # [B, D]
            # Use top-1 for suppression calculation
            idx = topk_indices[:, 0]
            selected = features[torch.arange(B, device=device), idx]
        elif selection_mode == "neighbor_avg":
            # Post-selection neighborhood averaging
            idx = masked_saliency.argmax(dim=-1)  # [B]
            selected = features[torch.arange(B, device=device), idx]  # [B, D]
            # Convert idx to spatial coordinates and average with neighbors
            idx_h = idx // patch_hw  # [B]
            idx_w = idx % patch_hw   # [B]
            # Gather neighbor features and average
            slot = _average_with_spatial_neighbors(
                features, idx_h, idx_w, patch_hw, neighbor_avg_radius
            )
        elif selection_mode == "knn_refine":
            # Select patch, then refine by averaging with k-nearest neighbors in feature space
            idx = masked_saliency.argmax(dim=-1)  # [B]
            selected = features[torch.arange(B, device=device), idx]  # [B, D]
            # Find k-nearest neighbors in feature space
            selected_norm = F.normalize(selected, dim=-1).unsqueeze(1)  # [B, 1, D]
            sim_to_selected = (features_norm * selected_norm).sum(dim=-1)  # [B, N]
            _, knn_indices = sim_to_selected.topk(soft_topk, dim=-1)  # [B, k]
            knn_features = features.gather(1, knn_indices.unsqueeze(-1).expand(-1, -1, D))  # [B, k, D]
            slot = knn_features.mean(dim=1)  # [B, D]
        elif selection_mode == "centroid":
            # Select patch, then compute centroid of all similar patches (soft region)
            idx = masked_saliency.argmax(dim=-1)  # [B]
            selected = features[torch.arange(B, device=device), idx]  # [B, D]
            # Compute similarity to selected and use as soft weights
            selected_norm = F.normalize(selected, dim=-1).unsqueeze(1)  # [B, 1, D]
            sim_to_selected = (features_norm * selected_norm).sum(dim=-1)  # [B, N]
            # Soft weights: only patches similar enough (above threshold)
            soft_weights = (sim_to_selected * mask).clamp(min=0)  # [B, N]
            soft_weights = soft_weights * (sim_to_selected > 0.5).float()  # Threshold
            soft_weights = soft_weights / (soft_weights.sum(dim=-1, keepdim=True) + 1e-8)
            slot = torch.einsum("bn,bnd->bd", soft_weights, features)  # [B, D]
        else:  # "hard" - original behavior
            idx = masked_saliency.argmax(dim=-1)  # [B]
            selected = features[torch.arange(B, device=device), idx]  # [B, D]
            slot = selected
        
        # Optional: aggregate (can combine with any selection mode)
        if aggregate:
            selected_norm = F.normalize(selected, dim=-1).unsqueeze(1)  # [B, 1, D]
            similarity = (features_norm * selected_norm).sum(dim=-1)  # [B, N]
            agg_weights = (similarity * mask).clamp(min=0)  # [B, N]
            agg_weights = agg_weights * (similarity > aggregate_threshold).float()
            agg_weights = agg_weights / (agg_weights.sum(dim=-1, keepdim=True) + 1e-8)
            slot = torch.einsum("bn,bnd->bd", agg_weights, features)
        
        slots.append(slot)
        
        # Suppress similar features (use selected for suppression regardless of mode)
        selected_norm = F.normalize(selected, dim=-1).unsqueeze(1)
        similarity = (features_norm * selected_norm).sum(dim=-1)
        feature_suppression = 1 - similarity.clamp(0, 1)
        
        # Optional: Spatial suppression (suppress nearby patches regardless of features)
        if spatial_suppression_radius > 0:
            idx_h = idx // patch_hw
            idx_w = idx % patch_hw
            spatial_suppression = _compute_spatial_suppression(
                idx_h, idx_w, patch_hw, spatial_suppression_radius, device
            )
            # Combine: blend feature and spatial suppression
            # strength=0: pure feature, strength=1: pure spatial
            suppression = (1 - spatial_suppression_strength) * feature_suppression + \
                          spatial_suppression_strength * spatial_suppression
        else:
            suppression = feature_suppression
        
        mask = mask * suppression
    
    return torch.stack(slots, dim=1)  # [B, n_slots, D]


def _average_with_spatial_neighbors(
    features: torch.Tensor,
    idx_h: torch.Tensor,
    idx_w: torch.Tensor,
    patch_hw: int,
    radius: int,
) -> torch.Tensor:
    """Average selected patch with its spatial neighbors."""
    B, N, D = features.shape
    device = features.device
    
    # Reshape features to spatial grid
    features_spatial = features.view(B, patch_hw, patch_hw, D)
    
    # Collect neighbor features for each batch element
    slots = []
    for b in range(B):
        h, w = idx_h[b].item(), idx_w[b].item()
        h_min, h_max = max(0, h - radius), min(patch_hw, h + radius + 1)
        w_min, w_max = max(0, w - radius), min(patch_hw, w + radius + 1)
        
        # Get neighbor patch features and average
        neighbor_feats = features_spatial[b, h_min:h_max, w_min:w_max, :]  # [h, w, D]
        slot = neighbor_feats.mean(dim=(0, 1))  # [D]
        slots.append(slot)
    
    return torch.stack(slots, dim=0)  # [B, D]


def _compute_spatial_suppression(
    idx_h: torch.Tensor,
    idx_w: torch.Tensor,
    patch_hw: int,
    radius: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute spatial suppression mask based on distance from selected patches.
    
    Returns suppression values in [0, 1] where 0 = fully suppressed (near selected),
    1 = not suppressed (far from selected).
    """
    B = idx_h.shape[0]
    N = patch_hw * patch_hw
    
    # Create coordinate grids
    coords_h = torch.arange(patch_hw, device=device).float()
    coords_w = torch.arange(patch_hw, device=device).float()
    grid_h, grid_w = torch.meshgrid(coords_h, coords_w, indexing='ij')
    grid_h = grid_h.reshape(1, N).expand(B, N)  # [B, N]
    grid_w = grid_w.reshape(1, N).expand(B, N)  # [B, N]
    
    # Compute distance from each patch to selected patch
    dist_h = (grid_h - idx_h.float().unsqueeze(1)).abs()  # [B, N]
    dist_w = (grid_w - idx_w.float().unsqueeze(1)).abs()  # [B, N]
    dist = torch.max(dist_h, dist_w)  # Chebyshev distance (square neighborhood)
    
    # Suppression: 0 within radius, linear ramp to 1 outside
    # Patches within radius get suppression = 0 (will be multiplied by mask)
    # Patches outside radius get suppression = 1 (no suppression)
    suppression = (dist / (radius + 1)).clamp(0, 1)
    
    return suppression


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
                 aggregate: bool = False, aggregate_threshold: float = 0.5,
                 neighbor_radius: int = 1, saliency_smoothing: int = 0,
                 selection_mode: str = "hard", soft_topk: int = 5,
                 neighbor_avg_radius: int = 1, saliency_alpha: float = 1.0,
                 spatial_suppression_radius: int = 0, spatial_suppression_strength: float = 0.5,
                 refine_linear: bool = False,
                 **kwargs):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        self.temperature = temperature
        self.saliency_mode = saliency_mode
        self.init_mode = init_mode
        self.aggregate = aggregate
        self.saliency_smoothing = saliency_smoothing
        self.aggregate_threshold = aggregate_threshold
        self.neighbor_radius = neighbor_radius
        self.selection_mode = selection_mode
        self.soft_topk = soft_topk
        self.neighbor_avg_radius = neighbor_avg_radius
        self.saliency_alpha = saliency_alpha
        self.spatial_suppression_radius = spatial_suppression_radius
        self.spatial_suppression_strength = spatial_suppression_strength
        self.fallback = nn.Parameter(torch.randn(1, n_slots, dim) * dim**-0.5)
        
        # Optional: learnable refinement with zero init
        self.refine_linear = refine_linear
        if refine_linear:
            self.refine = nn.Linear(dim, dim)
            nn.init.zeros_(self.refine.weight)
            nn.init.zeros_(self.refine.bias)

    def forward(self, batch_size: int, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if features is None:
            return self.fallback.expand(batch_size, -1, -1)
        
        if features.ndim == 4:
            if self.init_mode == "first_frame":
                first_frame_feat = features[:, 0]
                slots = greedy_slot_initialization(
                    first_frame_feat, self.n_slots, self.temperature, 
                    self.saliency_mode, self.aggregate, self.aggregate_threshold,
                    self.neighbor_radius, self.saliency_smoothing,
                    self.selection_mode, self.soft_topk, self.neighbor_avg_radius,
                    self.saliency_alpha, self.spatial_suppression_radius,
                    self.spatial_suppression_strength
                )
            else:
                slots = greedy_slot_initialization(
                    features, self.n_slots, self.temperature, 
                    self.saliency_mode, self.aggregate, self.aggregate_threshold,
                    self.neighbor_radius, self.saliency_smoothing,
                    self.selection_mode, self.soft_topk, self.neighbor_avg_radius,
                    self.saliency_alpha, self.spatial_suppression_radius,
                    self.spatial_suppression_strength
                )
        else:
            slots = greedy_slot_initialization(
                features, self.n_slots, self.temperature, 
                self.saliency_mode, self.aggregate, self.aggregate_threshold,
                self.neighbor_radius, self.saliency_smoothing,
                self.selection_mode, self.soft_topk, self.neighbor_avg_radius,
                self.saliency_alpha, self.spatial_suppression_radius,
                self.spatial_suppression_strength
            )
        
        # Apply learnable refinement: slots = slots + refine(slots)
        if self.refine_linear:
            slots = slots + self.refine(slots)
        
        return slots


def cluster_and_centroid(
    features: torch.Tensor,
    n_clusters: int,
    method: str = "agglomerative",
    affinity: str = "cosine",
    linkage: str = "average",
) -> torch.Tensor:
    """Cluster features and return centroids.
    
    Args:
        features: [N, D] feature vectors
        n_clusters: Number of clusters
        method: 'agglomerative', 'kmeans', or 'spectral'
        affinity: Distance metric for agglomerative ('cosine', 'euclidean')
        linkage: Linkage type for agglomerative ('average', 'complete', 'single')
    
    Returns:
        centroids: [n_clusters, D] cluster centroids
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
    
    device = features.device
    features_np = features.detach().cpu().numpy()
    N, D = features_np.shape
    
    # Handle edge case: fewer features than clusters
    if N <= n_clusters:
        # Pad with mean if needed
        centroids = np.zeros((n_clusters, D), dtype=features_np.dtype)
        centroids[:N] = features_np
        if N < n_clusters:
            centroids[N:] = features_np.mean(axis=0)
        return torch.from_numpy(centroids).to(device)
    
    # Run clustering
    if method == "agglomerative":
        # For cosine affinity, normalize features first
        if affinity == "cosine":
            features_normalized = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8)
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters, metric="euclidean", linkage=linkage
            )
            labels = clusterer.fit_predict(features_normalized)
        else:
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters, metric=affinity, linkage=linkage
            )
            labels = clusterer.fit_predict(features_np)
    elif method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = clusterer.fit_predict(features_np)
    elif method == "spectral":
        clusterer = SpectralClustering(
            n_clusters=n_clusters, affinity="nearest_neighbors", random_state=42
        )
        labels = clusterer.fit_predict(features_np)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Compute centroids
    centroids = np.zeros((n_clusters, D), dtype=features_np.dtype)
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            centroids[k] = features_np[mask].mean(axis=0)
        else:
            # Empty cluster: use global mean
            centroids[k] = features_np.mean(axis=0)
    
    return torch.from_numpy(centroids).to(device)


def cluster_slot_initialization(
    features: torch.Tensor,
    n_slots: int,
    method: str = "agglomerative",
    affinity: str = "cosine",
    linkage: str = "average",
) -> torch.Tensor:
    """Cluster features and return centroids as slot initialization.
    
    Args:
        features: [B, N, D] for image or [B*T, N, D] for flattened video
        n_slots: Number of slots/clusters
        method: Clustering method
        affinity: Distance metric
        linkage: Linkage type
    
    Returns:
        slots: [B, n_slots, D]
    """
    # Handle video input [B, T, N, D]
    if features.ndim == 4:
        B, T, N, D = features.shape
        features_flat = features.view(B * T, N, D)
        slots = cluster_slot_initialization(features_flat, n_slots, method, affinity, linkage)
        return slots.view(B, T, n_slots, D)
    
    B, N, D = features.shape
    device = features.device
    
    # Cluster each batch element
    slots_list = []
    for b in range(B):
        centroids = cluster_and_centroid(
            features[b], n_slots, method, affinity, linkage
        )
        slots_list.append(centroids)
    
    return torch.stack(slots_list, dim=0).to(device, dtype=features.dtype)


class ClusterFeatureInit(nn.Module):
    """Initialize slots using cluster centroids of feature patches."""
    
    def __init__(
        self,
        n_slots: int,
        dim: int,
        cluster_method: str = "agglomerative",
        affinity: str = "cosine",
        linkage: str = "average",
        init_mode: str = "first_frame",
        **kwargs,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        self.cluster_method = cluster_method
        self.affinity = affinity
        self.linkage = linkage
        self.init_mode = init_mode
        self.fallback = nn.Parameter(torch.randn(1, n_slots, dim) * dim**-0.5)

    def forward(self, batch_size: int, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if features is None:
            return self.fallback.expand(batch_size, -1, -1)
        
        if features.ndim == 4:  # Video: [B, T, N, D]
            if self.init_mode == "first_frame":
                return cluster_slot_initialization(
                    features[:, 0], self.n_slots, 
                    self.cluster_method, self.affinity, self.linkage
                )
            else:  # per_frame
                return cluster_slot_initialization(
                    features, self.n_slots,
                    self.cluster_method, self.affinity, self.linkage
                )
        
        # Image: [B, N, D]
        return cluster_slot_initialization(
            features, self.n_slots,
            self.cluster_method, self.affinity, self.linkage
        )
