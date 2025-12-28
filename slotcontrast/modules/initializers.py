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


def greedy_slot_initialization(
    features: torch.Tensor,
    n_slots: int,
    temperature: float = 0.1,
    saliency_mode: str = "norm",
    aggregate: bool = False,
    aggregate_threshold: float = 0.0,
    neighbor_radius: int = 1,
    saliency_smoothing: int = 0,
) -> torch.Tensor:
    # Handle video input [B, T, N, D]
    if features.ndim == 4:
        B, T, N, D = features.shape
        features_flat = features.view(B * T, N, D)
        slots = greedy_slot_initialization(
            features_flat, n_slots, temperature, saliency_mode, 
            aggregate, aggregate_threshold, neighbor_radius, saliency_smoothing
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
        saliency = local_sim - global_sim
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
                 aggregate: bool = False, aggregate_threshold: float = 0.5,
                 neighbor_radius: int = 1, saliency_smoothing: int = 0, **kwargs):
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
        self.fallback = nn.Parameter(torch.randn(1, n_slots, dim) * dim**-0.5)

    def forward(self, batch_size: int, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if features is None:
            return self.fallback.expand(batch_size, -1, -1)
        
        if features.ndim == 4:
            if self.init_mode == "first_frame":
                first_frame_feat = features[:, 0]
                return greedy_slot_initialization(
                    first_frame_feat, self.n_slots, self.temperature, 
                    self.saliency_mode, self.aggregate, self.aggregate_threshold,
                    self.neighbor_radius, self.saliency_smoothing
                )
            else:
                return greedy_slot_initialization(
                    features, self.n_slots, self.temperature, 
                    self.saliency_mode, self.aggregate, self.aggregate_threshold,
                    self.neighbor_radius, self.saliency_smoothing
                )
        
        return greedy_slot_initialization(
            features, self.n_slots, self.temperature, 
            self.saliency_mode, self.aggregate, self.aggregate_threshold,
            self.neighbor_radius, self.saliency_smoothing
        )


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
