import argparse
import os
from pathlib import Path
from typing import Optional

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as tvt
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from slotcontrast.modules.encoders import TimmExtractor
from slotcontrast.modules.initializers import compute_neighbor_similarity
from slotcontrast.data.transforms import build_inference_transform


def prepare_video_from_frames(frame_dir: Path, transform_config, max_frames: Optional[int] = None):
    """Load video from JPEG frames directory, following inference.py's prepare_video pattern.
    
    Args:
        frame_dir: Directory containing {frame_num:05d}.jpg files
        transform_config: Transform configuration with input_size and dataset_type
        max_frames: Optional limit on number of frames
        
    Returns:
        inputs: Dict with 'video' and 'video_visualization' keys, matching inference.py format
    """
    # Load frames
    frame_files = sorted([f for f in frame_dir.glob("*.jpg")])
    if max_frames:
        frame_files = frame_files[:max_frames]
    
    if len(frame_files) == 0:
        return None
    
    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file).convert('RGB')
        img_array = np.array(img)
        frames.append(img_array)
    
    # Stack to video: [T, H, W, 3], then convert to tensor
    video = np.stack(frames, axis=0)
    video = torch.from_numpy(video).float() / 255.0
    
    # Prepare visualization: resize to input_size (following inference.py)
    video_vis = video.permute(0, 3, 1, 2)  # [T, 3, H, W]
    video_vis = tvt.Resize((transform_config.input_size, transform_config.input_size))(video_vis)
    video_vis = video_vis.permute(1, 0, 2, 3)  # [3, T, H, W]
    
    # Apply transforms for model input (following inference.py)
    if transform_config:
        tfs = build_inference_transform(transform_config)
        video = video.permute(3, 0, 1, 2)  # [3, T, H, W]
        video = tfs(video).permute(1, 0, 2, 3)  # [T, 3, H, W]
    else:
        video = video.permute(0, 3, 1, 2)  # [T, 3, H, W]
    
    # Add batch dimension (following inference.py)
    inputs = {
        "video": video.unsqueeze(0),  # [1, T, 3, H, W]
        "video_visualization": video_vis.unsqueeze(0)  # [1, 3, T, H, W]
    }
    return inputs


def extract_features_from_video(inputs: dict, model: TimmExtractor, device: str = 'cuda'):
    """Extract DINOv2 features from video inputs.
    
    Args:
        inputs: Dict with 'video' key containing [1, T, 3, H, W] tensor
        model: TimmExtractor instance
        device: Device to run on
        
    Returns:
        features: [T, N, D] where N is number of patches, D is feature dim
    """
    model.eval()
    with torch.no_grad():
        video = inputs["video"][0].to(device)  # [T, 3, H, W]
        
        # Extract features in batches
        batch_size = 8
        features_list = []
        for i in range(0, len(video), batch_size):
            batch = video[i:i+batch_size]
            feats = model(batch)
            if isinstance(feats, dict):
                feats = feats['vit_block12']
            features_list.append(feats.cpu())
        
        features = torch.cat(features_list, dim=0)
    
    return features


def compute_local_consistency_saliency(
    features: torch.Tensor,
    neighbor_radius: int = 1,
    saliency_alpha: float = 1.0,
) -> torch.Tensor:
    """Compute local consistency saliency from features.
    
    Args:
        features: [T, N, D] patch features
        neighbor_radius: Radius for neighbor similarity (1=3x3, 2=5x5)
        saliency_alpha: Weight for global similarity subtraction
        
    Returns:
        saliency: [T, N] saliency scores (higher = more salient)
    """
    T, N, D = features.shape
    patch_hw = int(N ** 0.5)
    assert patch_hw * patch_hw == N, f"Expected square patch grid, got {N} patches"
    
    # L2 normalize features
    features_norm = F.normalize(features, dim=-1)
    
    # Local similarity: mean cosine sim to spatial neighbors
    local_sim = compute_neighbor_similarity(features_norm, patch_hw, patch_hw, neighbor_radius)
    
    # Global similarity: cosine sim to global mean
    global_mean = F.normalize(features.mean(dim=1, keepdim=True), dim=-1)  # [T, 1, D]
    global_sim = (features_norm * global_mean).sum(dim=-1)  # [T, N]
    
    # High local_sim + low global_sim = interior of distinct object
    saliency = local_sim - saliency_alpha * global_sim
    
    return saliency


def visualize_dinov2_pca(
    features: torch.Tensor,
) -> np.ndarray:
    """DINOv2 PCA visualization: compute PCA across all frames.
    
    Args:
        features: [T, N, D] patch features
        
    Returns:
        pca_images: [T, h, w, 3] RGB images with PCA visualization
    """
    T, N, D = features.shape
    patch_hw = int(N ** 0.5)
    
    # Compute PCA across ALL frames (DINOv2 paper approach)
    features_flat = features.reshape(-1, D).numpy()
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features_flat)
    pca_features = pca_features.reshape(T, N, 3)
    
    # Normalize each component using percentile-based clipping for enhanced contrast
    for c in range(3):
        pca_c = pca_features[:, :, c]
        # Use 1st and 99th percentile for robust normalization
        p_low, p_high = np.percentile(pca_c, [1, 99])
        pca_c_clipped = np.clip(pca_c, p_low, p_high)
        pca_features[:, :, c] = (pca_c_clipped - p_low) / (p_high - p_low + 1e-8)
    
    pca_images = []
    for t in range(T):
        pca_frame = pca_features[t]  # [N, 3]
        
        # Reshape to spatial
        pca_map = pca_frame.reshape(patch_hw, patch_hw, 3)
        
        pca_images.append(pca_map)
    
    return np.array(pca_images)


def create_sidebyside_visualization(
    inputs: dict,
    pca_images: np.ndarray,
    saliency: torch.Tensor,
) -> np.ndarray:
    """Create side-by-side visualization: left=PCA, right=saliency heatmap.
    
    Args:
        inputs: Dict with 'video_visualization' key
        pca_images: [T, h, w, 3] PCA visualizations
        saliency: [T, N] saliency scores
        
    Returns:
        combined_frames: [T, H, W*2, 3] side-by-side frames
    """
    video_vis = inputs["video_visualization"][0].permute(1, 0, 2, 3)  # [T, 3, H, W]
    
    T, N = saliency.shape
    patch_hw = int(N ** 0.5)
    _, _, H, W = video_vis.shape
    
    combined_frames = []
    for t in range(T):
        # Left: PCA visualization
        pca_map = pca_images[t]  # [h, w, 3]
        pca_map_torch = torch.from_numpy(pca_map).permute(2, 0, 1).unsqueeze(0).float()
        pca_map_resized = F.interpolate(pca_map_torch, size=(H, W), mode='bilinear', align_corners=False)
        pca_map_resized = pca_map_resized.squeeze(0).permute(1, 2, 0).numpy()
        
        # Overlay PCA on original image
        frame_np = video_vis[t].permute(1, 2, 0).numpy()
        pca_vis = (0.3 * frame_np + 0.7 * pca_map_resized)
        pca_vis = np.clip(pca_vis, 0, 1)
        
        # Right: Saliency heatmap
        saliency_map = saliency[t].reshape(patch_hw, patch_hw).numpy()
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        # Resize saliency
        saliency_torch = torch.from_numpy(saliency_map).unsqueeze(0).unsqueeze(0).float()
        saliency_resized = F.interpolate(saliency_torch, size=(H, W), mode='bilinear', align_corners=False)
        saliency_resized = saliency_resized.squeeze().numpy()
        
        # Apply colormap
        cmap = plt.get_cmap('hot')
        saliency_colored = cmap(saliency_resized)[:, :, :3]
        
        # Concatenate horizontally
        combined = np.concatenate([pca_vis, saliency_colored], axis=1)  # [H, W*2, 3]
        combined = (combined * 255).astype(np.uint8)
        
        combined_frames.append(combined)
    
    return np.array(combined_frames)


def visualize_saliency(
    inputs: dict,
    saliency: torch.Tensor,
    alpha: float = 0.5,
) -> np.ndarray:
    """Create visualization with saliency heatmap overlaid on frames.
    
    Args:
        inputs: Dict with 'video_visualization' key [1, 3, T, H, W]
        saliency: [T, N] saliency scores
        alpha: Transparency of heatmap overlay
        
    Returns:
        vis_frames: [T, H, W, 3] uint8 frames with saliency overlay
    """
    # Get visualization frames: [1, 3, T, H, W] -> [T, 3, H, W]
    video_vis = inputs["video_visualization"][0].permute(1, 0, 2, 3)  # [T, 3, H, W]
    
    T, N = saliency.shape
    patch_hw = int(N ** 0.5)
    _, _, H, W = video_vis.shape
    
    vis_frames = []
    for t in range(T):
        # Reshape saliency to spatial map
        saliency_map = saliency[t].reshape(patch_hw, patch_hw)
        
        # Resize to frame size
        saliency_map = saliency_map.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]
        saliency_map = F.interpolate(saliency_map, size=(H, W), mode='bilinear', align_corners=False)
        saliency_map = saliency_map.squeeze().numpy()
        
        # Normalize to [0, 1]
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        # Apply colormap
        cmap = plt.get_cmap('hot')
        heatmap = cmap(saliency_map)[:, :, :3]  # RGB only
        
        # Convert frame to numpy
        frame_np = video_vis[t].permute(1, 2, 0).numpy()
        
        # Blend
        blended = (1 - alpha) * frame_np + alpha * heatmap
        blended = (blended * 255).astype(np.uint8)
        
        vis_frames.append(blended)
    
    return np.array(vis_frames)


def save_visualization(
    vis_frames: np.ndarray,
    output_path: Path,
    fps: int = 10,
):
    """Save visualization frames as individual images or single PNG.
    
    Args:
        vis_frames: [T, H, W, 3] uint8 frames
        output_path: Path to save (PNG for single frame, MP4 for video)
        fps: Frames per second (unused, kept for compatibility)
    """
    if len(vis_frames) == 1:
        # Save single frame directly
        imageio.imwrite(output_path, vis_frames[0])
    else:
        # Save frames directory
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(vis_frames):
            imageio.imwrite(frames_dir / f"{i:05d}.png", frame)


def process_video(
    video_dir: Path,
    model: TimmExtractor,
    output_dir: Path,
    transform_config,
    neighbor_radius: int = 1,
    saliency_alpha: float = 1.0,
    max_frames: Optional[int] = None,
    first_frame: bool = False,
    device: str = 'cuda',
):
    """Process single video: extract features, compute saliency, visualize."""
    video_id = video_dir.name
    
    # Prepare inputs (following inference.py pattern)
    if first_frame:
        max_frames = 1
    inputs = prepare_video_from_frames(video_dir, transform_config, max_frames)
    if inputs is None:
        print(f"Skipping {video_id}: no frames found")
        return
    
    T = inputs["video"].shape[1]
    print(f"Processing {video_id}: {T} frames")
    
    # Extract features
    features = extract_features_from_video(inputs, model, device)
    
    # Compute DINOv2 PCA
    pca_images = visualize_dinov2_pca(features)
    
    # Compute local consistency saliency
    saliency = compute_local_consistency_saliency(features, neighbor_radius, saliency_alpha)
    
    # Create side-by-side visualization: PCA (left) + Saliency (right)
    vis_frames = create_sidebyside_visualization(inputs, pca_images, saliency)
    
    # Save
    if first_frame:
        output_path = output_dir / f"{video_id}_pca_saliency.png"
    else:
        output_path = output_dir / f"{video_id}_pca_saliency.mp4"
    save_visualization(vis_frames, output_path)
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize DINOv2 local consistency saliency for YTVIS2021 videos")
    parser.add_argument(
        "--data-dir",
        default="/home/zhiyuan/Codes/exp_slotcontrast/data/ytvis2021_raw/valid/JPEGImages",
        help="Path to YTVIS JPEGImages directory containing video subdirectories"
    )
    parser.add_argument(
        "--output-dir",
        default="/home/zhiyuan/Codes/exp_slotcontrast/data/ytvis2021_saliency",
        help="Output directory for saliency visualizations"
    )
    parser.add_argument(
        "--model",
        default="vit_base_patch14_dinov2.lvd142m",
        help="DINOv2 model name"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input size for transforms (default: 224)"
    )
    parser.add_argument(
        "--neighbor-radius",
        type=int,
        default=1,
        help="Radius for neighbor similarity computation (1=3x3, 2=5x5)"
    )
    parser.add_argument(
        "--saliency-alpha",
        type=float,
        default=1.0,
        help="Weight for global similarity subtraction"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of videos to process"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames per video"
    )
    parser.add_argument(
        "--video-ids",
        nargs="+",
        help="Specific video IDs to process (e.g., 00f88c4f0a 01c88b5b60)"
    )
    parser.add_argument(
        "--first-frame",
        action="store_true",
        help="Process only the first frame of each video"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create transform config (like inference.py) using OmegaConf
    transform_config = OmegaConf.create({
        'dataset_type': 'video',
        'input_size': args.input_size
    })
    
    # Load DINOv2 model
    print(f"Loading {args.model}...")
    model = TimmExtractor(
        model=args.model,
        pretrained=True,
        frozen=True,
        features="vit_block12",
        model_kwargs={"dynamic_img_size": True},
    )
    model = model.to(args.device)
    model.eval()
    
    # Get video directories
    if args.video_ids:
        video_dirs = [data_dir / vid for vid in args.video_ids if (data_dir / vid).exists()]
    else:
        video_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if args.max_videos:
        video_dirs = video_dirs[:args.max_videos]
    
    print(f"Processing {len(video_dirs)} videos...")
    
    # Process videos
    for video_dir in tqdm(video_dirs):
        try:
            process_video(
                video_dir,
                model,
                output_dir,
                transform_config,
                args.neighbor_radius,
                args.saliency_alpha,
                args.max_frames,
                args.first_frame,
                args.device,
            )
        except Exception as e:
            print(f"Error processing {video_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
