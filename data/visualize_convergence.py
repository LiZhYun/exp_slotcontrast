import argparse
import os
from pathlib import Path
from typing import Optional, Dict, List

import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as tvt
from tqdm import tqdm
from omegaconf import OmegaConf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from slotcontrast import configuration, models
from slotcontrast.data.transforms import build_inference_transform
from slotcontrast.visualizations import color_map, draw_segmentation_masks_on_image


def load_model_from_checkpoint(checkpoint_path: str, config_path: str, device: str = 'cuda'):
    config = configuration.load_config(config_path)
    model = models.build(config.model, config.optimizer)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    model.to(device)
    return model, config


def prepare_first_frame(frame_dir: Path, transform_config) -> Optional[Dict]:
    frame_files = sorted([f for f in frame_dir.glob("*.jpg")])
    if len(frame_files) == 0:
        return None
    
    img = Image.open(frame_files[0]).convert('RGB')
    img_array = np.array(img)
    video = torch.from_numpy(img_array).unsqueeze(0).float() / 255.0
    
    video_vis = video.permute(0, 3, 1, 2)
    video_vis = tvt.Resize((transform_config.input_size, transform_config.input_size))(video_vis)
    video_vis = video_vis.permute(1, 0, 2, 3)
    
    if transform_config:
        tfs = build_inference_transform(transform_config)
        video = video.permute(3, 0, 1, 2)
        video = tfs(video).permute(1, 0, 2, 3)
    else:
        video = video.permute(0, 3, 1, 2)
    
    inputs = {
        "video": video.unsqueeze(0),
        "video_visualization": video_vis.unsqueeze(0)
    }
    return inputs


def extract_intermediate_slots(model, inputs: Dict, n_iters: int, device: str = 'cuda') -> List[Dict]:
    model.eval()
    video = inputs["video"].to(device)
    B, T, C, H, W = video.shape
    
    intermediate_outputs = []
    
    with torch.no_grad():
        # Extract features using encoder (handles MapOverTime wrapper)
        encoder_output = model.encoder(video, None)
        features_main = encoder_output["features"]  # [B, T, N, D]
        
        # Initialize slots for first frame
        if model.use_backbone_features and "backbone_features" in encoder_output:
            backbone_features = encoder_output["backbone_features"]
            init_output = model.initializer(batch_size=B, features=backbone_features[:, 0])
            encoder_module = getattr(model.encoder, 'module', model.encoder)
            if isinstance(init_output, tuple):
                raw_slots, _, _ = init_output
                slots = encoder_module.output_transform(raw_slots) if encoder_module.output_transform else raw_slots
            else:
                slots = encoder_module.output_transform(init_output) if encoder_module.output_transform else init_output
        else:
            init_output = model.initializer(batch_size=B, features=features_main[:, 0])
            if isinstance(init_output, tuple):
                slots, _, _ = init_output
            else:
                slots = init_output
        
        # Get grouper (unwrap ScanOverTime -> LatentProcessor -> corrector)
        processor = model.processor
        if hasattr(processor, 'module'):
            processor = processor.module
        if hasattr(processor, 'corrector'):
            grouper = processor.corrector
        else:
            grouper = processor
        
        features_t = features_main[:, 0]  # [B, N, D]
        
        features_norm = grouper.norm_features(features_t)
        keys = grouper.to_k(features_norm)
        values = grouper.to_v(features_norm)
        
        for iter_idx in range(n_iters):
            slots, pre_norm_attn = grouper.step(slots, keys, values)
            
            # Add time dimension for video decoder: [B, n_slots, D] -> [B, T, n_slots, D]
            slots_for_decoder = slots.unsqueeze(1)
            decoder_output = model.decoder(slots_for_decoder)
            masks = decoder_output['masks'][:, 0]  # Remove time dim: [B, T, n_slots, N] -> [B, n_slots, N]
            
            intermediate_outputs.append({
                'iteration': iter_idx + 1,
                'slots': slots.cpu(),
                'masks': masks.cpu(),
                'attention': pre_norm_attn.cpu()
            })
    
    return intermediate_outputs


def visualize_masks_on_frame(frame: torch.Tensor, masks: torch.Tensor, n_slots: int, alpha: float = 0.5) -> np.ndarray:
    _, H, W = frame.shape
    N = masks.shape[-1]
    patch_hw = int(N ** 0.5)
    
    masks_spatial = masks[0].reshape(n_slots, patch_hw, patch_hw)
    masks_upsampled = F.interpolate(
        masks_spatial.unsqueeze(0),
        size=(H, W),
        mode='bilinear',
        align_corners=False
    )[0]
    
    hard_masks = torch.argmax(masks_upsampled, dim=0)
    masks_bool = torch.stack([hard_masks == i for i in range(n_slots)])
    
    frame_uint8 = (frame * 255).clamp(0, 255).byte()
    cmap = color_map(n_slots)
    
    vis_frame = draw_segmentation_masks_on_image(
        frame_uint8,
        masks_bool,
        alpha=alpha,
        colors=cmap
    )
    
    return vis_frame.permute(1, 2, 0).cpu().numpy()


def create_convergence_figure(inputs: Dict, standard_outputs: List[Dict], grounded_outputs: List[Dict],
                              n_slots_standard: int, n_slots_grounded: int, max_iters: int = 3):
    frame = inputs["video_visualization"][0, :, 0]
    
    # Create standard row (3 iterations)
    standard_row = []
    for iter_idx in range(max_iters):
        masks = standard_outputs[min(iter_idx, len(standard_outputs) - 1)]['masks']
        vis = visualize_masks_on_frame(frame, masks, n_slots_standard, alpha=0.6)
        standard_row.append(vis)
    standard_img = np.concatenate(standard_row, axis=1)
    
    # Create grounded image (only 1st iteration)
    masks = grounded_outputs[0]['masks']
    grounded_img = visualize_masks_on_frame(frame, masks, n_slots_grounded, alpha=0.6)
    
    return standard_img, grounded_img


def process_video(video_dir: Path, standard_model, grounded_model, output_dir: Path, transform_config,
                 n_slots_standard: int, n_slots_grounded: int, max_iters: int = 3, device: str = 'cuda'):
    video_id = video_dir.name
    
    inputs = prepare_first_frame(video_dir, transform_config)
    if inputs is None:
        print(f"Skipping {video_id}: no frames found")
        return
    
    print(f"Processing {video_id}: first frame only")
    
    standard_outputs = extract_intermediate_slots(standard_model, inputs, max_iters, device)
    grounded_outputs = extract_intermediate_slots(grounded_model, inputs, 1, device)  # Only 1 iteration
    
    standard_img, grounded_img = create_convergence_figure(inputs, standard_outputs, grounded_outputs, 
                                                            n_slots_standard, n_slots_grounded, max_iters)
    
    # Save separate images
    standard_path = output_dir / f"{video_id}_standard.png"
    grounded_path = output_dir / f"{video_id}_grounded.png"
    
    imageio.imwrite(standard_path, standard_img)
    imageio.imwrite(grounded_path, grounded_img)
    
    print(f"  Saved: {standard_path}")
    print(f"  Saved: {grounded_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize slot attention convergence")
    parser.add_argument("--data-dir", default="/home/zhiyuan/Codes/exp_slotcontrast/data/ytvis2021_raw/valid/JPEGImages")
    parser.add_argument("--output-dir", default="/home/zhiyuan/Codes/exp_slotcontrast/data/ytvis2021_convergence")
    parser.add_argument("--standard-checkpoint", default="/home/zhiyuan/Codes/exp_slotcontrast/checkpoints/baseline_ytvis_3iter/checkpoints/step=100000-v1.ckpt")
    parser.add_argument("--standard-config", default="configs/slotcontrast/ytvis2021.yaml")
    parser.add_argument("--grounded-checkpoint", default="/home/zhiyuan/Codes/exp_slotcontrast/checkpoints/t3slotcontrast_b4.ckpt")
    parser.add_argument("--grounded-config", default="configs/slotcontrast/ytvis2021.yaml")
    parser.add_argument("--n-slots-standard", type=int, default=7)
    parser.add_argument("--n-slots-grounded", type=int, default=7)
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--video-ids", nargs="+")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transform_config = OmegaConf.create({
        'dataset_type': 'video',
        'input_size': args.input_size
    })
    
    print(f"Loading standard model from {args.standard_checkpoint}...")
    standard_model, _ = load_model_from_checkpoint(args.standard_checkpoint, args.standard_config, args.device)
    standard_model.initializer.n_slots = args.n_slots_standard
    
    print(f"Loading grounded model from {args.grounded_checkpoint}...")
    grounded_model, _ = load_model_from_checkpoint(args.grounded_checkpoint, args.grounded_config, args.device)
    grounded_model.initializer.n_slots = args.n_slots_grounded
    
    if args.video_ids:
        video_dirs = [data_dir / vid for vid in args.video_ids if (data_dir / vid).exists()]
    else:
        video_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if args.max_videos:
        video_dirs = video_dirs[:args.max_videos]
    
    print(f"\nProcessing {len(video_dirs)} videos...")
    
    for video_dir in tqdm(video_dirs):
        try:
            process_video(video_dir, standard_model, grounded_model, output_dir, transform_config,
                         args.n_slots_standard, args.n_slots_grounded, args.max_iters, args.device)
        except Exception as e:
            print(f"Error processing {video_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
