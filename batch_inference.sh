#!/bin/bash

# Batch inference with metrics for YTVIS2021 validation videos

# # baseline inference
# python data/batch_inference.py \
#     --checkpoint /home/zhiyuan/Codes/exp_slotcontrast/checkpoints/baseline_ytvis_3iter/checkpoints/step=100000-v1.ckpt \
#     --config configs/inference/ytvis2021_baseline.yaml \
#     --data-dir /home/zhiyuan/Codes/exp_slotcontrast/data/ytvis2021_raw/valid \
#     --output-dir /home/zhiyuan/Codes/exp_slotcontrast/data/ytvis2021_inference \
#     --n-slots 7 \
#     --device cuda

# grounded_correspondence inference with config overrides
python data/batch_inference.py \
    --checkpoint /home/zhiyuan/Codes/exp_slotcontrast/checkpoints/gcv2_pos_emb_dino_exp/ytvis/2026-02-11-16-37-54_slotcontrast_gcv2_learnable_pos_emb_dinov3_large_block18_2/checkpoints/step=100000-v1.ckpt \
    --config configs/inference/ytvis2021_gc.yaml \
    --data-dir /home/zhiyuan/Codes/exp_slotcontrast/data/ytvis2021_raw/valid \
    --output-dir /home/zhiyuan/Codes/exp_slotcontrast/data/ytvis2021_inference \
    --n-slots 7 \
    --device cuda \
    --max-videos 10 \
    model.encoder.use_pos_embed=true \
    model.encoder.pos_embed.name=utils.LearnedPositionEmbed \
    model.encoder.backbone.features=vit_block18 \
    model.encoder.main_features_key=vit_block18 \
    globals.DINO_MODEL=vit_large_patch16_dinov3.lvd1689m \
    globals.NUM_PATCHES=1024 \
    globals.FEAT_DIM=1024 \
    dataset.train_pipeline.transforms.input_size=512 \
    dataset.val_pipeline.transforms.input_size=512 \


# To process specific videos for testing:
# python data/batch_inference.py --checkpoint <path> --config <path> --video-ids 00f88c4f0a 01c88b5b60

# To limit number of videos:
# python data/batch_inference.py --checkpoint <path> --config <path> --max-videos 10
