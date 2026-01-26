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

# grounded_correspondence inference
python data/batch_inference.py \
    --checkpoint /home/zhiyuan/Codes/exp_slotcontrast/checkpoints/GC_ytvis_3iter/checkpoints/step=100000-v1.ckpt \
    --config configs/inference/ytvis2021_gc.yaml \
    --data-dir /home/zhiyuan/Codes/exp_slotcontrast/data/ytvis2021_raw/valid \
    --output-dir /home/zhiyuan/Codes/exp_slotcontrast/data/ytvis2021_inference \
    --n-slots 7 \
    --device cuda

# To process specific videos for testing:
# python data/batch_inference.py --checkpoint <path> --config <path> --video-ids 00f88c4f0a 01c88b5b60

# To limit number of videos:
# python data/batch_inference.py --checkpoint <path> --config <path> --max-videos 10
