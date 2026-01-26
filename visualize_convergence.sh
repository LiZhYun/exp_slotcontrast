#!/bin/bash

# Visualize slot attention convergence across iterations

python data/visualize_convergence.py \
    --data-dir /home/zhiyuan/Codes/exp_slotcontrast/data/ytvis2021_raw/valid/JPEGImages \
    --output-dir /home/zhiyuan/Codes/exp_slotcontrast/data/ytvis2021_convergence \
    --standard-checkpoint /home/zhiyuan/Codes/exp_slotcontrast/checkpoints/baseline_ytvis_3iter/checkpoints/step=100000-v1.ckpt \
    --standard-config configs/inference/ytvis2021_baseline.yaml \
    --grounded-checkpoint /home/zhiyuan/Codes/exp_slotcontrast/checkpoints/greedy_first_ytvis_1iter/checkpoints/step=100000-v1.ckpt \
    --grounded-config configs/inference/ytvis2021_greedy_first.yaml \
    --n-slots-standard 7 \
    --n-slots-grounded 7 \
    --max-iters 3 \
    --device cuda
