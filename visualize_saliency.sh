#!/bin/bash

# Example script to visualize saliency maps for YTVIS2021 validation videos

# Process first 5 videos as test
python data/visualize_saliency.py \
    --data-dir /home/zhiyuan/Codes/exp_slotcontrast/data/ytvis2021_raw/valid/JPEGImages \
    --output-dir /home/zhiyuan/Codes/exp_slotcontrast/data/ytvis2021_saliency \
    --neighbor-radius 2 \
    --saliency-alpha 0.5 \
    --device cuda \
    --first-frame

# To process specific videos:
# python data/visualize_saliency.py --video-ids 00f88c4f0a 01c88b5b60

# To process all videos:
# python data/visualize_saliency.py --max-videos None
