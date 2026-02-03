#!/bin/bash

# Extract MOVI-C validation data
echo "Extracting MOVI-C validation data..."
python data/extract_movi_validation.py \
    --dataset movi_c \
    --data-dir /home/zhiyuan/Codes/exp_slotcontrast/data

# # Extract MOVI-D validation data
# echo "Extracting MOVI-D validation data..."
# python data/extract_movi_validation.py \
#     --dataset movi_d \
#     --data-dir /home/zhiyuan/Codes/exp_slotcontrast/data

# # Extract MOVI-E validation data
# echo "Extracting MOVI-E validation data..."
# python data/extract_movi_validation.py \
#     --dataset movi_e \
#     --data-dir /home/zhiyuan/Codes/exp_slotcontrast/data

# echo "Done! Extracted data to:"
# echo "  - data/movi_c_raw/valid/JPEGImages/"
# echo "  - data/movi_d_raw/valid/JPEGImages/"
# echo "  - data/movi_e_raw/valid/JPEGImages/"

# To test with limited videos:
# python data/extract_movi_validation.py --dataset movi_d --max-videos 10
