#!/bin/bash

# Dataset configuration - change this to switch datasets
# Available configs: configs/slotcontrast/ytvis2021.yaml, configs/slotcontrast/movi_c.yaml, configs/slotcontrast/movi_e.yaml
CONFIG_FILE="configs/slotcontrast/movi_e.yaml"

# Define experiment configurations
# Format: "experiment_name use_ttt3r use_gated use_gated_predictor use_ttt use_gru loss_ss loss_cycle window_size"

INIT_TH=(0.0 0.1 0.2 0.3)
MATCH_TH=(0.5 1.0 1.5 2.0)

CONFIGS=(

    # # ytvis2021
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.0 2.0 7"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.0 1.5 7"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.0 1.0 7"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.0 0.5 7"

    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.1 2.0 7"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.1 1.5 7"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.1 1.0 7"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.1 0.5 7"

    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.2 2.0 7"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.2 1.5 7"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.2 1.0 7"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.2 0.5 7"

    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.3 2.0 7"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.3 1.5 7"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.3 1.0 7"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.3 0.5 7"

    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.0 2.0 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.0 1.5 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.0 1.0 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.0 0.5 20"

    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.1 2.0 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.1 1.5 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.1 1.0 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.1 0.5 20"

    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.2 2.0 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.2 1.5 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.2 1.0 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.2 0.5 20"

    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.3 2.0 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.3 1.5 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.3 1.0 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 2 0.3 0.5 20"
    

    # movi_e
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.0 2.0 15"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.0 1.5 15"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.0 1.0 15"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.0 0.5 15"

    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.2 2.0 15"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.2 1.5 15"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.2 1.0 15"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.2 0.5 15"

    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.3 2.0 15"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.3 1.5 15"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.3 1.0 15"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.3 0.5 15"

    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.4 2.0 15"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.4 1.5 15"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.4 1.0 15"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.4 0.5 15"

    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.0 2.0 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.0 1.5 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.0 1.0 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.0 0.5 20"

    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.2 2.0 20"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.2 1.5 20"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.2 1.0 20"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.2 0.5 20"

    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.3 2.0 20"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.3 1.5 20"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.3 1.0 20"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.3 0.5 20"

    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.4 2.0 20"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.4 1.5 20"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.4 1.0 20"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.4 0.5 20"

    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.5 2.0 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.5 1.5 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.5 1.0 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.5 0.5 20"

    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.8 2.0 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.8 1.5 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.8 1.0 20"
    # "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.8 0.5 20"

    )
# n_object 都是很低 且 先上升一点 再下降的趋势 identity ratio一直很高 且ari mbo的趋势也和n_object类似 在n_object高的时候表现最好
SEED=42
    
for config in "${CONFIGS[@]}"; do
    read -r exp_name loss_ss init_name init_mode predictor neighbor_radius init_threshold match_threshold NUM_SLOTS <<< "$config"

    echo "Submitting: $exp_name with config: $CONFIG_FILE"
    
    sbatch --job-name="sc_${exp_name}" mahti_slurm.sh "${CONFIG_FILE}" \
        "experiment_name=slotcontrast_${exp_name}" \
        "model.loss_weights.loss_ss=${loss_ss}" \
        "model.initializer.name=${init_name}" \
        "model.initializer.init_mode=${init_mode}" \
        "model.initializer.neighbor_radius=${neighbor_radius}" \
        "model.initializer.init_threshold=${init_threshold}" \
        "model.predictor.name=${predictor}" \
        "model.predictor.match_threshold=${match_threshold}" \
        "globals.NUM_SLOTS=${NUM_SLOTS}" \
        "seed=${SEED}"
done

echo "All jobs submitted!"
