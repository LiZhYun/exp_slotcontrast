#!/bin/bash

# ytvis2021.yaml, movi_c.yaml, movi_e.yaml
CONFIG_FILE="configs/slotcontrast/movi_e.yaml"


CONFIGS=(

    # ytvis2021
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 40"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 41"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 42"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 43"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 44"
    

    # movi_c
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 40"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 41"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44"

    # movi_e
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 40"
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 41"
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42"
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43"
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44"


    )
    
for config in "${CONFIGS[@]}"; do
    read -r exp_name init_name init_mode predictor neighbor_radius SEED <<< "$config"

    echo "Submitting: $exp_name with config: $CONFIG_FILE"
    
    sbatch --job-name="sc_${exp_name}" triton_slurm.sh "${CONFIG_FILE}" \
        "experiment_name=slotcontrast_${exp_name}" \
        "model.initializer.name=${init_name}" \
        "model.initializer.init_mode=${init_mode}" \
        "model.initializer.neighbor_radius=${neighbor_radius}" \
        "model.predictor.name=${predictor}" \
        "seed=${SEED}"
done

echo "All jobs submitted!"
