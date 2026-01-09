#!/bin/bash

# =============================================================================
# Multi-Experiment Batch Submission for Triton H200
# =============================================================================
# Groups multiple experiments into single SLURM jobs for efficient GPU usage
#
# Usage: ./exp_slot_triton.sh
# =============================================================================

CONFIG_FILE="configs/slotcontrast/movi_e.yaml"
EXPS_PER_JOB=6  # Number of experiments per SLURM job (based on ~13GB per exp, 141GB total)

# Define experiment configurations
# Format: "experiment_name loss_ss init_name init_mode predictor neighbor_radius init_threshold match_threshold NUM_SLOTS"

CONFIGS=(
    # Add your experiment configurations here
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.0 2.0 15"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.1 1.5 15"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.1 1.0 15"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.1 0.5 15"

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
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.1 1.5 20"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.1 1.0 20"
    "per_frame_vari_greedy_local_mem_hungarian_w_contrast 0.5 GreedyFeatureInitV2 per_frame networks.HungarianMemoryMatcher 1 0.1 0.5 20"

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
)

SEED=42

# =============================================================================
# Group experiments and submit SLURM jobs
# =============================================================================

# Build experiment arguments
declare -a EXP_ARGS=()

for config in "${CONFIGS[@]}"; do
    read -r exp_name loss_ss init_name init_mode predictor neighbor_radius init_threshold match_threshold NUM_SLOTS <<< "$config"
    
    # Build override string for this experiment
    overrides="experiment_name=slotcontrast_${exp_name}"
    overrides+=" model.loss_weights.loss_ss=${loss_ss}"
    overrides+=" model.initializer.name=${init_name}"
    overrides+=" model.initializer.init_mode=${init_mode}"
    overrides+=" model.predictor.name=${predictor}"
    overrides+=" model.initializer.neighbor_radius=${neighbor_radius}"
    overrides+=" model.initializer.init_threshold=${init_threshold}"
    overrides+=" model.predictor.match_threshold=${match_threshold}"
    overrides+=" globals.NUM_SLOTS=${NUM_SLOTS}"
    overrides+=" seed=${SEED}"
    
    EXP_ARGS+=("$overrides")
done

# Group experiments into batches and submit
total_exps=${#EXP_ARGS[@]}
job_count=0

if [[ $total_exps -eq 0 ]]; then
    echo "No experiments configured. Add experiments to CONFIGS array."
    exit 1
fi

echo "=============================================="
echo "Submitting ${total_exps} experiments"
echo "Experiments per job: ${EXPS_PER_JOB}"
echo "=============================================="

for ((i=0; i<total_exps; i+=EXPS_PER_JOB)); do
    # Build command with experiments separated by ---
    cmd_args=""
    batch_size=0
    
    for ((j=i; j<i+EXPS_PER_JOB && j<total_exps; j++)); do
        if [[ -n "$cmd_args" ]]; then
            cmd_args+=" --- "
        fi
        cmd_args+="${EXP_ARGS[$j]}"
        ((batch_size++))
    done
    
    job_count=$((job_count + 1))
    echo ""
    echo "Job ${job_count}: Experiments $((i+1)) to $((i+batch_size))"
    
    # Submit the batch job
    sbatch --job-name="sc_batch_${job_count}" triton_slurm_multi.sh "${CONFIG_FILE}" ${cmd_args}
done

echo ""
echo "=============================================="
echo "Submitted ${job_count} SLURM jobs for ${total_exps} experiments"
echo "=============================================="
