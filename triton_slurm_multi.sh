#!/bin/bash
#SBATCH --job-name=slotcontrast_multi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --output=logs/train_multi_%j.out
#SBATCH --error=logs/train_multi_%j.err

# =============================================================================
# Multi-Experiment Training on Single H200 GPU
# =============================================================================
# This script runs multiple experiments in parallel on a single H200 GPU (141GB)
# Each experiment uses ~13GB, so we can safely run 6-8 experiments simultaneously
#
# Usage:
#   sbatch triton_slurm_multi.sh <config_file> <exp1_overrides> --- <exp2_overrides> --- ...
#
# Example:
#   sbatch triton_slurm_multi.sh configs/slotcontrast/movi_e.yaml \
#       "experiment_name=exp1 model.loss_weights.loss_ss=0.5" --- \
#       "experiment_name=exp2 model.loss_weights.loss_ss=1.0" --- \
#       "experiment_name=exp3 model.loss_weights.loss_ss=0.3"
# =============================================================================

module load mamba
module load triton-dev/2025.1-gcc
module load gcc/13.3.0
module load cuda/12.6.2

export HF_HOME=$WRKDIR/.huggingface_cache
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_EXTENSIONS_DIR=$WRKDIR/torch_extensions

source activate slotcontrast

export PYTHONPATH="${PWD}:$PYTHONPATH"

DATA_DIR="/scratch/work/liz23/slotcontrast/data"
OUTPUT_DIR="/scratch/work/liz23/slotcontrast/logs"

# =============================================================================
# Configuration
# =============================================================================
MAX_PARALLEL_EXPS=6          # Max experiments to run in parallel (conservative)
GPU_MEMORY_FRACTION=0.15     # Memory fraction per experiment (~21GB each)
NUM_WORKERS_PER_EXP=2        # Reduced workers when running multiple experiments

# =============================================================================
# Parse arguments: config_file followed by experiment overrides separated by ---
# =============================================================================
CONFIG_FILE=$1
shift

# Split remaining arguments by "---" delimiter
declare -a EXPERIMENTS=()
current_exp=""

for arg in "$@"; do
    if [[ "$arg" == "---" ]]; then
        if [[ -n "$current_exp" ]]; then
            EXPERIMENTS+=("$current_exp")
        fi
        current_exp=""
    else
        if [[ -n "$current_exp" ]]; then
            current_exp="$current_exp $arg"
        else
            current_exp="$arg"
        fi
    fi
done

# Add the last experiment if exists
if [[ -n "$current_exp" ]]; then
    EXPERIMENTS+=("$current_exp")
fi

NUM_EXPERIMENTS=${#EXPERIMENTS[@]}

if [[ $NUM_EXPERIMENTS -eq 0 ]]; then
    echo "Error: No experiments provided"
    echo "Usage: sbatch triton_slurm_multi.sh <config> <exp1_args> --- <exp2_args> --- ..."
    exit 1
fi

echo "=============================================="
echo "Multi-Experiment Training on H200 GPU"
echo "=============================================="
echo "Config: ${CONFIG_FILE}"
echo "Number of experiments: ${NUM_EXPERIMENTS}"
echo "GPU memory fraction per exp: ${GPU_MEMORY_FRACTION}"
echo "Workers per experiment: ${NUM_WORKERS_PER_EXP}"
echo "=============================================="

# =============================================================================
# Launch experiments in parallel with GPU memory limiting
# =============================================================================
declare -a PIDS=()

for i in "${!EXPERIMENTS[@]}"; do
    exp_args="${EXPERIMENTS[$i]}"
    exp_idx=$((i + 1))
    
    echo ""
    echo "[Experiment ${exp_idx}/${NUM_EXPERIMENTS}] Starting..."
    echo "  Args: ${exp_args}"
    
    # Set per-process GPU memory limit via environment variable
    # PyTorch will respect PYTORCH_CUDA_ALLOC_CONF for memory management
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
    
    # Launch training in background with memory fraction limit
    (
        python slotcontrast/train.py "${CONFIG_FILE}" \
            ${exp_args} \
            --data-dir "${DATA_DIR}" \
            --log-dir "${OUTPUT_DIR}" \
            --gpu-memory-fraction ${GPU_MEMORY_FRACTION} \
            "dataset.num_workers=${NUM_WORKERS_PER_EXP}" \
            "dataset.num_val_workers=1"
    ) &
    
    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}"
    
    # Small delay to avoid race conditions during initialization
    sleep 5
    
    # If we've reached max parallel experiments, wait for one to finish
    if [[ ${#PIDS[@]} -ge $MAX_PARALLEL_EXPS ]]; then
        echo ""
        echo "Reached max parallel experiments (${MAX_PARALLEL_EXPS}). Waiting for one to complete..."
        wait -n  # Wait for any one process to complete
        
        # Remove completed PIDs from array
        new_pids=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        PIDS=("${new_pids[@]}")
    fi
done

echo ""
echo "=============================================="
echo "All experiments launched. Waiting for completion..."
echo "Active PIDs: ${PIDS[*]}"
echo "=============================================="

# Wait for all remaining experiments to complete
wait

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
