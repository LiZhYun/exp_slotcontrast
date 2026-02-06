#!/bin/bash
#SBATCH --job-name=exp+slotcontrast_train
#SBATCH --account=project_462001066
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ============================================
# POST3R Training on CSC LUMI with ROCm
# ============================================

module load rocm
module load cray-python

# ===== Cache and Data Directories =====
# Base cache directory
export CACHE_DIR="/scratch/project_462001066/.cache"

# Pip cache
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export PIP_NO_CACHE_DIR=0

# HuggingFace cache directories
export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_METRICS_CACHE="${HF_HOME}/metrics"

# ModelScope cache
export MODELSCOPE_CACHE="${CACHE_DIR}/modelscope"
export MODELSCOPE_HUB_CACHE="${MODELSCOPE_CACHE}/hub"

# Torch extensions and cache
export TORCH_HOME="${CACHE_DIR}/torch"
export TORCH_EXTENSIONS_DIR="${CACHE_DIR}/torch_extensions"

# XDG cache (used by some libraries)
export XDG_CACHE_HOME="${CACHE_DIR}"

# Triton cache (for kernel compilation)
export TRITON_CACHE_DIR="${CACHE_DIR}/triton"

# WandB cache directories
export WANDB_DIR="${CACHE_DIR}/wandb"
export WANDB_CACHE_DIR="${CACHE_DIR}/wandb_cache"

# Create cache directories if they don't exist
mkdir -p "${PIP_CACHE_DIR}"
mkdir -p "${HF_HOME}"
mkdir -p "${HUGGINGFACE_HUB_CACHE}"
mkdir -p "${HF_DATASETS_CACHE}"
mkdir -p "${TRANSFORMERS_CACHE}"
mkdir -p "${HF_METRICS_CACHE}"
mkdir -p "${MODELSCOPE_CACHE}"
mkdir -p "${MODELSCOPE_HUB_CACHE}"
mkdir -p "${TORCH_HOME}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"
mkdir -p "${WANDB_DIR}"
mkdir -p "${WANDB_CACHE_DIR}"
mkdir -p "${TRITON_CACHE_DIR}"

echo "Cache directories created at: ${CACHE_DIR}"

source /scratch/project_462001066/slotcontrast/bin/activate

export PIP_CACHE_DIR=/scratch/project_462001066/.cache/pip
export HF_HOME=/scratch/project_462001066/.huggingface_cache
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_EXTENSIONS_DIR=/scratch/project_462001066/torch_extensions
export PYTHONPATH="${PWD}:$PYTHONPATH"
# ===== MIOpen cache (CRITICAL FIX) =====
export MIOPEN_USER_DB_PATH="/scratch/project_462001066/.cache/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="/scratch/project_462001066/.cache/miopen"
export MIOPEN_DISABLE_CACHE=0
DATA_DIR="/scratch/project_462001066/POST3R/data"
OUTPUT_DIR="/scratch/project_462001066/rethinkingocl/logs"

# First argument is the config file, rest are additional arguments
CONFIG_FILE=$1
shift

srun python slotcontrast/train.py "${CONFIG_FILE}" \
        "$@" \
        --data-dir ${DATA_DIR} \
        --log-dir ${OUTPUT_DIR} \
