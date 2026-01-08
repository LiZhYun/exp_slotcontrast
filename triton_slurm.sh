#!/bin/bash
#SBATCH --job-name=slotcontrast_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=0-10:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

#--gres=gpu:h200_2g.35gb:1
#--partition=gpu-h200-35g-ia

module load mamba
module load triton-dev/2025.1-gcc
module load gcc/13.3.0
module load cuda/12.6.2
export HF_HOME=/$WRKDIR/.huggingface_cache
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_EXTENSIONS_DIR=$WRKDIR/torch_extensions

source activate slotcontrast

export PYTHONPATH="${PWD}:$PYTHONPATH"

DATA_DIR="/scratch/work/liz23/slotcontrast/data"
OUTPUT_DIR="/scratch/work/liz23/slotcontrast/logs"

# First argument is the config file, rest are additional arguments
CONFIG_FILE=$1
shift

srun python slotcontrast/train.py "${CONFIG_FILE}" \
        "$@" \
        --data-dir ${DATA_DIR} \
        --log-dir ${OUTPUT_DIR} \
