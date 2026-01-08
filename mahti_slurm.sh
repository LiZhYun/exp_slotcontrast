#!/bin/bash
#SBATCH --job-name=slotcontrast_train
#SBATCH --account=project_2017204
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpumedium
#SBATCH --time=0-20:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

#--gres=gpu:h200_2g.35gb:1
#--partition=gpu-h200-35g-ia

module load python-data
export HF_HOME=/scratch/project_2017204/.huggingface_cache
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_EXTENSIONS_DIR=/scratch/project_2017204/torch_extensions

source /projappl/project_2017204/slotcontrast/bin/activate

export PYTHONPATH="${PWD}:$PYTHONPATH"

DATA_DIR="/scratch/project_2017204/exp_sc_data/"
OUTPUT_DIR="/projappl/project_2017204/exp_slotcontrast/logs"

# First argument is the config file, rest are additional arguments
CONFIG_FILE=$1
shift

srun python slotcontrast/train.py "${CONFIG_FILE}" \
        "$@" \
        --data-dir ${DATA_DIR} \
        --log-dir ${OUTPUT_DIR} \
