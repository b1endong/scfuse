#!/bin/bash
#SBATCH --job-name=scfuse
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
# usage: sbatch scripts/slurm/train.sbatch CONFIG [SEED] [key.sub=value ...]
#        sbatch --array=0-4%3 scripts/slurm/train.sbatch CONFIG [key.sub=value ...]
set -euo pipefail
source "${SCFUSE_ENV:-.venv}/bin/activate"
CONFIG=$1; shift
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SEEDS=(16 32 64 128 256)
  SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
else
  SEED=$1; shift
fi
python -m scfuse.train --config "$CONFIG" --seed "$SEED" --resume "$@"
