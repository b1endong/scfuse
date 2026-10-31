#!/bin/bash
#SBATCH --job-name=scfuse_job
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
# usage: sbatch --job-name=NAME scripts/slurm/job.sbatch MODULE [args ...]
set -euo pipefail
source "${SCFUSE_ENV:-.venv}/bin/activate"
MODULE=$1; shift
python -m "$MODULE" "$@"
