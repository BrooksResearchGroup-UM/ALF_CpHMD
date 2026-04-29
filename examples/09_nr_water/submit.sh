#!/bin/bash
#SBATCH --job-name=nr_alf
#SBATCH --partition=ada5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=nr_alf_%j.out

set -eo pipefail

source ~/software/mambaforge/etc/profile.d/conda.sh
conda activate chm_12.9

cd "$SLURM_SUBMIT_DIR"

# Optional setup refresh if prep/ is missing:
# python run.py setup

bash ../_run_native_alf.sh cphmd_config.yaml
