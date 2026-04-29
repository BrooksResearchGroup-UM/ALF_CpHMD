#!/bin/bash
#SBATCH --job-name=lys_mpi
#SBATCH --partition=ada5000
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=lys_mpi_%j.out

set -eo pipefail

source ~/software/mambaforge/etc/profile.d/conda.sh
conda activate chm_12.9

cd "$SLURM_SUBMIT_DIR"

# Optional setup refresh if prep/ is missing:
# python run.py setup

bash ../_run_native_alf.sh cphmd_config.yaml
