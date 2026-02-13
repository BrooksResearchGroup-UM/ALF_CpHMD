#!/bin/bash
#SBATCH --job-name=nr_alf
#SBATCH --partition=ada5000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=nr_alf_%j.out

source ~/software/mambaforge/etc/profile.d/conda.sh
conda activate chm_12.9

cd "$SLURM_SUBMIT_DIR"

# Run ALF (solvation + patching already done)
python run.py alf
