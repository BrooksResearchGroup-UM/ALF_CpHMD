#!/bin/bash
#SBATCH --job-name=asp_single
#SBATCH --partition=gpu2080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=asp_single_%j.out

# Activate conda environment
source ~/software/mambaforge/etc/profile.d/conda.sh
conda activate chm_12.9

cd "$SLURM_SUBMIT_DIR"

# Run ALF (solvation + patching already done)
python run.py alf
