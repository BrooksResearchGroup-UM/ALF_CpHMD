#!/bin/bash
#SBATCH --job-name=tyr_water
#SBATCH --partition=gpu2080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=tyr_water_%j.out

# Activate conda environment
source ~/software/mambaforge/etc/profile.d/conda.sh
conda activate chm_12.9

cd "$SLURM_SUBMIT_DIR"

# Run full workflow: build → solvate → patch → alf
python run.py all
