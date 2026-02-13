#!/bin/bash
#SBATCH --job-name=glu_water
#SBATCH --partition=gpu2080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=glu_water_%j.out

# Activate conda environment
source ~/software/mambaforge/etc/profile.d/conda.sh
conda activate chm_12.9

cd "$SLURM_SUBMIT_DIR"

# Step 1: Solvate + patch (serial, no GPU needed for setup)
# python run.py solvate
# python run.py patch

# Step 2: Run ALF (requires GPU)
python run.py alf

# Or run everything in one go:
# python run.py all
