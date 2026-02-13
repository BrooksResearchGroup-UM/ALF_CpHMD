#!/bin/bash
#SBATCH --job-name=lysozyme_cphmd
#SBATCH --partition=ada6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=lysozyme_%j.out

# Activate conda environment
source ~/software/mambaforge/etc/profile.d/conda.sh
conda activate chm_12.9

cd "$SLURM_SUBMIT_DIR"

# Run ALF (solvation + patching already done)
python run.py alf
