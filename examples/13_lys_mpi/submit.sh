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

# Activate conda environment
source ~/software/mambaforge/etc/profile.d/conda.sh
conda activate chm_12.9

cd "$SLURM_SUBMIT_DIR"

# Step 1: Solvate + patch (serial, no GPU needed)
# Uncomment if prep/ does not exist yet:
# python run.py solvate
# python run.py patch

# Step 2: Initialize then run ALF with MPI (one replica per GPU)
# nreps is auto-detected from MPI communicator size (= ntasks)
# Per-rank output goes to python_log_rank{0..4}.out
cphmd init -c cphmd_config.yaml
mpirun -np "$SLURM_NTASKS" \
    --bind-to none --map-by slot \
    --mca pml ob1 --mca btl tcp,self \
    -x OMP_NUM_THREADS=1 \
    cphmd run -c cphmd_config.yaml
