#!/bin/bash
# SLURM parameters
#SBATCH --job-name=setup_50
#SBATCH -A brooks
#SBATCH -p gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=setup_50.out
#SBATCH --error=setup_50.err
#SBATCH --array=0-50
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G

setup=50
drxn=-1

module load charmm 

$CHARMM_EXEC -i minimize_setup.inp -o out/3c02_setup${setup}_build.log setup=$setup drxn=$drxn