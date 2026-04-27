#!/bin/bash
#SBATCH --job-name=hsp_prod
#SBATCH --partition=ada5000
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=hsp_prod_%x_%j.out

set -eo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch submit_production_repeat.sh sim_N [production_config.yaml]" >&2
    exit 2
fi

REPEAT_DIR="$1"
CONFIG_PATH="${2:-production_config.yaml}"

source ~/software/mambaforge/etc/profile.d/conda.sh
conda activate chm_12.9
set -u

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$(cd ../.. && pwd):${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export PMIX_MCA_gds=hash
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_mtl=^ofi
mpi_env=(
    -x OMP_NUM_THREADS=1
    -x PYTHONPATH
    -x PMIX_MCA_gds
    -x OMPI_MCA_pml
    -x OMPI_MCA_btl
    -x OMPI_MCA_mtl
)
for name in CPHMD_DEBUG_REX_STATE CPHMD_DEBUG_TIMINGS CPHMD_DEBUG_PRODUCTION_BIAS; do
    if [[ -n "${!name:-}" ]]; then
        mpi_env+=(-x "$name")
    fi
done

"$CONDA_PREFIX/bin/python" -m cphmd.cli.main init \
    -c "$CONFIG_PATH" \
    --run-dir "$REPEAT_DIR"

mpirun -np "$SLURM_NTASKS" \
    --bind-to none --map-by slot \
    --mca pml ob1 --mca btl tcp,self --mca mtl ^ofi \
    "${mpi_env[@]}" \
    "$CONDA_PREFIX/bin/python" -m cphmd.cli.main run \
    -c "$CONFIG_PATH" \
    --run-dir "$REPEAT_DIR"
