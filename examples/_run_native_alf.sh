#!/bin/bash
set -eo pipefail

CONFIG_PATH="${1:-cphmd_config.yaml}"
NREPS="${SLURM_NTASKS:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PMIX_MCA_gds="${PMIX_MCA_gds:-hash}"
export OMPI_MCA_pml="${OMPI_MCA_pml:-ob1}"
export OMPI_MCA_btl="${OMPI_MCA_btl:-self,tcp}"
export OMPI_MCA_mtl="${OMPI_MCA_mtl:-^ofi}"

python_cmd="${CONDA_PREFIX:+$CONDA_PREFIX/bin/}python"

"$python_cmd" -m cphmd.cli.main init -c "$CONFIG_PATH"

mpirun -np "$NREPS" \
    --bind-to none --map-by slot \
    --mca pml ob1 --mca btl tcp,self --mca mtl ^ofi \
    -x OMP_NUM_THREADS \
    -x PMIX_MCA_gds \
    -x OMPI_MCA_pml \
    -x OMPI_MCA_btl \
    -x OMPI_MCA_mtl \
    "$python_cmd" -m cphmd.cli.main run -c "$CONFIG_PATH"
