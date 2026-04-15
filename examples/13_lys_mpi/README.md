# 13 — Lysine (LYS) in Water (MPI Multi-GPU with CpHMD)

Same system as `04_lys_water` but using **MPI parallel replicas** across
multiple GPUs with **CpHMD pH coupling enabled**.

Each MPI rank runs one replica on its own GPU; `nreps` is auto-detected
from the MPI communicator size. The 5 replicas fan out around the
effective pH with `delta_pKa` spacing (1.0 in Phase 1, 0.5 in Phase 2,
0.25 in Phase 3).

This example demonstrates the recommended production setup: `mpirun` with
`--ntasks=N --gpus-per-task=1` for N parallel pH replicas.

## Prerequisites

- pyCHARMM environment with mpi4py (`conda activate chm_12.9`)
- Multiple GPUs (one per replica; default 5 in `submit.sh`)
- Pre-built `solvated/prep/` files (included, or run `python run.py solvate && python run.py patch`)

## Quick Start

```bash
# Run on SLURM cluster (5 GPUs, one replica each)
sbatch submit.sh

# Or run locally with 2 GPUs
mpirun -np 2 cphmd run alf -c cphmd_config.yaml --pH

# Setup steps (if prep/ is missing)
python run.py solvate
python run.py patch
```

## MPI Configuration

The SLURM script requests:
- `--ntasks=5` — 5 MPI ranks (= 5 replicas)
- `--gpus-per-task=1` — one GPU per replica
- `--bind-to none --map-by slot` — flexible GPU binding

The `cphmd` package auto-detects the MPI communicator size and assigns
GPUs based on `OMPI_COMM_WORLD_LOCAL_RANK` or `SLURM_LOCALID`.

## pH/pKa Logic

For a single LYS site (macro-pKa ~10.5), the `effective_pH` is
automatically set to the site's macro-pKa. This is the PHMD reference
pH around which replicas are spread.

For multi-site proteins with different macro-pKas (e.g., ASP + LYS),
`effective_pH` would be set to 0.0 (neutral reference) instead.

## Differences from 04_lys_water

| | 04_lys_water | 13_lys_mpi |
|---|---|---|
| GPUs | 1 | 5 (configurable) |
| Replicas | 1 | 5 (auto from MPI size) |
| pH coupling | off (`pH` omitted) | on (`pH: true`, auto pKa=10.5) |
| Launch | `python run.py alf` | `mpirun -np 5 cphmd run alf -c cphmd_config.yaml --pH` |
| SLURM | `--gres=gpu:1` | `--ntasks=5 --gpus-per-task=1` |
