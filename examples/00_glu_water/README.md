# 00 — Glutamic Acid (GLU) in Water

Standard ALF workflow for a single GLU residue solvated in water.
No CpHMD pH coupling — this example demonstrates basic ALF free energy
flattening for a 3-state system (GLU deprotonated + two protonated forms).

## Prerequisites

- pyCHARMM environment (`conda activate chm_12.9`)
- GPU for BLaDE acceleration
- Pre-built `pdb/glu.pdb` structure

## Quick Start

```bash
# Optional setup refresh if prep/ is missing
python run.py setup

# Or run setup steps individually
python run.py solvate   # Solvate GLU in octahedral water box
python run.py patch     # Apply titratable patches (GLH1, GLH2)

# Native ALF run
cphmd init -c cphmd_config.yaml
mpirun -np 1 cphmd run -c cphmd_config.yaml
```

## Configuration

All parameters are in `cphmd_config.yaml`. Key settings:

| Parameter | Value | Description |
|-----------|-------|-------------|
| pH | None | No CpHMD (standard ALF) |
| auto_phase | true | Automatic phase transitions 1 → 2 → 3 |
| auto_stop | true | Stop when converged in Phase 3 |
| g_imp_bins | 20,32,32 | G_imp resolution per phase |

## SLURM Submission

```bash
sbatch submit.sh
```

## Output

After convergence (~50-100 runs):
- `state/` — Native run-state marker and scheduler metadata
- `res/rep00/` — Per-MD-block lambda parquet and checkpoint files
- `analysis*/` — Bias parameters per iteration
- `plots/` — Dashboard and optional diagnostic plots
