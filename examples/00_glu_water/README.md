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
# Run all steps sequentially
python run.py all

# Or run steps individually
python run.py solvate   # Solvate GLU in octahedral water box
python run.py patch     # Apply titratable patches (GLH1, GLH2)
python run.py alf       # Run ALF simulation (requires GPU)
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
- `analysis*/` — Bias parameters per iteration
- `run*/` — MD trajectory data
- `plots/` — Energy profile convergence plots
