# 01 — Single Residue (ASP) in Water

Full CpHMD workflow for a single ASP residue, including the build step.
No pre-built PDB is needed — the `build` step creates an
ALA-ALA-ASP-ALA-ALA pentapeptide with ACE/CT3 caps.

This example demonstrates a 3-state system: ASP deprotonated +
two protonated forms (ASH1 on OD1, ASH2 on OD2).

## Prerequisites

- pyCHARMM environment (`conda activate chm_12.9`)
- GPU for BLaDE acceleration

## Quick Start

```bash
# Optional setup refresh if prep/ is missing
python run.py setup

# Or run setup steps individually
python run.py build     # Build ASP pentapeptide structure
python run.py solvate   # Solvate in octahedral water box
python run.py patch     # Apply titratable patches (ASH1, ASH2)

# Native ALF run
cphmd init -c cphmd_config.yaml
mpirun -np 1 cphmd run -c cphmd_config.yaml
```

## Configuration

All parameters are in `cphmd_config.yaml`. Key settings:

| Parameter | Value | Description |
|-----------|-------|-------------|
| residue | ASP | Amino acid to build |
| template | ALA ALA {res} ALA ALA | Pentapeptide context |
| selected_residues | PROA:3 | ASP is residue 3 in pentapeptide |
| pH | None | No CpHMD (standard ALF) |
| auto_phase | true | Automatic phase transitions 1 → 2 → 3 |
| auto_stop | true | Stop when converged in Phase 3 |

## SLURM Submission

```bash
sbatch submit.sh
```

## Output

After convergence (~50-100 runs):
- `pdb/asp.pdb` — Built structure
- `prep/` — Patched system files
- `state/` — Native run-state marker and scheduler metadata
- `res/rep00/` — Per-MD-block lambda parquet and checkpoint files
- `analysis*/` — Bias parameters per iteration
