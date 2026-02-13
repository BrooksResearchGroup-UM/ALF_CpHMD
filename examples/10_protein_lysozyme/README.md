# 02 - Protein Lysozyme (Multi-Site CpHMD)

Hen egg-white lysozyme (2LZT) with all titratable residues at pH 4.0.

This example demonstrates CpHMD on a full protein with multiple titratable
sites and inter-site coupling. No build step is needed — the pre-built
protein PSF/CRD are provided in `pdb/`.

## Titratable Residues

Lysozyme contains 32 titratable residues:
- 2 GLU (E7, E35)
- 7 ASP (D18, D48, D52, D66, D87, D101, D119)
- 1 HIS (H15)
- 6 LYS (K1, K13, K33, K96, K97, K116)
- 3 TYR (Y20, Y23, Y53)
- 13 ARG (not typically titratable in standard CpHMD)

Key validation targets:
- **GLU35**: pKa ~6.2 (elevated, catalytic proton donor)
- **ASP52**: pKa ~3.6 (depressed, charge stabilization)

## Workflow

```bash
# Step 1: Solvate the protein
python run.py solvate

# Step 2: Apply titratable patches (all residues)
python run.py patch

# Step 3: Run ALF simulation
python run.py alf

# Or run all steps:
python run.py all
```

## SLURM Submission

```bash
# Run solvation + patching locally (fast, no GPU needed):
python run.py solvate
python run.py patch

# Submit ALF to GPU cluster:
sbatch submit.sh alf
```

## Configuration

See `cphmd_config.yaml` for all settings. Key options:
- `alf.pH: true` — enable CpHMD (effective pH auto-computed from macro-pKa values)
- `alf.coupling: 1` — full inter-site coupling (c/x/s terms)
- `patch.selected_residues: []` — empty list patches all titratable residues
- `alf.hh_plots: true` — generate Henderson-Hasselbalch titration curves

## Files

```
02_protein_lysozyme/
├── pdb/
│   ├── molecule.psf          # Pre-built lysozyme PSF
│   └── molecule.crd          # Pre-built lysozyme coordinates
├── cphmd_config.yaml         # Workflow configuration
├── run.py                    # Unified workflow runner
├── submit.sh                 # SLURM submission template
└── README.md                 # This file
```

## Expected Output

After solvation and patching:
```
solvated/
└── prep/
    ├── system.psf, system.crd     # Patched structure
    ├── system_hmr.psf/crd         # HMR variant
    ├── patches.dat                # Site definitions with pKa values
    └── box.dat, fft.dat           # Box parameters
```

After ALF simulation:
```
solvated/
├── analysis{N}/               # Bias parameters per iteration
├── run{N}/                    # MD trajectories
├── plots/                     # Population and convergence plots
└── ewbs_state.json            # Convergence tracking
```
