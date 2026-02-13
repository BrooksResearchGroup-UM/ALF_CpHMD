# 03 - Neutral Red (Custom Ligand)

Demonstrates CpHMD with a **custom titratable ligand** (Neutral Red).
The NRDU patch converts the protonated cation (+1) to the neutral form (0)
by deleting the H36 proton from the N2 ring nitrogen.

## Ligand Details

| Property | Value |
|----------|-------|
| Residue name | NRED |
| Protonation site | N2 (ring nitrogen) |
| pKa (free) | 6.8 |
| Charge change | +1 -> 0 |
| Patch | NRDU (deprotonation) |

## Directory Structure

```
03_nr/
├── cphmd_config.yaml      # Workflow configuration
├── run.py                 # Workflow script
├── submit.sh              # SLURM submission
├── pdb/                   # Input structures
│   ├── nr_protonated.psf  # Protonated NR (starting state)
│   ├── nr_protonated.crd
│   └── nr_protonated.pdb
└── toppar/                # Ligand topology/parameters
    ├── nr_protonated.cgenff.rtf  # CGenFF topology
    ├── nr_protonated.cgenff.prm  # CGenFF parameters
    └── nr_titratable.str         # Titratable patch (NRDU)
```

## Usage

```bash
# Setup (CPU, no GPU needed)
python run.py solvate
python run.py patch

# ALF simulation (requires GPU)
python run.py alf         # Interactive
sbatch submit.sh          # SLURM batch

# Or run everything
python run.py all
```

## Key Points

- CpHMD requires the **protonated** form as the starting state
- The `ligand_patches` section in `cphmd_config.yaml` defines the titratable site
- `extra_files` must include ligand RTF/PRM/STR in solvation, patching, and ALF sections
- `selected_residues: []` skips standard amino acid patching (ligand only)
