# CpHMD - Constant pH Molecular Dynamics

A Python package for Constant pH Molecular Dynamics (CpHMD) using
Adaptive Landscape Flattening (ALF) with pyCHARMM and BLaDE GPU acceleration.

## Features

- **Automated patching** of titratable residues (GLU, ASP, HIS, LYS, etc.)
- **GPU-accelerated** simulations via pyCHARMM/BLaDE
- **ALF optimization** for accurate pKa predictions
- **Energy profile analysis** with convergence tracking
- **Flexible CLI** for all workflow steps
- **Parquet support** for efficient lambda file storage

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ALF_CpHMD.git
cd ALF_CpHMD

# Install in development mode
pip install -e .
```

### Requirements

- Python 3.10+
- pyCHARMM with BLaDE support
- NumPy, SciPy, Pandas
- PyArrow (for Parquet support)
- Typer, Rich (for CLI)
- Matplotlib/Plotly (optional, for visualization)

## Quick Start

### 1. Patch titratable residues

Transform your solvated structure into a CpHMD-ready system:

```python
from cphmd.core import PatchConfig, patch_system

config = PatchConfig(
    input_folder="solvated/prep",
    selected_residues=["GLU", "ASP"],  # Or specific: ["PROA:35"]
)
patch_system(config)
```

Or via CLI:
```bash
cphmd run patch -i solvated/prep -s GLU -s ASP
```

### 2. Run ALF simulation

Optimize bias potentials through iterative simulations:

```python
from cphmd.core import ALFConfig, run_alf_simulation

config = ALFConfig(
    input_folder="solvated",
    pH=7.0,
    temperature=298.15,
    start=1,
    end=20,
)
run_alf_simulation(config)
```

Or via CLI:
```bash
cphmd run alf -i solvated --pH 7.0 --start 1 --end 20
```

### 3. Analyze results

Check convergence and generate production files:

```bash
# Energy profile analysis
cphmd analyze energy -i solvated

# Generate production block files
cphmd analyze block -i solvated/prep
```

## Package Structure

```
cphmd/
├── core/           # Core simulation modules
│   ├── patching.py       # Titratable residue patching
│   ├── alf_runner.py     # ALF simulation orchestrator
│   ├── alf_utils.py      # ALF utility functions
│   ├── bias_search.py    # Optimal bias search
│   └── generate_block.py # MSLD block generation
├── analysis/       # Analysis modules
│   ├── energy_profiles.py # Energy landscape visualization
│   └── volume.py          # Molecular volume calculation
├── utils/          # Utilities
│   └── lambda_io.py       # Lambda file I/O (binary/parquet)
└── cli/            # Command-line interface
    └── main.py            # Typer-based CLI
```

## CLI Commands

```bash
# Setup commands
cphmd setup create-aa    # Create amino acid structures
cphmd setup solvate      # Solvate a system

# Run commands
cphmd run patch          # Apply CpHMD patches
cphmd run alf            # Run ALF simulation
cphmd run bias-search    # Search for optimal bias

# Analysis commands
cphmd analyze energy     # Energy profile analysis
cphmd analyze block      # Generate MSLD block files
cphmd analyze volume     # Volume calculation

# Utility commands
cphmd utils lambda-convert  # Convert .lmd to .parquet
cphmd utils lambda-info     # Show lambda file info
```

## Configuration Options

### ALFConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 298.15 | Simulation temperature (K) |
| `pH` | 7.0 | Target pH value |
| `hmr` | True | Hydrogen mass repartitioning |
| `start/end` | 1/10 | ALF iteration range |
| `restrains` | "SCAT" | Restraint type (SCAT/NOE) |
| `elec_type` | "pmeex" | PME method |
| `vdw_type` | "vswitch" | VDW method |

### PatchConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `structure_file` | "solvated" | Input structure base name |
| `hmr` | True | Apply hydrogen mass repartitioning |
| `selected_residues` | [] | Residues to patch (empty = all) |
| `extra_files` | [] | Additional topology files |

## Examples

See the `examples/` directory for documented examples:

- `01_single_residue/` - Basic GLU in water
- `02_protein_lysozyme/` - Protein with multiple sites

## References

1. Hayes, R.L., et al. (2015) J. Phys. Chem. B 119, 7030-7038.
   *Adaptive Landscape Flattening for pKa Prediction*

2. Brooks, B.R., et al. (2009) J. Comput. Chem. 30, 1545-1614.
   *CHARMM: A program for macromolecular energy, minimization, and dynamics*

## License

MIT License - See LICENSE file for details.
