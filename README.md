# CpHMD - Constant pH Molecular Dynamics with ALF

A self-contained Python package for Constant pH Molecular Dynamics (CpHMD) using
Adaptive Landscape Flattening (ALF) with pyCHARMM and BLaDE GPU acceleration.

Fully automates the ALF workflow: system preparation, iterative bias optimization,
convergence monitoring, and pKa prediction -- all from a single CLI command.

> **Closed alpha.** This project originally diverged from
> [RyanLeeHayes/ALF](https://github.com/RyanLeeHayes/ALF) and has since been
> extensively rewritten with significant new features including adaptive cutoffs,
> automatic phase switching, EWBS/RMSD convergence, inter-site coupling control,
> Henderson-Hasselbalch analysis, converged single-site presets, YAML-driven
> configuration, and a unified CLI -- making it a standalone package with no
> external ALF dependency.

## Key Features

### Automated System Preparation
- **Titratable residue patching** -- auto-detects GLU, ASP, HIS, LYS, CYS and applies
  MSLD patches with correct protonation states
- **Ligand support** -- custom RTF/PRM files for non-standard titratable groups
- **Hydrogen mass repartitioning** (HMR) -- 4 fs timesteps for faster sampling
- **Solvation** -- automated waterbox setup with octahedral crystal, ion placement (SLTCAP)

### Three-Phase ALF Optimization

The simulation automatically progresses through three phases:

| Phase | Purpose | Cutoffs |
|-------|---------|---------|
| **Phase 1** | Explore -- large bias updates, find landscape shape | Fixed staged (large early, tighter after run 20) |
| **Phase 2** | Warmup -- log-space decay from loose to tight cutoffs | Smooth 20-run transition; x/s terms activate |
| **Phase 3** | Production -- tight cutoffs for precise convergence | Fixed (with recovery if population collapses >70%) |

### Adaptive Bias Cutoffs
Per-parameter clipping prevents large bias jumps from destabilizing the simulation:
- **Phase 1**: Fixed staged cutoffs (no adaptive scaling feedback)
- **Phase 2**: Log-space decay prevents the cliff between Phase 1 and Phase 3
- **Phase 3**: Fixed tight cutoffs with automatic recovery if populations collapse

### Convergence Monitoring

Three complementary convergence metrics:

- **Population-based** (default) -- monitors lambda populations at each pH replica;
  phase transitions based on visited-states overlap and sample counts
- **RMSD-based** -- monitors G-file free energy profiles from WHAM; computes per-site
  RMSD between current and lagged bias landscapes; persistent state in `rmsd_state.json`
- **EWBS** (Exponentially Weighted Bias Stability) -- tracks EWMA of per-type RMS bias
  changes (b/c/x/s); gates Phase 2-to-3 transition and Phase 3 stop; persistent state
  in `ewbs_state.json`

### Automatic Phase Switching
Enable `--auto-phase` to let the simulation decide when to advance:
- **Phase 1 -> 2**: Visited-states gate (>= 2 states per site) with quality guard
  (>= 3 multi-state runs from accumulated data)
- **Phase 2 -> 3**: EWBS below threshold for 5 consecutive runs
- **Phase 3 -> STOP**: Convergence criteria met (balanced populations, low RMSD,
  and/or low EWBS for 10 consecutive runs)

### Analysis Methods

**WHAM** (default) -- GPU-accelerated weighted histogram analysis:
- SVD-based free energy solve with per-parameter cutoff clipping
- Produces G-file profiles for every site and pair
- Transition counting for convergence diagnostics
- In-memory CUDA data passing (no intermediate text files)

**LMALF** -- L-BFGS multisite lambda optimization:
- Gradient-based alternative; can converge faster for well-behaved systems
- Automatic WHAM fallback when gradients are flat (Phase 1) or ill-conditioned
- Configurable iteration limits and convergence tolerance

**Hybrid** -- WHAM for early phases, LMALF for later phases.

### Convergence Visualization
Automatically generated plots during simulation:
- **Population convergence** -- per-site substate populations over ALF iterations
- **RMSD convergence** -- per-site G-file RMSD showing bias landscape stabilization
- **Pairwise RMSD** -- per lambda-pair RMSD with top-N worst pairs highlighted
- **b-bias convergence** -- per-site linear bias drift tracking
- **WHAM free energy profiles** -- 1D/2D G-file landscape snapshots
- **Henderson-Hasselbalch curves** -- titration curves with pKa fits across pH replicas
- **Energy profiles** -- 2D/3D bias energy landscapes on simplex grids

### Converged Single-Site Presets
Pre-computed bias parameters from cubic box simulations:
- Available for ASP, GLU, HSP, LYS, TYR across 10 electrostatic configurations
- Enable via `ALFConfig(use_presets=True)` or YAML `use_presets: true`
- For proteins: residue types auto-detected from `patches.dat`; intra-site biases
  loaded from presets, inter-site coupling starts at zero

### Inter-site Coupling Control
Configurable coupling between titratable sites:
- `coupling=0` (default): No inter-site terms -- simpler, fewer parameters
- `coupling=1`: Full coupling (c + x + s inter-site biases)
- `coupling=2`: Coupling constants only (c terms, no x/s)
- `coupling_profile`: Independent control over inter-site profile monitoring

### Additional Capabilities
- **Parquet lambda files** -- smaller and faster to read than CHARMM binary
- **G_imp entropy computation** -- ideal mixing free energies computed locally via
  Monte Carlo integration, cached at `~/.cache/cphmd/G_imp/`; bundled fallback data
  included for common configurations
- **FNEX centralization** -- all bias shape constants (OMEGA_DECAY, CHI_OFFSET,
  OMEGA_SCALE) derive from a single FNEX parameter; configurable via `ALFConfig(fnex=...)`
- **DCA / Potts model** -- Direct Coupling Analysis for multi-site free energy estimation
- **Bias search** -- post-hoc optimal bias selection from completed simulations
- **Transition analysis** -- lambda state transition counting with connectivity metrics
- **YAML-driven workflow** -- configure all steps (build, solvate, patch, ALF) from a
  single `cphmd_config.yaml` file

## Installation

```bash
git clone <repository-url>
cd ALF_CpHMD
pip install -e .
```

For development (adds pytest, black, ruff, mypy):

```bash
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.9
- pyCHARMM with BLaDE GPU support
- NVIDIA GPU with CUDA (for WHAM/LMALF solver and BLaDE MD)
- mpi4py (for parallel replica simulations; install separately or via `pip install -e ".[mpi]"`)
- NumPy >= 1.20, SciPy >= 1.7, Pandas >= 1.3
- PyYAML >= 6.0
- Typer >= 0.9, Rich >= 12.0 (CLI)
- Matplotlib >= 3.4 (visualization)
- PyArrow >= 10.0 (optional, for Parquet lambda files; install via `pip install -e ".[analysis]"`)

### CUDA WHAM Library

The pre-compiled `libwham.so` is included. To recompile after editing `wham.cu`:

```bash
cd cphmd/wham/src && make
```

## Quick Start

### 1. Patch titratable residues

```bash
cphmd run patch -i solvated/prep -s GLU -s ASP
```

Or from Python:
```python
from cphmd.core import PatchConfig, patch_system

config = PatchConfig(
    input_folder="solvated/prep",
    selected_residues=["GLU", "ASP"],
)
patch_system(config)
```

### 2. Run ALF simulation

```bash
mpirun -np 5 cphmd run alf -i solvated --pH 7.0 \
    --start 1 --end 100 \
    --auto-phase --auto-stop \
    --hh-plots
```

Or from Python:
```python
from cphmd.core import ALFConfig, run_alf_simulation

config = ALFConfig(
    input_folder="solvated",
    pH=7.0,
    start=1, end=100,
    auto_phase_switch=True,
    auto_stop=True,
    generate_hh_plots=True,
)
run_alf_simulation(config)
```

### 3. YAML-driven workflow

Create a `cphmd_config.yaml` and run the full pipeline:

```bash
cphmd run workflow -c cphmd_config.yaml              # all steps
cphmd run workflow -c cphmd_config.yaml --step patch  # single step
```

### 4. Analyze results

```bash
# Energy profile analysis with convergence plots
cphmd analyze energy -i solvated

# Generate production MSLD block files
cphmd analyze block -i solvated/prep

# Bias search for optimal iteration
cphmd run bias-search -i solvated
```

## CLI Reference

```
cphmd setup create-aa         Create amino acid / nucleic acid template structures
cphmd setup solvate           Solvate a molecular system in a water box

cphmd run patch               Apply CpHMD patches to titratable residues
cphmd run alf                 Run ALF simulation (requires MPI + GPU)
cphmd run bias-search         Search for optimal bias parameters from results
cphmd run workflow            Run a multi-step workflow from a YAML config file

cphmd analyze energy          Analyze and visualize energy profiles across iterations
cphmd analyze block           Generate MSLD block and restraint files
cphmd analyze volume          Calculate molecular volume (requires pyCHARMM)

cphmd utils lambda-convert    Convert CHARMM binary .lmd files to Parquet
cphmd utils lambda-info       Show metadata for a lambda file (.lmd or .parquet)
```

### Key ALF Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pH` | None | Target pH (None = standard ALF without pH coupling) |
| `--temp` | 298.15 | Simulation temperature (K) |
| `--phase` | 1 | Initial phase (1, 2, or 3) |
| `--auto-phase` | off | Automatic phase switching |
| `--auto-stop` | off | Stop when converged in Phase 3 |
| `--convergence-mode` | population | Convergence criterion: `population` or `rmsd` |
| `--coupling` | 0 | Inter-site coupling: 0=none, 1=full, 2=c-only |
| `--analysis-method` | wham | Analysis: `wham`, `lmalf`, or `hybrid` |
| `--hh-plots` | off | Generate Henderson-Hasselbalch titration curves |
| `--hmr/--no-hmr` | on | Hydrogen mass repartitioning (4 fs timestep) |
| `--restrains` | SCAT | Restraint type: SCAT or NOE |
| `--elec` | pmeex | Electrostatics: pmeex, pmeon, pmenn, fshift, fswitch |
| `--fnex` | 5.5 | FNEX softmax constraint parameter |
| `--g-imp-bins` | auto | G_imp resolution (single int or per-phase: `20,32,32`) |
| `-c` / `--config` | None | YAML config file (overrides all other options) |

## Project Structure

```
cphmd/
├── core/                         # Core simulation engine
│   ├── alf_runner.py                  # ALF simulation orchestrator (ALFConfig, ALFSimulation)
│   ├── alf_utils.py                   # ALF variable/energy utilities, in-memory WHAM input
│   ├── free_energy.py                 # WHAM free energy solver (SVD + adaptive cutoffs)
│   ├── patching.py                    # Titratable residue patching
│   ├── charmm_utils.py               # pyCHARMM session management
│   ├── cphmd_params.py               # pH-dependent bias calculations
│   ├── phase_switcher.py             # Phase transition logic, EWBS, stop criteria
│   ├── rmsd_convergence.py           # G-file RMSD convergence tracking
│   ├── bias_constants.py             # Centralized FNEX-derived constants
│   ├── entropy.py                     # G_imp ideal mixing free energies (MC integration)
│   ├── block_builder.py              # MSLD block command generation
│   ├── generate_block.py             # Production block file generation
│   ├── restraints.py                  # SCAT/NOE restraint generation
│   ├── bias_search.py                # Optimal bias parameter search
│   ├── transitions.py                # Lambda transition counting
│   ├── variance.py                    # Per-site variance estimation
│   └── patch_atoms.py                # Atom-level patch definitions
├── analysis/                     # Post-simulation analysis and visualization
│   ├── energy_profiles.py            # Energy landscape visualization (2D/3D)
│   ├── henderson_hasselbalch.py      # Titration curve fitting + pKa
│   ├── population_convergence.py     # Population convergence plots
│   ├── rmsd_convergence_plot.py      # RMSD convergence plots (per-site + per-pair)
│   ├── wham_profiles.py              # WHAM G-file profile plotting
│   ├── dca.py                        # Direct Coupling Analysis (Potts model)
│   └── volume.py                     # Molecular volume calculation
├── setup/                        # System preparation
│   ├── create_aa.py                  # Amino acid / nucleic acid template generation
│   └── solvate.py                    # Solvation with ions (SLTCAP)
├── config/                       # Configuration system
│   ├── loader.py                     # YAML config loader with CLI override merging
│   └── defaults/                     # Default YAML configs (alf, patch, solvation)
├── wham/                         # GPU WHAM/LMALF engine
│   ├── __init__.py                   # Python ctypes bindings + in-memory data packing
│   └── src/                          # CUDA C source (wham.cu, Makefile)
├── presets/                      # Converged single-site bias presets
│   └── biases.py                     # ASP/GLU/HSP/LYS/TYR across 10 electrostatic configs
├── utils/                        # Utilities
│   └── lambda_io.py                  # Lambda file I/O (binary read, Parquet read/write)
├── data/                         # Bundled data
│   └── G_imp/                        # Pre-computed G_imp entropy tables
└── cli/                          # Command-line interface
    └── main.py                       # Typer-based CLI with setup/run/analyze/utils groups
```

## Examples

The `examples/` directory contains working setups. Each example includes a
`cphmd_config.yaml`, a `run.py` workflow script, and a `submit.sh` SLURM template.

| Example | Description |
|---------|-------------|
| `00_glu_water` | Single GLU in water -- basic 3-state ALF (no pH coupling) |
| `01_single_residue` | Single ASP pentapeptide -- full build-to-ALF pipeline |
| `02_protein_lysozyme` | Hen egg-white lysozyme -- multi-site CpHMD at pH 4.0 with coupling |
| `03_nr` | Neutral Red -- custom titratable ligand with NRDU patch |
| `05_glu_lmalf` | GLU in water -- LMALF analysis method instead of WHAM |
| `06_glu_hybrid` | GLU in water -- hybrid WHAM/LMALF analysis |

## How It Works

### ALF in Brief

ALF iteratively flattens the free energy landscape so that all protonation states
are sampled equally. Each iteration:

1. **Simulate** -- run `nreps` replicas with current bias potentials (BLaDE GPU)
2. **Analyze** -- WHAM/LMALF solves for free energy differences from lambda trajectories
3. **Update** -- adjust bias parameters (b, c, x, s) with adaptive cutoffs
4. **Monitor** -- check convergence, generate plots, switch phases if needed

The bias potential has the form:
```
V_bias = b_i * (lambda_i - lambda_central) + c_ij * lambda_i * lambda_j
       + x_i * f(lambda) + s_i * g(lambda)
```

where `b` are linear biases, `c` are pairwise coupling terms, `x` are skew corrections
(exponential decay, rate = FNEX), and `s` are endpoint corrections (sigmoid,
offset = 4*exp(-FNEX)). For CpHMD, pH-dependent shifts are added:
```
b_shift = sign * kT * ln(10) * (pH - pKa_ref)
```

### Convergence Criteria

Phase transitions use multi-run confirmation to avoid premature transitions.
In Phase 3, convergence requires:

- **Population mode**: Balanced physical state populations across replicas
- **RMSD mode**: G-file profile RMSD below threshold for sustained iterations
- **EWBS**: Bias change rate below threshold for 10 consecutive runs

## Testing

```bash
pytest tests/             # full suite (143 tests)
pytest tests/ -k phase    # filter by keyword
pytest tests/ -v --tb=short
```

## References

1. Hayes, R.L.; Armacost, K.A.; Vilseck, J.Z.; Brooks, C.L. III (2017)
   *Adaptive Landscape Flattening Accelerates Sampling of Alchemical Space in Multisite lambda Dynamics.*
   J. Phys. Chem. B **121**, 3626-3635. [DOI: 10.1021/acs.jpcb.6b09656](https://doi.org/10.1021/acs.jpcb.6b09656)

2. Hayes, R.L.; Buckner, J.; Brooks, C.L. III (2021)
   *BLaDE: A Basic Lambda Dynamics Engine for GPU-Accelerated Molecular Dynamics Free Energy Calculations.*
   J. Chem. Theory Comput. **17**, 6799-6807. [DOI: 10.1021/acs.jctc.1c00833](https://doi.org/10.1021/acs.jctc.1c00833)

3. Knight, J.L.; Brooks, C.L. III (2011)
   *Multisite lambda Dynamics for Simulated Structure-Activity Relationship Studies.*
   J. Chem. Theory Comput. **7**, 2728-2739. [DOI: 10.1021/ct200444f](https://doi.org/10.1021/ct200444f)

4. Goh, G.B.; Hulbert, B.S.; Zhou, H.; Brooks, C.L. III (2014)
   *Constant pH Molecular Dynamics of Proteins in Explicit Solvent with Proton Tautomerism.*
   Proteins **82**, 1319-1331. [DOI: 10.1002/prot.24499](https://doi.org/10.1002/prot.24499)

5. Buckner, J.; Liu, X.; Chakravorty, A.; Wu, Y.; Cervantes, L.F.; Lai, T.T.; Brooks, C.L. III (2023)
   *pyCHARMM: Embedding CHARMM Functionality in a Python Framework.*
   J. Chem. Theory Comput. **19**, 3752-3762. [DOI: 10.1021/acs.jctc.3c00364](https://doi.org/10.1021/acs.jctc.3c00364)

6. Brooks, B.R., et al. (2009)
   *CHARMM: The Biomolecular Simulation Program.*
   J. Comput. Chem. **30**, 1545-1614. [DOI: 10.1002/jcc.21287](https://doi.org/10.1002/jcc.21287)

## License

MIT License -- See LICENSE file for details.
