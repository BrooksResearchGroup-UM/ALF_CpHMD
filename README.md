# CpHMD - Constant pH Molecular Dynamics with ALF

A self-contained Python package for Constant pH Molecular Dynamics (CpHMD) using
Adaptive Landscape Flattening (ALF) with pyCHARMM and BLaDE GPU acceleration.

Fully automates the ALF workflow: system preparation, iterative bias optimization,
convergence monitoring, and pKa prediction — all from a single CLI command.

## Key Features

### Automated System Preparation
- **Titratable residue patching** — auto-detects GLU, ASP, HIS, LYS, CYS and applies
  MSLD patches with correct protonation states
- **Ligand support** — custom RTF/PRM files for non-standard titratable groups
- **Hydrogen mass repartitioning** (HMR) — 4 fs timesteps for faster sampling
- **Solvation** — automated waterbox setup with octahedral crystal, ion placement (SLTCAP)
- **Legacy format support** — reads msld-py-prep style inputs (`prep_format="legacy"`)

### Three-Phase ALF Optimization
The simulation automatically progresses through three phases:

| Phase | Purpose | Cutoffs |
|-------|---------|---------|
| **Phase 1** | Explore — large bias updates, find landscape shape | Adaptive (loosen if bottlenecked) |
| **Phase 2** | Warmup — log-space decay from loose to tight cutoffs | Smooth 20-run transition |
| **Phase 3** | Production — tight cutoffs for precise convergence | Fixed (with recovery if population collapses) |

### Adaptive Bias Cutoffs
Per-parameter clipping prevents large bias jumps from destabilizing the simulation:
- **Phase 1**: Adaptive cutoffs that loosen automatically when the scaling bottleneck is detected
- **Phase 2**: Smooth warmup decay prevents the cliff between Phase 1 and Phase 2
- **Phase 3**: Fixed tight cutoffs with automatic recovery if populations collapse (>70% imbalance)
- **Population dampening**: Per-site scaling based on worst-sampled substate

### Two Convergence Modes

**Population-based** (default) — monitors lambda populations at each pH replica:
- Phase transitions based on overlap and sample counts
- Stop criteria based on physical state balance and bias stability
- pKa convergence tracking across pH replicas

**RMSD-based** — monitors G-file free energy profiles from WHAM:
- Computes per-site RMSD between current and lagged bias landscapes
- Better for large multi-site systems (5+ substates) where population metrics are noisy
- Persistent state saved to `rmsd_state.json` for seamless resume

### Automatic Phase Switching
Enable `--auto-phase` to let the simulation decide when to advance:
- **Phase 1 → 2**: Sufficient lambda overlap and sample counts
- **Phase 2 → 3**: Bias landscape stabilized
- **Phase 3 → STOP**: Convergence criteria met (balanced populations or low RMSD)

Wrong-direction detection in Phase 1 automatically loosens cutoffs if populations
diverge for 2 consecutive iterations.

### Analysis Methods

**WHAM** (default) — GPU-accelerated weighted histogram analysis:
- SVD-based free energy solve with per-parameter cutoff clipping
- Produces G-file profiles for every site and pair
- Transition counting for convergence diagnostics

**LMALF** — L-BFGS multisite lambda optimization:
- Alternative gradient-based method
- Configurable iteration limits and convergence tolerance

### Convergence Visualization
Automatically generated plots during simulation:
- **Population convergence** — per-site substate populations over ALF iterations
- **RMSD convergence** — per-site G-file RMSD showing bias landscape stabilization
- **Pairwise RMSD** — per lambda-pair RMSD with top-N worst pairs highlighted
- **Henderson-Hasselbalch curves** — titration curves with pKa fits across pH replicas
- **Energy profiles** — 2D/3D bias energy landscapes on simplex grids

### Inter-site Coupling Control
Configurable coupling between titratable sites:
- `coupling=0` (default): No inter-site terms — simpler, fewer parameters
- `coupling=1`: Full coupling (c + x + s inter-site biases)
- `coupling=2`: Coupling constants only (c terms, no x/s)
- `coupling_profile`: Independent control over inter-site profile monitoring

### Additional Capabilities
- **Parquet lambda files** — 8x smaller, 17x faster to read than CHARMM binary
- **Entropy computation** — G_imp ideal mixing free energies computed locally and cached
- **DCA / Potts model** — Direct Coupling Analysis for multi-site free energy estimation
- **Bias search** — post-hoc optimal bias selection from completed simulations
- **Transition analysis** — lambda state transition counting with connectivity metrics
- **Variance estimation** — per-site and per-pair variance from replica data

## Installation

```bash
git clone https://github.com/yourusername/ALF_CpHMD.git
cd ALF_CpHMD
pip install -e .
```

### Requirements
- Python 3.10+
- pyCHARMM with BLaDE GPU support
- mpi4py (for parallel replica simulations)
- NumPy, SciPy, Pandas
- PyArrow (for Parquet support)
- Typer, Rich (CLI)
- Matplotlib (optional, for visualization)

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

### 3. Analyze results

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
cphmd setup create-aa        # Create amino acid template structures
cphmd setup solvate           # Solvate system in waterbox

cphmd run patch               # Apply CpHMD patches to titratable residues
cphmd run alf                 # Run ALF simulation (requires MPI)
cphmd run bias-search         # Search for optimal bias parameters

cphmd analyze energy          # Energy profile analysis + plots
cphmd analyze block           # Generate MSLD block/restraint files
cphmd analyze volume          # Molecular volume calculation

cphmd utils lambda-convert    # Convert .lmd to .parquet
cphmd utils lambda-info       # Show lambda file metadata
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
| `--analysis-method` | wham | Analysis: `wham` or `lmalf` |
| `--hh-plots` | off | Generate Henderson-Hasselbalch titration curves |
| `--hmr/--no-hmr` | on | Hydrogen mass repartitioning (4fs timestep) |
| `--restrains` | SCAT | Restraint type: SCAT or NOE |
| `--elec` | pmeex | Electrostatics: pmeex, pmeon, pmenn, fshift, fswitch |

## Project Structure

```
cphmd/
├── core/                    # Core simulation engine
│   ├── alf_runner.py             # ALF simulation orchestrator
│   ├── patching.py               # Titratable residue patching
│   ├── charmm_utils.py           # pyCHARMM session management
│   ├── alf_utils.py              # ALF variable/energy utilities
│   ├── free_energy.py            # WHAM free energy solver (SVD + cutoffs)
│   ├── cphmd_params.py           # pH-dependent bias calculations
│   ├── block_builder.py          # MSLD block command generation
│   ├── restraints.py             # SCAT/NOE restraint generation
│   ├── phase_switcher.py         # Automatic phase transition logic
│   ├── rmsd_convergence.py       # G-file RMSD convergence tracking
│   ├── transitions.py            # Lambda transition counting
│   ├── variance.py               # Per-site variance estimation
│   ├── entropy.py                # G_imp ideal mixing free energies
│   ├── bias_search.py            # Optimal bias parameter search
│   └── generate_block.py         # Production block file generation
├── analysis/                # Post-simulation analysis
│   ├── energy_profiles.py        # Energy landscape visualization
│   ├── henderson_hasselbalch.py  # Titration curve fitting + pKa
│   ├── population_convergence.py # Population convergence plots
│   ├── rmsd_convergence_plot.py  # RMSD convergence plots (per-site + per-pair)
│   ├── dca.py                    # Direct Coupling Analysis (Potts model)
│   └── volume.py                 # Molecular volume calculation
├── setup/                   # System preparation
│   ├── create_aa.py              # Amino acid template generation
│   └── solvate.py                # Solvation with ions
├── utils/                   # Utilities
│   └── lambda_io.py              # Lambda file I/O (binary/Parquet)
├── cli/                     # Command-line interface
│   └── main.py                   # Typer-based CLI
├── wham/                    # GPU WHAM engine
├── presets/                 # Pre-configured bias presets
└── data/                    # Bundled data (G_imp tables, etc.)
```

## Examples

The `examples/` directory contains working setups:

| Example | Description |
|---------|-------------|
| `00_glu_water` | Single GLU in water — simplest test case |
| `01_single_residue` | Single titratable residue basics |
| `02_protein_lysozyme` | Multi-site protein (lysozyme) |
| `03_nr` | Custom ligand (NR) with NRED patch |
| `04_hsp_water` | Histidine (3-state: HSP/HSD/HSE) |
| `05_glu_lmalf` | GLU with LMALF analysis method |
| `10_rbf+nr` | Multi-ligand system |

## How It Works

### ALF in Brief

ALF iteratively flattens the free energy landscape so that all protonation states
are sampled equally. Each iteration:

1. **Simulate** — run `nreps` replicas with current bias potentials
2. **Analyze** — WHAM solves for free energy differences from lambda trajectories
3. **Update** — adjust bias parameters (b, c, x, s) with adaptive cutoffs
4. **Monitor** — check convergence, generate plots, switch phases if needed

The bias potential has the form:
```
V_bias = b_i * (λ_i - λ_central) + c_ij * λ_i * λ_j + x_i * f(λ) + s_i * g(λ)
```

where `b` are linear biases, `c` are pairwise coupling terms, `x` are skew corrections,
and `s` are endpoint corrections. For CpHMD, pH-dependent shifts are added:
```
b_shift = sign * kT * ln(10) * (pH - pKa_ref)
```

### Convergence Criteria

Phase transitions use a two-confirmation system to avoid premature transitions.
In Phase 3, convergence requires:

- **Population mode**: Balanced physical state populations across replicas
- **RMSD mode**: G-file profile RMSD below threshold for sustained iterations

## References

1. Hayes, R.L.; Armacost, K.A.; Vilseck, J.Z.; Brooks, C.L. III (2017)
   *Adaptive Landscape Flattening Accelerates Sampling of Alchemical Space in Multisite λ Dynamics.*
   J. Phys. Chem. B **121**, 3626-3635. [DOI: 10.1021/acs.jpcb.6b09656](https://doi.org/10.1021/acs.jpcb.6b09656)

2. Hayes, R.L.; Buckner, J.; Brooks, C.L. III (2021)
   *BLaDE: A Basic Lambda Dynamics Engine for GPU-Accelerated Molecular Dynamics Free Energy Calculations.*
   J. Chem. Theory Comput. **17**, 6799-6807. [DOI: 10.1021/acs.jctc.1c00833](https://doi.org/10.1021/acs.jctc.1c00833)

3. Knight, J.L.; Brooks, C.L. III (2011)
   *Multisite λ Dynamics for Simulated Structure-Activity Relationship Studies.*
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

MIT License — See LICENSE file for details.
