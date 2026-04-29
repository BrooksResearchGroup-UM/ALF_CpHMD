# ALF CpHMD

ALF CpHMD is a research implementation of ALF-based constant-pH molecular
dynamics workflows for CHARMM and pyCHARMM. The current development line uses
the native pyCHARMM runtime and targets recent CHARMM/pyCHARMM builds.

This repository is distributed under the research-use terms in `LICENSE`.

## Requirements

- Python 3.9 or newer.
- CHARMM with pyCHARMM 0.5.1 or newer. The tested local environment is
  `chm_12.9`.
- MPI support for multi-replica runs. Install the optional `mpi` extra when
  `mpi4py` is not already available.
- CUDA and a compatible CHARMM build for BLaDE or CUDA WHAM analysis.
- `crimm` only for setup workflows that prepare structures from PDB input.

pyCHARMM is provided by the CHARMM installation, not by this package. Activate
the CHARMM environment before installing or running CpHMD.

## Install

For development:

```bash
pip install -e .
pip install -e ".[mpi,setup,analysis,dev]"
```

For a runtime-only editable install:

```bash
pip install -e ".[mpi,analysis]"
```

The package includes bundled CHARMM topology/parameter files under `toppar/`,
precomputed G_imp tables under `cphmd/data/G_imp/`, and the CUDA WHAM source
under `cphmd/wham/src/`.

## WHAM Library

`cphmd/wham/libwham.so` is the CUDA WHAM shared library used by pKa and ALF
analysis paths. It can be rebuilt from source:

```bash
cd cphmd/wham/src
make
```

or:

```bash
cd cphmd/wham
./build.sh
```

If CUDA is unavailable, use `native.analysis_backend: disabled` for smoke tests
that only verify CHARMM system loading and short dynamics.

## Native Runtime

The current public CLI is:

```bash
cphmd init -c cphmd_config.yaml
mpirun -np <nreps> cphmd run -c cphmd_config.yaml
cphmd status --run-dir <run_dir>
```

`cphmd init` writes the run-state marker and validates the runtime inputs.
`cphmd run` must be launched with an MPI world size equal to `alf.nreps`.

Minimal config shape:

```yaml
master_seed: 20260419
run_dir: runs/example

native:
  dynamics_backend: domdec-cpu
  analysis_backend: disabled

alf:
  input_folder: examples/00_glu_water
  nreps: 1
  start: 1
  end: 5
  md_block_steps: 10
  lambda_save_steps: 5
  coordinate_save_steps: 10
  checkpoint_interval_steps: 10
  phase1_iteration_steps: 10
  phase2_iteration_steps: 20
  phase3_iteration_steps: 20
  ph: false
```

If `alf.ph` or `alf.pH` is absent, pH coupling is disabled. A numeric `ph`/`pH`
value or `ph: true` enables the pH path. Use `ph_values`, `pH_values`,
`ph_start`/`ph_end`, or the auto-derived ladder for multi-replica pH runs.
For pH-aware WHAM analysis, keep `use_gshift: true` in the ALF section.

## Dynamics Backends

Set the dynamics engine with `native.dynamics_backend`:

- `blade`: BLaDE GPU dynamics.
- `domdec-cpu`: DOMDEC CPU dynamics. Useful for smoke tests without a GPU.
- `domdec-gpu`: DOMDEC GPU dynamics.

Set the analysis engine with `native.analysis_backend`:

- `cuda-wham`: CUDA WHAM/LMALF analysis.
- `disabled`: fixed-MD-block smoke mode with no ALF bias rebuild cycle.

`use_blade` is not accepted in YAML. Use `native.dynamics_backend` instead.

Runtime intervals can be written in time or step units. For example, use either
`md_block_ps` or `md_block_steps`, `phase1_iteration_ps` or
`phase1_iteration_steps`, `exchange_interval_ps` or `exchange_interval_steps`,
and `production.duration_ns` or `production.duration_steps`. If both forms are
present for the same quantity, the config is rejected. The old
`nsteps_per_segment`, `phase*_repeats`, and `checkpoint_every_segments` keys are
accepted for existing configs. In that compatibility mode, `phase*_repeats`
means MD blocks per ALF iteration; new examples use explicit MD block and ALF
iteration language instead.

## Examples

Examples are meant to run from inside their own folders. Use `python run.py
setup` only when regenerating setup inputs, then launch the native runtime with
`cphmd init` and `cphmd run`:

```bash
cd examples/01_asp_water
python run.py setup
cphmd init -c cphmd_config.yaml
mpirun -np 1 cphmd run -c cphmd_config.yaml
```

See `examples/README.md` for the full example workflow. The
`examples/validation/` configs are short smoke runs that reuse prepared
systems:

```bash
python -m cphmd.cli.main init -c examples/validation/asp_no_ph_domdec_cpu/cphmd_config.yaml
mpirun -np 1 python -m cphmd.cli.main run -c examples/validation/asp_no_ph_domdec_cpu/cphmd_config.yaml

python -m cphmd.cli.main init -c examples/validation/asp_ph_domdec_cpu/cphmd_config.yaml
mpirun -np 1 python -m cphmd.cli.main run -c examples/validation/asp_ph_domdec_cpu/cphmd_config.yaml
```

Use the BLaDE validation config on a GPU node:

```bash
python -m cphmd.cli.main init -c examples/validation/asp_ph_blade_gpu/cphmd_config.yaml
mpirun -np 1 python -m cphmd.cli.main run -c examples/validation/asp_ph_blade_gpu/cphmd_config.yaml
```

These validation configs are intended to catch setup, pyCHARMM, pH, and backend
dispatch errors. They are not long enough to produce scientifically meaningful
pKa estimates.

## Project Layout

- `cphmd/cli/`: native CLI commands.
- `cphmd/config/`: YAML loading and config compatibility.
- `cphmd/core/`: compatibility shims and shared CpHMD utilities.
- `cphmd/native/`: narrow pyCHARMM wrappers.
- `cphmd/setup/`: structure preparation, solvation, and patching helpers.
- `cphmd/simulation/`: native runtime context, loop, checkpointing, and archive
  writers.
- `cphmd/training/`: ALF training hooks and native analysis integration.
- `cphmd/wham/`: CUDA WHAM bindings and source.
- `toppar/`: bundled CHARMM topology and parameter inputs.

## Development Checks

Useful local checks:

```bash
python -m compileall -q cphmd toppar
python -m pytest tests/unit -q
ruff check cphmd scripts
black --check cphmd scripts
```

pyCHARMM and GPU smoke checks should be run in the CHARMM environment and, for
GPU backends, through the site scheduler.
