# Examples

Each example is meant to run from inside its own folder. Setup outputs are
written in the example folder and the native runtime uses `prep/`, `state/`,
and `res/` directly; there is no separate `solvated/` run folder.

## Standard Flow

```bash
cd examples/01_asp_water

# Optional, only needed when regenerating inputs.
python run.py setup

# Native ALF/CpHMD run.
cphmd init -c cphmd_config.yaml
mpirun -np 1 cphmd run -c cphmd_config.yaml
```

For multi-replica examples, set `-np` to `alf.nreps` and use one MPI rank per
replica. The Slurm scripts in each example use the same native launch path.

`python run.py` is only a setup helper. It accepts `setup`, `all` as a setup
alias, `build`, `prepare`, `solvate`, and `patch`. ALF and production MD should
be launched with `cphmd init` and `cphmd run` so MPI, GPU mapping, and restart
state are explicit.

## Production Repeats

Production examples can run independent repeats into explicit run directories:

```bash
cd examples/03_hsp_water
sbatch submit_production_repeat.sh sim_1 production_config.yaml
sbatch submit_production_repeat.sh sim_2 production_config.yaml
sbatch submit_production_repeat.sh sim_3 production_config.yaml

cphmd analyze production-pka sim_1 sim_2 sim_3 --output analysis/production_pka
```
