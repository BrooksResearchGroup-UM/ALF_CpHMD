# Validation Examples

These configs are short runtime checks for the native pyCHARMM path. They reuse
the prepared system in `examples/01_asp_water`.

Run from the repository root after activating the CHARMM/pyCHARMM environment:

```bash
python -m cphmd.cli.main init -c examples/validation/asp_no_ph_domdec_cpu/cphmd_config.yaml
mpirun -np 1 python -m cphmd.cli.main run -c examples/validation/asp_no_ph_domdec_cpu/cphmd_config.yaml

python -m cphmd.cli.main init -c examples/validation/asp_ph_domdec_cpu/cphmd_config.yaml
mpirun -np 1 python -m cphmd.cli.main run -c examples/validation/asp_ph_domdec_cpu/cphmd_config.yaml
```

The BLaDE config requires a GPU node:

```bash
python -m cphmd.cli.main init -c examples/validation/asp_ph_blade_gpu/cphmd_config.yaml
mpirun -np 1 python -m cphmd.cli.main run -c examples/validation/asp_ph_blade_gpu/cphmd_config.yaml
```

These runs are intentionally too short for scientific pKa estimates. They are
only meant to verify system loading, backend dispatch, pH setup, and short
dynamics.
