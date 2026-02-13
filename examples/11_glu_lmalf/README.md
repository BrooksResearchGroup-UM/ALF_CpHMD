# 05 - Glutamic Acid with LMALF Analysis

Same system as `00_glu_water` (single GLU in water) but uses **LMALF**
(Likelihood Maximization ALF) for bias parameter fitting instead of WHAM.

## WHAM vs LMALF

| Method | Approach | Status |
|--------|----------|--------|
| **WHAM** | Iterative histogram reweighting | Stable, default |
| **LMALF** | L-BFGS quasi-Newton optimization | Experimental |

LMALF directly optimizes the likelihood function using gradients, which can
converge faster than WHAM's iterative reweighting for well-behaved systems.

## Usage

```bash
# Solvate (uses PDB from ../00_glu_water/pdb/)
python run.py solvate
python run.py patch

# Run ALF with LMALF analysis
python run.py alf         # Interactive
sbatch submit.sh          # SLURM batch
```

## CLI Equivalent

```bash
cphmd run alf -i solvated -c cphmd_config.yaml
```

Or without config file:
```bash
cphmd run alf -i solvated --analysis-method lmalf --hydrogens --g-imp-bins 20,32,32
```

## LMALF Options

- `analysis_method: lmalf` -- use L-BFGS optimization
- `lmalf_max_iter: 0` -- max L-BFGS iterations (0 = default 250)
- `lmalf_tolerance: 0.0` -- convergence tolerance (0 = default 1.25e-3)

## Known Limitations

1. **Phase 1 flat biases**: LMALF may produce all-zero output when starting from
   flat biases. The code automatically falls back to WHAM in this case.
2. **L-BFGS direction**: May fail with "hi is pointing wrong way" when the
   gradient is very small. Handled by automatic WHAM fallback.
