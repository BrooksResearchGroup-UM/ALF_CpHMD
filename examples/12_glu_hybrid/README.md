# 06 - Glutamic Acid with Hybrid Analysis

Same system as `00_glu_water` (single GLU in water) but uses a **hybrid**
analysis method that combines WHAM and LMALF approaches.

## Usage

```bash
# Solvate (uses PDB from ../00_glu_water/pdb/)
python run.py solvate
python run.py patch

# Run ALF with hybrid analysis
python run.py alf         # Interactive
sbatch submit.sh          # SLURM batch
```

## CLI Equivalent

```bash
cphmd run alf -i solvated -c cphmd_config.yaml
```

Or without config file:
```bash
cphmd run alf -i solvated --analysis-method hybrid --hydrogens --g-imp-bins 20,32,32
```

## How Hybrid Works

The hybrid method uses WHAM for initial phases (where LMALF may struggle with
flat biases) and switches to LMALF for later phases where its gradient-based
optimization can converge faster.
