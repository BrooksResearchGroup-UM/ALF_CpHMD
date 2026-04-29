# 06 - Glutamic Acid with Hybrid Analysis

Same system as `00_glu_water` (single GLU in water) but uses a **hybrid**
analysis method that combines WHAM and LMALF approaches.

## Usage

```bash
# Optional setup refresh if prep/ is missing
python run.py setup

# Or run setup steps individually
python run.py solvate
python run.py patch

# Native ALF run with hybrid analysis
cphmd init -c cphmd_config.yaml
mpirun -np 1 cphmd run -c cphmd_config.yaml

# Slurm batch
sbatch submit.sh
```

## Native CLI

```bash
cphmd init -c cphmd_config.yaml
mpirun -np 1 cphmd run -c cphmd_config.yaml
```

Set `analysis_method: hybrid` and any hybrid-analysis options in `cphmd_config.yaml`.

## How Hybrid Works

The hybrid method uses WHAM for initial phases (where LMALF may struggle with
flat biases) and switches to LMALF for later phases where its gradient-based
optimization can converge faster.
