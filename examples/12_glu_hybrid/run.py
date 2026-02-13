#!/usr/bin/env python3
"""
Glutamic Acid ALF with Hybrid Analysis

Same system as 00_glu_water but uses a hybrid (WHAM + LMALF) approach
for bias parameter fitting.

Usage:
    python run.py solvate    # Step 1: Solvate
    python run.py patch      # Step 2: Apply titratable patches
    python run.py alf        # Step 3: Run ALF with hybrid analysis
    python run.py all        # Run all steps
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cphmd.config import run_workflow

CONFIG = SCRIPT_DIR / "cphmd_config.yaml"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    step = sys.argv[1].lower()
    valid = {"solvate", "patch", "alf", "all"}
    if step not in valid:
        print(f"Unknown step: {step}")
        print(f"Valid steps: {', '.join(sorted(valid))}")
        sys.exit(1)

    run_workflow(CONFIG, step)


if __name__ == "__main__":
    main()
