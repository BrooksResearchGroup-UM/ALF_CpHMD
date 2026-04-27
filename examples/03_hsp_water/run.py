#!/usr/bin/env python3
"""Run CpHMD workflow from YAML config."""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cphmd.config import run_workflow

if __name__ == "__main__":
    example_dir = Path(__file__).resolve().parent
    os.chdir(example_dir)
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    run_workflow(example_dir / "cphmd_config.yaml", step)
