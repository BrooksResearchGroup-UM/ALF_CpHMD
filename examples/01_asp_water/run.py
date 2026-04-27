#!/usr/bin/env python3
"""Run CpHMD workflow from YAML config."""
import os
import sys
from pathlib import Path
from cphmd.config import run_workflow
if __name__ == "__main__":
    example_dir = Path(__file__).resolve().parent
    os.chdir(example_dir)
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    run_workflow(example_dir / "cphmd_config.yaml", step)
