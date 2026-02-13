#!/usr/bin/env python3
"""Run CpHMD workflow from YAML config."""
import sys
from pathlib import Path

from cphmd.config import run_workflow

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    run_workflow(Path(__file__).parent / "cphmd_config.yaml", step)
