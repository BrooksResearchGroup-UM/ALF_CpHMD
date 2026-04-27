#!/usr/bin/env python3
"""Run CpHMD workflow from YAML config.

For MPI jobs (alf step), mpi4py must be imported BEFORE pyCHARMM
so that MPI is already initialized when CHARMM's C library loads.
"""
import os
import sys
from pathlib import Path

# Initialize MPI early (before any pyCHARMM import triggered by cphmd.core).
# Only needed for the 'alf' step; solvate/patch/build are serial.
step = sys.argv[1] if len(sys.argv) > 1 else "all"
if step in ("alf", "all"):
    from mpi4py import MPI  # noqa: F401

from cphmd.config import run_workflow

if __name__ == "__main__":
    example_dir = Path(__file__).resolve().parent
    os.chdir(example_dir)
    run_workflow(example_dir / "cphmd_config.yaml", step)
