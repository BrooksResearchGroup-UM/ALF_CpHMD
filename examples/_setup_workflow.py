"""Shared setup helper for example ``run.py`` scripts.

ALF and production MD should be launched through ``cphmd init`` and
``cphmd run`` so MPI and GPU assignment are explicit.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SETUP_STEPS = ("build", "prepare", "solvate", "patch")


def main(example_dir: Path) -> None:
    example_dir = example_dir.resolve()
    repo_root = example_dir.parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from cphmd.config import run_workflow

    os.chdir(example_dir)
    step = sys.argv[1].lower() if len(sys.argv) > 1 else "setup"
    config = example_dir / "cphmd_config.yaml"

    if step in {"alf", "run"}:
        raise SystemExit(
            "Use native launch for ALF:\n"
            "  cphmd init -c cphmd_config.yaml\n"
            "  mpirun -np <nreps> cphmd run -c cphmd_config.yaml"
        )

    if step in {"setup", "all"}:
        for setup_step in SETUP_STEPS:
            try:
                run_workflow(config, setup_step)
            except ValueError as exc:
                if f"no '{setup_step}' section" in str(exc):
                    continue
                raise
        return

    if step not in SETUP_STEPS:
        valid = ", ".join(("setup", *SETUP_STEPS))
        raise SystemExit(f"Unknown setup step {step!r}. Use one of: {valid}")

    run_workflow(config, step)
