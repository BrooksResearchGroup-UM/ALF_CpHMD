#!/usr/bin/env python3
"""Run local setup steps for this example."""
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from _setup_workflow import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
