#!/usr/bin/env python3
"""Fail if chunk-per-sbatch legacy orchestration patterns remain."""

from __future__ import annotations

import re
import sys
from pathlib import Path

FORBIDDEN = [
    re.compile(r"\brun_chunk\b"),
    re.compile(r"\bsubmit_chunk\b"),
    re.compile(r"chunk_idx.*sbatch"),
    re.compile(r"typer\.Argument.*chunk"),
    re.compile(r"--chunk\b"),
]

ROOT = Path(__file__).resolve().parent.parent
SEARCH_ROOTS = (ROOT / "cphmd", ROOT / "scripts")
EXCLUDE_DIRS = {"tests", "docs", "specs", ".worktrees", ".venv", "__pycache__"}
EXTS = {".py", ".sh"}


def main() -> int:
    hits: list[tuple[Path, int, str, str]] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            if any(part in EXCLUDE_DIRS for part in path.parts):
                continue
            if path.suffix not in EXTS:
                continue
            if path.name == "check_no_legacy_chunk_sbatch.py":
                continue
            text = path.read_text(errors="ignore")
            for line_no, line in enumerate(text.splitlines(), 1):
                for pattern in FORBIDDEN:
                    if pattern.search(line):
                        hits.append((path, line_no, pattern.pattern, line.rstrip()))

    if hits:
        for path, line_no, pattern, line in hits:
            print(f"{path}:{line_no}: matches /{pattern}/: {line}", file=sys.stderr)
        return 1
    print("OK: no legacy chunk-per-sbatch patterns found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
