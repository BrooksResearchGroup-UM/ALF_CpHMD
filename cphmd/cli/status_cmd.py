"""Read-only status reporting for native CpHMD run directories."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import typer

SUMMARY_STALE_SECS = 300


def report_status(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    init_marker = run_dir / "state" / "initialized.json"
    if not init_marker.exists():
        return {"run_dir": str(run_dir), "state": "not initialized"}

    summary_path = run_dir / "status_summary.json"
    if summary_path.exists() and time.time() - summary_path.stat().st_mtime < SUMMARY_STALE_SECS:
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            return {
                "run_dir": str(run_dir),
                "state": "error",
                "errors": [f"status summary corrupt at {summary_path}"],
            }
        if _summary_has_expected_ranks(summary, init_marker):
            return summary

    ranks: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    checkpoints = sorted(run_dir.rglob("res/rep*/checkpoint.json"))
    for checkpoint_file in checkpoints:
        rank_name = checkpoint_file.parent.name
        try:
            payload = json.loads(checkpoint_file.read_text())
            loop_state = payload["loop_state"]
        except json.JSONDecodeError:
            errors.append(f"checkpoint corrupt at {checkpoint_file}")
            continue
        except (KeyError, TypeError):
            errors.append(f"checkpoint missing required fields at {checkpoint_file}")
            continue

        ranks[rank_name] = {
            "segment_idx": int(loop_state.get("segment_idx", 0)),
            "phase": loop_state.get("phase", "init"),
            "rex_attempted": list(loop_state.get("rex_attempted", ())),
            "rex_accepted": list(loop_state.get("rex_accepted", ())),
        }

    if errors:
        return {"run_dir": str(run_dir), "state": "error", "errors": errors, "ranks": ranks}

    if not checkpoints:
        return {"run_dir": str(run_dir), "state": "initialized", "phase": "init", "segments": 0}

    last_checkpoint_time = max(path.stat().st_mtime for path in checkpoints)
    max_segment = max((rank["segment_idx"] for rank in ranks.values()), default=0)
    return {
        "run_dir": str(run_dir),
        "state": "running" if ranks else "initialized",
        "segments": max_segment,
        "ranks": ranks,
        "init_time": init_marker.stat().st_mtime,
        "last_checkpoint_time": last_checkpoint_time,
    }


def render_status(report: dict[str, Any]) -> str:
    state = report.get("state", "unknown")
    lines = [f"run_dir: {report.get('run_dir', '')}", f"state: {state}"]
    if state == "error":
        lines.extend(str(error) for error in report.get("errors", ()))
    if "phase" in report:
        lines.append(f"phase: {report['phase']}")
    if "segments" in report:
        lines.append(f"md_blocks: {report['segments']}")
    for rank, rank_info in sorted((report.get("ranks") or {}).items()):
        lines.append(
            f"{rank}: phase={rank_info.get('phase')} md_block={rank_info.get('segment_idx')}"
        )
    return "\n".join(lines)


def register(app: typer.Typer) -> None:
    @app.command("status")
    def _cmd(
        run_dir: Path = typer.Option(Path("."), "--run-dir", "-r", help="Run directory"),
        json_output: bool = typer.Option(False, "--json", help="Emit JSON"),
    ) -> None:
        report = report_status(run_dir)
        if json_output:
            typer.echo(json.dumps(report, indent=2, sort_keys=True))
        else:
            typer.echo(render_status(report))
        if report.get("state") == "error":
            raise typer.Exit(2)


def _summary_has_expected_ranks(summary: dict[str, Any], init_marker: Path) -> bool:
    expected = _expected_ranks(init_marker)
    if not expected:
        return True
    return expected.issubset(set((summary.get("ranks") or {}).keys()))


def _expected_ranks(init_marker: Path) -> set[str]:
    try:
        payload = json.loads(init_marker.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    try:
        nreps = int(payload.get("nreps", 0))
    except (TypeError, ValueError):
        return set()
    if nreps <= 1:
        return set()
    return {f"rep{rank:02d}" for rank in range(nreps)}
