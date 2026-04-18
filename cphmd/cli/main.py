"""Native CpHMD CLI entry point."""

from __future__ import annotations

import typer

from cphmd.cli import init_cmd, run_cmd, status_cmd

app = typer.Typer(help="CpHMD native runtime.")

init_cmd.register(app)
run_cmd.register(app)
status_cmd.register(app)


if __name__ == "__main__":
    app()
