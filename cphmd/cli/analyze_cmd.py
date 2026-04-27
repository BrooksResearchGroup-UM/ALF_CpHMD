"""Post-run analysis commands."""

from __future__ import annotations

from pathlib import Path

import typer

from cphmd.analysis.pka_analyzer import PKaAnalysisConfig, PKaAnalyzer

analysis_app = typer.Typer(help="Analyze completed CpHMD runs.")


def run_production_pka_analysis(
    *,
    run_dirs: list[Path],
    output_dir: Path,
    name: str | None = None,
    lambda_cutoff: float = 0.97,
    skip_ps: float = 0.0,
    bin_size_ps: float = 1000.0,
    n_bootstrap: int = 1000,
    n_bootstrap_bin: int = 500,
    n_jobs: int = 1,
    min_ph_points: int = 3,
    fix_hill: bool = False,
) -> Path | None:
    cfg = PKaAnalysisConfig(
        input_folder=run_dirs,
        output_dir=output_dir,
        analysis_name=name,
        lambda_cutoff=lambda_cutoff,
        skip_ps=skip_ps,
        bin_size_ps=bin_size_ps,
        n_bootstrap=n_bootstrap,
        n_bootstrap_bin=n_bootstrap_bin,
        n_jobs=n_jobs,
        min_ph_points=min_ph_points,
        fix_hill=fix_hill,
    )
    return PKaAnalyzer(cfg).run().pka_summary_file


@analysis_app.command("production-pka")
def _production_pka(
    run_dirs: list[Path] = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Production run directories, for example sim_1 sim_2 sim_3.",
    ),
    output_dir: Path = typer.Option(Path("analysis_pka"), "-o", "--output-dir"),
    name: str | None = typer.Option(None, "--name", help="Name used in output file names."),
    lambda_cutoff: float = typer.Option(0.97, "--lambda-cutoff"),
    skip_ps: float = typer.Option(0.0, "--skip-ps"),
    bin_size_ps: float = typer.Option(1000.0, "--bin-size-ps"),
    n_bootstrap: int = typer.Option(1000, "--n-bootstrap"),
    n_bootstrap_bin: int = typer.Option(500, "--n-bootstrap-bin"),
    n_jobs: int = typer.Option(1, "-j", "--n-jobs"),
    min_ph_points: int = typer.Option(3, "--min-ph-points"),
    fix_hill: bool = typer.Option(False, "--fix-hill"),
) -> None:
    summary = run_production_pka_analysis(
        run_dirs=run_dirs,
        output_dir=output_dir,
        name=name,
        lambda_cutoff=lambda_cutoff,
        skip_ps=skip_ps,
        bin_size_ps=bin_size_ps,
        n_bootstrap=n_bootstrap,
        n_bootstrap_bin=n_bootstrap_bin,
        n_jobs=n_jobs,
        min_ph_points=min_ph_points,
        fix_hill=fix_hill,
    )
    if summary is not None:
        typer.echo(f"pKa summary: {summary}")


def register(app: typer.Typer) -> None:
    app.add_typer(analysis_app, name="analyze")
