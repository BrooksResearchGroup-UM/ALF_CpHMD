"""
Main CLI for CpHMD package.

Usage:
    cphmd setup create-aa    # Create amino acid structures
    cphmd setup solvate      # Solvate a system
    cphmd run patch          # Apply CpHMD patches
    cphmd run alf            # Run ALF simulation
    cphmd run bias-search    # Search for optimal bias
    cphmd analyze energy     # Analyze energy profiles
    cphmd analyze block      # Generate MSLD block files
    cphmd analyze volume     # Volume analysis
"""

import typer
from rich.console import Console

from cphmd import __version__

app = typer.Typer(
    name="cphmd",
    help="ALF-based Constant pH Molecular Dynamics toolkit",
    no_args_is_help=True,
)

console = Console()

# Sub-commands
setup_app = typer.Typer(help="System setup commands")
run_app = typer.Typer(help="Simulation run commands")
analyze_app = typer.Typer(help="Analysis commands")

app.add_typer(setup_app, name="setup")
app.add_typer(run_app, name="run")
app.add_typer(analyze_app, name="analyze")


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
):
    """CpHMD - Constant pH Molecular Dynamics with ALF."""
    if version:
        console.print(f"cphmd version {__version__}")
        raise typer.Exit()


# Setup commands
@setup_app.command("create-aa")
def create_aa(
    output: str = typer.Option("pdb", "-o", "--output", help="Output directory"),
):
    """Create amino acid/nucleic acid template structures."""
    console.print("[yellow]create-aa not yet implemented[/yellow]")
    # TODO: Import and call cphmd.setup.create_aa


@setup_app.command("solvate")
def solvate(
    input_file: str = typer.Option(..., "-i", "--input", help="Input PDB file"),
    output: str = typer.Option("solvated", "-o", "--output", help="Output folder"),
    padding: float = typer.Option(10.0, "--pad", help="Padding around molecule (Angstroms)"),
    salt: float = typer.Option(0.15, "-s", "--salt", help="Salt concentration (M)"),
):
    """Solvate a molecular system in a water box."""
    console.print("[yellow]solvate not yet implemented[/yellow]")
    # TODO: Import and call cphmd.setup.solvate


# Run commands
@run_app.command("patch")
def patch(
    input_folder: str = typer.Option(..., "-i", "--input", help="Input folder"),
    hmr: bool = typer.Option(True, "--hmr/--no-hmr", help="Enable hydrogen mass repartitioning"),
):
    """Apply CpHMD patches to titratable residues."""
    console.print("[yellow]patch not yet implemented[/yellow]")
    # TODO: Import and call cphmd.core.patching


@run_app.command("alf")
def alf(
    config: str = typer.Option(None, "-c", "--config", help="Configuration YAML file"),
    replicas: int = typer.Option(8, "-n", "--replicas", help="Number of replicas"),
):
    """Run ALF simulation."""
    console.print("[yellow]alf not yet implemented[/yellow]")
    # TODO: Import and call cphmd.core.alf_runner


@run_app.command("bias-search")
def bias_search(
    input_folder: str = typer.Option(..., "-i", "--input", help="Analysis folder"),
    cutoff: float = typer.Option(0.985, "-c", "--cutoff", help="Physical state cutoff"),
):
    """Search for optimal bias parameters."""
    console.print("[yellow]bias-search not yet implemented[/yellow]")
    # TODO: Import and call cphmd.core.bias_search


# Analysis commands
@analyze_app.command("energy")
def energy_profiles(
    input_folder: str = typer.Option(..., "-i", "--input", help="Analysis folder"),
):
    """Analyze and visualize energy profiles."""
    console.print("[yellow]energy analysis not yet implemented[/yellow]")
    # TODO: Import and call cphmd.analysis.energy_profiles


@analyze_app.command("block")
def block(
    input_folder: str = typer.Option(..., "-i", "--input", help="Input folder"),
    restrain_type: str = typer.Option("SCAT", "--restrain-type", help="SCAT or NOE"),
):
    """Generate MSLD block and restraint files."""
    console.print("[yellow]block generation not yet implemented[/yellow]")
    # TODO: Import and call cphmd.analysis.block_generator


@analyze_app.command("volume")
def volume(
    input_folder: str = typer.Option(..., "-i", "--input", help="Input folder"),
):
    """Calculate system volume properties."""
    console.print("[yellow]volume analysis not yet implemented[/yellow]")
    # TODO: Import and call cphmd.analysis.volume


if __name__ == "__main__":
    app()
