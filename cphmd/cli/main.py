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
    mol_type: str = typer.Option(
        "both", "-t", "--type", help="Molecule type: amino, nucleic, or both"
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing files"),
):
    """Create amino acid/nucleic acid template structures."""
    from cphmd.setup import create_all_templates

    console.print(f"[cyan]Creating {mol_type} acid templates in {output}/[/cyan]")
    results = create_all_templates(
        output_dir=output,
        molecule_type=mol_type,  # type: ignore
        overwrite=overwrite,
    )
    total = len(results["amino"]) + len(results["nucleic"])
    console.print(f"[green]Created {total} template structures[/green]")


@setup_app.command("solvate")
def solvate(
    input_file: str = typer.Option(..., "-i", "--input", help="Input structure (without extension)"),
    output: str = typer.Option("solvated", "-o", "--output", help="Output folder"),
    padding: float = typer.Option(10.0, "--pad", help="Padding around molecule (Angstroms)"),
    salt: float = typer.Option(0.10, "-s", "--salt", help="Salt concentration (M)"),
    crystal_type: str = typer.Option("OCTAHEDRAL", "--crystal", help="Crystal type"),
    temperature: float = typer.Option(298.15, "-t", "--temp", help="Temperature (K)"),
    no_ions: bool = typer.Option(False, "--no-ions", help="Skip ion placement"),
    ion_method: str = typer.Option("SLTCAP", "--ion-method", help="Ion method: AN or SLTCAP"),
):
    """Solvate a molecular system in a water box."""
    from cphmd.setup import SolvationConfig, solvate_system

    console.print(f"[cyan]Solvating {input_file} → {output}/[/cyan]")
    console.print(f"[dim]Crystal: {crystal_type}, Pad: {padding}Å, Salt: {salt}M[/dim]")

    config = SolvationConfig(
        input_file=input_file,
        output_dir=output,
        crystal_type=crystal_type,  # type: ignore
        padding=padding,
        salt_concentration=salt,
        temperature=temperature,
        skip_ions=no_ions,
        ion_method=ion_method,  # type: ignore
    )

    result_dir = solvate_system(config)
    console.print(f"[green]Solvation complete: {result_dir}/solvated.pdb[/green]")


# Run commands
@run_app.command("patch")
def patch(
    input_folder: str = typer.Option(..., "-i", "--input", help="Input folder"),
    structure: str = typer.Option("solvated", "-f", "--file", help="Structure file name"),
    hmr: bool = typer.Option(True, "--hmr/--no-hmr", help="Enable hydrogen mass repartitioning"),
    hmr_waters: bool = typer.Option(False, "--hmr-waters/--no-hmr-waters", help="Apply HMR to waters"),
    residues: list[str] = typer.Option(None, "-s", "--select", help="Residues to patch (e.g., ASP GLU PROA:15)"),
):
    """Apply CpHMD patches to titratable residues."""
    from cphmd.core import PatchConfig, patch_system

    console.print(f"[cyan]Patching titratable residues in {input_folder}/[/cyan]")
    console.print(f"[dim]HMR: {hmr}, HMR waters: {hmr_waters}[/dim]")

    config = PatchConfig(
        input_folder=input_folder,
        structure_file=structure,
        hmr=hmr,
        hmr_waters=hmr_waters,
        selected_residues=residues or [],
    )

    result_dir = patch_system(config)
    console.print(f"[green]Patching complete: {result_dir}/prep/system.pdb[/green]")


@run_app.command("alf")
def alf(
    input_folder: str = typer.Option(..., "-i", "--input", help="Input folder with prep/ directory"),
    temperature: float = typer.Option(298.15, "-t", "--temp", help="Temperature (K)"),
    pH: float = typer.Option(None, "-pH", "--pH", help="Target pH for CpHMD (None for standard ALF)"),
    hmr: bool = typer.Option(False, "--hmr/--no-hmr", help="Use hydrogen mass repartitioning"),
    start: int = typer.Option(1, "-s", "--start", help="Start run number"),
    end: int = typer.Option(20, "-e", "--end", help="End run number"),
    phase: int = typer.Option(1, "-p", "--phase", help="Initial phase (1, 2, or 3)"),
    nreps: int = typer.Option(None, "-n", "--nreps", help="Number of replicas (default: MPI size)"),
    restrains: str = typer.Option("SCAT", "-r", "--restrains", help="Restraint type: SCAT or NOE"),
):
    """Run ALF simulation with optional CpHMD.

    This command requires MPI execution:
        mpirun -np <nprocs> cphmd run alf -i <folder> [options]
    """
    from cphmd.core import ALFConfig, run_alf_simulation

    console.print(f"[cyan]Starting ALF simulation for {input_folder}/[/cyan]")
    console.print(f"[dim]Temp: {temperature}K, pH: {pH}, Phase: {phase}, Runs: {start}-{end}[/dim]")

    config = ALFConfig(
        input_folder=input_folder,
        temperature=temperature,
        pH=pH,
        hmr=hmr,
        start=start,
        end=end,
        phase=phase,  # type: ignore
        nreps=nreps,
        restrains=restrains,  # type: ignore
    )

    run_alf_simulation(config)
    console.print(f"[green]ALF simulation complete[/green]")


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
