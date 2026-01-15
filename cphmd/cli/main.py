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
    restrain_hydrogens: bool = typer.Option(False, "-H", "--hydrogens", help="Include hydrogens in restraints"),
    elec_type: str = typer.Option("pmeex", "--elec", help="Electrostatics: pmeex, pmeon, pmenn, fshift, fswitch"),
    vdw_type: str = typer.Option("vswitch", "--vdw", help="VDW method: vswitch or vfswitch"),
):
    """Run ALF simulation with optional CpHMD.

    This command requires MPI execution:
        mpirun -np <nprocs> cphmd run alf -i <folder> [options]
    """
    from cphmd.core import ALFConfig, run_alf_simulation

    console.print(f"[cyan]Starting ALF simulation for {input_folder}/[/cyan]")
    console.print(f"[dim]Temp: {temperature}K, pH: {pH}, Phase: {phase}, Runs: {start}-{end}[/dim]")
    console.print(f"[dim]Restraints: {restrains}, Hydrogens: {restrain_hydrogens}[/dim]")
    console.print(f"[dim]Electrostatics: {elec_type}, VDW: {vdw_type}[/dim]")

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
        restrain_hydrogens=restrain_hydrogens,
        elec_type=elec_type,  # type: ignore
        vdw_type=vdw_type,  # type: ignore
    )

    run_alf_simulation(config)
    console.print(f"[green]ALF simulation complete[/green]")


@run_app.command("bias-search")
def bias_search(
    input_folder: str = typer.Option(..., "-i", "--input", help="Input folder with analysis directories"),
    cutoff: float = typer.Option(0.985, "-c", "--cutoff", help="Lambda cutoff for population counting"),
    adjustment: str = typer.Option("0", "-a", "--adjust", help="Bias adjustment: + (positive), - (negative), 0 (none)"),
    temperature: float = typer.Option(298.15, "-t", "--temp", help="Temperature (K)"),
    alpha: float = typer.Option(10.0, "--alpha", help="Imbalance penalty factor"),
    output_dir: str = typer.Option("variables", "-o", "--output", help="Output directory for variables file"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Print detailed output"),
    no_plot: bool = typer.Option(False, "--no-plot", help="Skip plot generation"),
):
    """Search for optimal bias parameters from ALF simulation results.

    Analyzes Lambda files from analysis directories to find runs with
    the best population balance across substituents.
    """
    from cphmd.core import BiasSearchConfig, run_bias_search

    console.print(f"[cyan]Running bias search for {input_folder}/[/cyan]")
    console.print(f"[dim]Cutoff: {cutoff}, Alpha: {alpha}, Adjustment: {adjustment}[/dim]")

    try:
        config = BiasSearchConfig(
            input_folder=input_folder,
            cutoff=cutoff,
            adjustment=adjustment,
            temperature=temperature,
            alpha=alpha,
            verbose=verbose,
            plot=not no_plot,
            output_dir=output_dir,
        )

        result = run_bias_search(config)
        console.print(f"[green]Best run: {result.best_iteration} (score: {result.best_score:.4f})[/green]")
        if result.variables_file:
            console.print(f"[green]Variables saved to: {result.variables_file}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


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
    input_folder: str = typer.Option(..., "-i", "--input", help="Input folder with prep/patches.dat"),
    restrain_type: str = typer.Option("SCAT", "--restrain-type", help="SCAT or NOE"),
    hydrogens: bool = typer.Option(False, "-H", "--hydrogens", help="Include hydrogens in restraints"),
    electrostatics: str = typer.Option("pmeex", "-e", "--elec", help="PME method: pmeex, pmeon, pmenn"),
    variables_dir: str = typer.Option("variables", "-v", "--var-dir", help="Variables directory"),
):
    """Generate MSLD block and restraint files for CpHMD production runs.

    Reads patches.dat and generates block.str (MSLD setup) and restrains.str
    (SCAT or NOE restraints) needed for production CpHMD simulations.
    """
    from cphmd.core import BlockGeneratorConfig, generate_block_files

    console.print(f"[cyan]Generating block files for {input_folder}/[/cyan]")
    console.print(f"[dim]Restraint: {restrain_type}, Hydrogens: {hydrogens}[/dim]")

    try:
        config = BlockGeneratorConfig(
            input_folder=input_folder,
            restrain_type=restrain_type,
            include_hydrogens=hydrogens,
            electrostatics=electrostatics,
            variables_dir=variables_dir,
        )

        result = generate_block_files(config)
        console.print(f"[green]Generated {result.n_blocks} blocks for {result.n_sites} sites[/green]")
        console.print(f"[green]Block file: {result.block_file}[/green]")
        console.print(f"[green]Restraint file: {result.restraint_file}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@analyze_app.command("volume")
def volume(
    input_folder: str = typer.Option(..., "-i", "--input", help="Input folder"),
):
    """Calculate system volume properties."""
    console.print("[yellow]volume analysis not yet implemented[/yellow]")
    # TODO: Import and call cphmd.analysis.volume


if __name__ == "__main__":
    app()
