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
utils_app = typer.Typer(help="Utility commands")

app.add_typer(setup_app, name="setup")
app.add_typer(run_app, name="run")
app.add_typer(analyze_app, name="analyze")
app.add_typer(utils_app, name="utils")


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
    input_file: str = typer.Option(None, "-i", "--input", help="Input structure (without extension)"),
    output: str = typer.Option(None, "-o", "--output", help="Output folder"),
    padding: float = typer.Option(None, "--pad", help="Padding around molecule (Angstroms)"),
    salt: float = typer.Option(None, "-s", "--salt", help="Salt concentration (M)"),
    crystal_type: str = typer.Option(None, "--crystal", help="Crystal type"),
    temperature: float = typer.Option(None, "-t", "--temp", help="Temperature (K)"),
    no_ions: bool = typer.Option(None, "--no-ions", help="Skip ion placement"),
    ion_method: str = typer.Option(None, "--ion-method", help="Ion method: AN or SLTCAP"),
    config: str = typer.Option(None, "-c", "--config", help="YAML config file"),
):
    """Solvate a molecular system in a water box."""
    from cphmd.config import config_to_solvation

    cli = {
        "input_file": input_file,
        "output_dir": output,
        "padding": padding,
        "salt": salt,
        "crystal_type": crystal_type,
        "temperature": temperature,
        "skip_ions": no_ions,
        "ion_method": ion_method,
    }
    solvation_config = config_to_solvation(config, cli)

    console.print(f"[cyan]Solvating {solvation_config.input_file} → {solvation_config.output_dir}/[/cyan]")

    from cphmd.setup import solvate_system

    result_dir = solvate_system(solvation_config)
    console.print(f"[green]Solvation complete: {result_dir}/solvated.pdb[/green]")


# Run commands
@run_app.command("patch")
def patch(
    input_folder: str = typer.Option(None, "-i", "--input", help="Input folder"),
    structure: str = typer.Option(None, "-f", "--file", help="Structure file name"),
    hmr: bool = typer.Option(None, "--hmr/--no-hmr", help="Enable hydrogen mass repartitioning"),
    hmr_waters: bool = typer.Option(None, "--hmr-waters/--no-hmr-waters", help="Apply HMR to waters"),
    residues: list[str] = typer.Option(None, "-s", "--select", help="Residues to patch (e.g., ASP GLU PROA:15)"),
    extra_files: list[str] = typer.Option(None, "--extra-files", help="Extra topology/parameter files (repeatable)"),
    toppar_dir: str = typer.Option(None, "--toppar-dir", help="Topology/parameter directory (default: bundled toppar)"),
    config: str = typer.Option(None, "-c", "--config", help="YAML config file"),
):
    """Apply CpHMD patches to titratable residues."""
    from cphmd.config import config_to_patch

    cli = {
        "input_folder": input_folder,
        "structure_file": structure,
        "hmr": hmr,
        "hmr_waters": hmr_waters,
        "selected_residues": residues,
        "extra_files": extra_files,
        "toppar_dir": toppar_dir,
    }
    patch_config = config_to_patch(config, cli)

    console.print(f"[cyan]Patching titratable residues in {patch_config.input_folder}/[/cyan]")
    console.print(f"[dim]HMR: {patch_config.hmr}, HMR waters: {patch_config.hmr_waters}[/dim]")

    from cphmd.core import patch_system

    result_dir = patch_system(patch_config)
    console.print(f"[green]Patching complete: {result_dir}/prep/system.pdb[/green]")


def _parse_g_imp_bins(value: str | None) -> "int | list[int] | None":
    """Parse --g-imp-bins CLI string into int, list[int], or None."""
    if value is None:
        return None
    if "," not in value:
        return int(value)
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise typer.BadParameter(
            "Per-phase g_imp_bins must have exactly 3 values (phase 1,2,3)"
        )
    return parts


@run_app.command("alf")
def alf(
    input_folder: str = typer.Option(None, "-i", "--input", help="Input folder with prep/ directory"),
    temperature: float = typer.Option(None, "-t", "--temp", help="Temperature (K)"),
    pH: bool = typer.Option(None, "--pH/--no-pH", help="Enable CpHMD pH coupling (effective_pH auto-computed from pKa)"),
    hmr: bool = typer.Option(None, "--hmr/--no-hmr", help="Use hydrogen mass repartitioning"),
    start: int = typer.Option(None, "-s", "--start", help="Start run number"),
    end: int = typer.Option(None, "-e", "--end", help="End run number"),
    phase: int = typer.Option(None, "-p", "--phase", help="Initial phase (1, 2, or 3)"),
    nreps: int = typer.Option(None, "-n", "--nreps", help="Number of replicas (default: MPI size)"),
    restrains: str = typer.Option(None, "-r", "--restrains", help="Restraint type: SCAT or NOE"),
    restrain_hydrogens: bool = typer.Option(None, "--hydrogens/--no-hydrogens", help="Include hydrogens in restraints"),
    no_pka_bias: bool = typer.Option(None, "--no-pka-bias/--pka-bias", help="Disable pKa-based bias shifts (use zero shifts)"),
    auto_phase: bool = typer.Option(None, "--auto-phase/--no-auto-phase", help="Enable automatic phase switching"),
    auto_stop: bool = typer.Option(None, "--auto-stop/--no-auto-stop", help="Enable automatic stop when converged in Phase 3"),
    convergence_mode: str = typer.Option(None, "--convergence-mode", help="Convergence mode: population or rmsd"),
    hh_plots: bool = typer.Option(None, "--hh-plots/--no-hh-plots", help="Generate Henderson-Hasselbalch plots"),
    cleanup: bool = typer.Option(None, "--cleanup/--no-cleanup", help="Remove old analysis directories"),
    elec_type: str = typer.Option(None, "--elec", help="Electrostatics: pmeex, pmeon, pmenn, fshift, fswitch"),
    vdw_type: str = typer.Option(None, "--vdw", help="VDW method: vswitch or vfswitch"),
    coupling: int = typer.Option(None, "--coupling", help="Inter-site coupling: 0=none, 1=full, 2=c-only"),
    coupling_profile: bool = typer.Option(None, "--coupling-profile/--no-coupling-profile", help="Inter-site profile monitoring (default: follows coupling)"),
    analysis_method: str = typer.Option(None, "--analysis-method", help="Analysis method: wham, lmalf, hybrid, or nonlinear"),
    lmalf_max_iter: int = typer.Option(None, "--lmalf-max-iter", help="LMALF max iterations (0=default)"),
    lmalf_tolerance: float = typer.Option(None, "--lmalf-tolerance", help="LMALF tolerance (0=default)"),
    lambda_mass: float = typer.Option(None, "--lambda-mass", help="Lambda mass in amu*A^2"),
    lambda_fbeta: float = typer.Option(None, "--lambda-fbeta", help="Lambda friction in ps^-1"),
    fnex: float = typer.Option(None, "--fnex", help="FNEX softmax constraint parameter"),
    gscale: float = typer.Option(None, "--gscale", help="Global Langevin friction coefficient (ps^-1)"),
    extra_files: list[str] = typer.Option(None, "--extra-files", help="Extra topology/parameter files (repeatable)"),
    g_imp_bins: str = typer.Option(None, "--g-imp-bins", help="G_imp bins: single int or comma-separated per-phase (e.g. '16,32,32')"),
    cutlsum: float = typer.Option(None, "--cutlsum", help="G12 conditional threshold"),
    config: str = typer.Option(None, "-c", "--config", help="YAML config file"),
):
    """Run ALF simulation with optional CpHMD.

    This command requires MPI execution:
        mpirun -np <nprocs> cphmd run alf -i <folder> [options]
    """
    # Initialize MPI via mpi4py BEFORE importing pyCHARMM (triggered by cphmd.core).
    # pyCHARMM detects MPI is already initialized and skips its own MPI_Init.
    from mpi4py import MPI  # noqa: F401 — side-effect import, must precede pyCHARMM

    from cphmd.config import config_to_alf

    cli = {
        "input_folder": input_folder,
        "temperature": temperature,
        "pH": pH,
        "hmr": hmr,
        "start": start,
        "end": end,
        "phase": phase,
        "nreps": nreps,
        "restrains": restrains,
        "restrain_hydrogens": restrain_hydrogens,
        "no_pka_bias": no_pka_bias,
        "auto_phase": auto_phase,
        "auto_stop": auto_stop,
        "convergence_mode": convergence_mode,
        "hh_plots": hh_plots,
        "cleanup": cleanup,
        "elec_type": elec_type,
        "vdw_type": vdw_type,
        "coupling": coupling,
        "coupling_profile": coupling_profile,
        "analysis_method": analysis_method,
        "lmalf_max_iter": lmalf_max_iter,
        "lmalf_tolerance": lmalf_tolerance,
        "lambda_mass": lambda_mass,
        "lambda_fbeta": lambda_fbeta,
        "fnex": fnex,
        "gscale": gscale,
        "extra_files": extra_files,
        "g_imp_bins": _parse_g_imp_bins(g_imp_bins),
        "cutlsum": cutlsum,
    }
    alf_config = config_to_alf(config, cli)

    # Only rank 0 prints status (avoids 5× duplicated output under MPI)
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        console.print(f"[cyan]Starting ALF simulation for {alf_config.input_folder}/[/cyan]")
        cphmd_str = "CpHMD" if alf_config.pH else "ALF"
        console.print(f"[dim]Temp: {alf_config.temperature}K, Mode: {cphmd_str}, Phase: {alf_config.phase}, Runs: {alf_config.start}-{alf_config.end}[/dim]")
        console.print(f"[dim]HMR: {alf_config.hmr}, Restraints: {alf_config.restrains}[/dim]")
        console.print(f"[dim]Electrostatics: {alf_config.elec_type}, VDW: {alf_config.vdw_type}[/dim]")
        if alf_config.no_pka_bias:
            console.print("[yellow]pKa bias disabled[/yellow]")
        if alf_config.auto_phase_switch:
            console.print("[green]Automatic phase switching enabled[/green]")
        if alf_config.auto_stop:
            console.print("[green]Automatic stop on convergence enabled[/green]")
        if alf_config.coupling > 0:
            mode = "full" if alf_config.coupling == 1 else "c-only"
            console.print(f"[cyan]Inter-site coupling: {mode}[/cyan]")
        if alf_config.analysis_method == "lmalf":
            console.print("[cyan]Using LMALF analysis method[/cyan]")
        elif alf_config.analysis_method == "nonlinear":
            console.print("[cyan]Using nonlinear L-BFGS analysis method[/cyan]")

    from cphmd.core import run_alf_simulation

    run_alf_simulation(alf_config)
    if comm.Get_rank() == 0:
        console.print("[green]ALF simulation complete[/green]")


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
def energy_profiles_cmd(
    input_folder: str = typer.Option(..., "-i", "--input", help="Folder with analysis directories"),
    output_dir: str = typer.Option("plots", "-o", "--output", help="Output directory for plots"),
    num_points: int = typer.Option(100, "-n", "--num-points", help="Grid points per dimension"),
    plot_format: str = typer.Option("png", "-f", "--format", help="Plot format: png, pdf, svg"),
    animation: bool = typer.Option(False, "--animation", help="Create animation (3D only)"),
):
    """Analyze and visualize ALF energy profiles across iterations.

    Computes bias energy landscapes on simplex grid and tracks
    RMSD convergence between iterations. Supports 2-state and 3-state systems.
    """
    from cphmd.analysis import EnergyProfileConfig, analyze_energy_profiles

    console.print(f"[cyan]Analyzing energy profiles in {input_folder}/[/cyan]")
    console.print(f"[dim]Grid: {num_points} points, Format: {plot_format}[/dim]")

    try:
        config = EnergyProfileConfig(
            input_folder=input_folder,
            num_points=num_points,
            output_dir=output_dir,
            create_animation=animation,
            plot_format=plot_format,
        )

        result = analyze_energy_profiles(config)
        console.print(f"[green]Analyzed {len(result.iterations)} iterations ({result.n_states} states)[/green]")
        console.print(f"[green]Final RMSD: {result.rmsd_values[-1]:.4f} kcal/mol[/green]")
        console.print(f"[green]RMSD plot: {result.rmsd_plot}[/green]")
        if result.profile_plot:
            console.print(f"[green]Profile plot: {result.profile_plot}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@analyze_app.command("block")
def block(
    input_folder: str = typer.Option(..., "-i", "--input", help="Input folder with prep/patches.dat"),
    restrain_type: str = typer.Option("SCAT", "--restrain-type", help="SCAT or NOE"),
    hydrogens: bool = typer.Option(False, "--hydrogens/--no-hydrogens", help="Include hydrogens in restraints"),
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
def volume_cmd(
    input_folder: str = typer.Option(..., "-i", "--input", help="System folder"),
    structure: str = typer.Option("solvated", "-f", "--file", help="Structure file name"),
    selection: str = typer.Option("sele segid PROA end", "-s", "--selection", help="CHARMM selection"),
    probe_radius: float = typer.Option(1.6, "-r", "--radius", help="Probe radius (Angstroms)"),
):
    """Calculate molecular volume using pyCHARMM.

    Computes total and cavity volumes for a molecular selection
    using CHARMM's COOR VOLU command. Requires pyCHARMM environment.
    """
    from cphmd.analysis import VolumeConfig, calculate_volume

    console.print(f"[cyan]Analyzing volumes in {input_folder}/[/cyan]")
    console.print(f"[dim]Selection: {selection}, Probe: {probe_radius} Å[/dim]")

    try:
        config = VolumeConfig(
            input_folder=input_folder,
            structure_file=structure,
            selection=selection,
            probe_radius=probe_radius,
        )

        result = calculate_volume(config)
        console.print(f"[green]Total volume: {result.total_volume}[/green]")
        console.print(f"[green]Cavity volume: {result.cavity_volume}[/green]")
        if result.surface_area:
            console.print(f"[green]Surface area: {result.surface_area}[/green]")
    except ImportError as e:
        console.print(f"[yellow]Note: {e}[/yellow]")
        console.print("[yellow]Volume analysis requires pyCHARMM environment.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


# Utility commands
@utils_app.command("lambda-convert")
def lambda_convert(
    input_files: list[str] = typer.Argument(..., help="Input lambda files (.lmd)"),
    output: str = typer.Option(None, "-o", "--output", help="Output directory or file"),
    compression: str = typer.Option("snappy", "-c", "--compression",
                                    help="Compression: snappy, gzip, lz4, zstd"),
    concat: bool = typer.Option(False, "--concat", help="Concatenate all inputs"),
):
    """Convert CHARMM binary lambda files to Parquet format.

    Parquet files are ~8x smaller and ~17x faster to read than binary.
    Use --concat to combine multiple files into one output.
    """
    from pathlib import Path

    from cphmd.utils import concatenate_lambda_files, convert_lambda_to_parquet

    if concat:
        # Concatenate mode
        output_path = output or "combined.parquet"
        combined = concatenate_lambda_files(input_files, output_path)
        console.print(f"[green]Combined {len(input_files)} files -> {output_path}[/green]")
        console.print(f"[dim]Total steps: {len(combined)}[/dim]")
    else:
        # Individual conversion
        for input_file in input_files:
            input_path = Path(input_file)
            if output:
                output_dir = Path(output)
                if output_dir.is_dir():
                    output_path = output_dir / input_path.with_suffix('.parquet').name
                else:
                    output_path = output_dir
            else:
                output_path = input_path.with_suffix('.parquet')

            result = convert_lambda_to_parquet(input_path, output_path, compression)
            console.print(f"[green]Converted: {input_path} -> {result}[/green]")


@utils_app.command("lambda-info")
def lambda_info(
    filepath: str = typer.Argument(..., help="Lambda file (.lmd or .parquet)"),
):
    """Show information about a lambda file."""
    from pathlib import Path

    from cphmd.utils import read_lambda_binary, read_lambda_parquet

    path = Path(filepath)
    console.print(f"[cyan]Lambda file: {path}[/cyan]")

    if path.suffix == '.lmd':
        data, meta = read_lambda_binary(path)
        console.print("[dim]Format: CHARMM binary[/dim]")
        console.print(f"  Steps: {meta.nfile}")
        console.print(f"  Blocks: {meta.nblocks}")
        console.print(f"  Sites: {meta.nsitemld}")
        console.print(f"  Time step: {meta.delta_t:.3f} ps")
        console.print(f"  Save freq: {meta.nsavl}")
        console.print(f"  Temperature: {meta.temp:.1f} K")
        console.print(f"  Title: {meta.title}")
    elif path.suffix == '.parquet':
        data = read_lambda_parquet(path)
        console.print("[dim]Format: Parquet[/dim]")
        console.print(f"  Steps: {len(data)}")
        console.print(f"  Columns: {data.shape[1]}")
        console.print(f"  Time range: {data[0, 0]:.1f} - {data[-1, 0]:.1f} ps")
    else:
        console.print(f"[red]Unknown format: {path.suffix}[/red]")
        raise typer.Exit(1)


@run_app.command("workflow")
def workflow(
    config: str = typer.Option(..., "-c", "--config", help="YAML config file"),
    step: str = typer.Option("all", "--step", help="Step to run: build, solvate, patch, alf, or all"),
):
    """Run a CpHMD workflow from a YAML config file.

    Executes one or more workflow steps in order: build -> solvate -> patch -> alf.
    Use --step to run a single step, or "all" to run the full pipeline.
    Steps without a config section are skipped when running "all".
    """
    from cphmd.config import run_workflow as _run_workflow

    console.print(f"[cyan]Running workflow step '{step}' from {config}[/cyan]")
    _run_workflow(config, step)
    console.print(f"[green]Workflow '{step}' complete[/green]")


if __name__ == "__main__":
    app()
