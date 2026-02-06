"""
ALF (Alchemical Lambda Free energy) simulation runner for CpHMD.

This module implements MPI-parallel constant-pH molecular dynamics (CpHMD) simulations
using the ALF method with Henderson-Hasselbalch curve analysis.

Key Features:
- MPI parallelization with automatic GPU assignment
- Phase-based simulation progression (equilibration → refinement → production)
- CpHMD with micro-pKa parameter computation
- Henderson-Hasselbalch curve generation and fitting

Architecture:
- ALFConfig: Configuration dataclass for simulation parameters
- ALFSimulation: Main orchestrator class managing the simulation lifecycle
- CpHMDParameters: Helper class for pH-dependent bias calculations
- HendersonHasselbalch: Fitting utilities for titration curves
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Any
import os
import sys
import time
import shutil

import numpy as np
import pandas as pd

from cphmd.core.phase_switcher import (
    _per_site_ranges,
    check_phase_transition,
    check_phase3_stop,
    load_lambda_data,
    calculate_populations,
    write_populations_file,
    PhaseTransitionConfig,
    StopCriteriaConfig,
)

from cphmd.core.alf_utils import (
    ALFInfo,
    ensure_alf_info,
    init_vars,
    set_vars_from_analysis_dir,
    get_energy_from_analysis_dir,
    convert_lambda_binary_to_text,
)
from cphmd.core.free_energy import get_free_energy5

# Type aliases
PhaseType = Literal[1, 2, 3]
RestrainType = Literal["SCAT", "NOE"]
ElecType = Literal["pmeex", "pmeon", "pmenn", "fshift", "fswitch"]
VdwType = Literal["vswitch", "vfswitch"]
AnalysisMethod = Literal["wham", "lmalf", "hybrid"]
ConvergenceMode = Literal["population", "rmsd"]
PrepFormat = Literal["default", "legacy", "auto"]


@dataclass
class ALFConfig:
    """Configuration for ALF simulation.

    Attributes:
        input_folder: Path to the prepared system folder (contains prep/ subdirectory)
        toppar_dir: Path to topology/parameter files
        temperature: Simulation temperature in Kelvin
        pH: Target pH for CpHMD (None for standard ALF without pH coupling)
        hmr: Whether to use hydrogen mass repartitioning (4fs timestep).
             None = auto (True for default prep, False for legacy prep).
        start: Starting run number
        end: Ending run number
        phase: Initial simulation phase (1, 2, or 3)
        nreps: Number of replicas to run
        restrains: Restraint method for titratable atoms ("SCAT" or "NOE")
        restrain_hydrogens: Include hydrogens in restraints (default False)
        no_x_bias: Disable skew bias updates
        no_s_bias: Disable endpoint bias updates
        cent_ncres: Number of residues for recentering (False to disable)
        elec_type: Electrostatics method (pmeex, pmeon, pmenn, fshift, fswitch)
        vdw_type: VDW method (vswitch, vfswitch).
                  None = auto (vswitch for default, vfswitch for legacy).
        gscale: Langevin friction coefficient (fbeta) applied to all atoms.
                None = auto (10.0 for default, 0.1 for legacy).
    """
    input_folder: str | Path
    toppar_dir: str | Path = "toppar"
    temperature: float = 298.15
    pH: float | None = None
    hmr: bool | None = None
    start: int = 1
    end: int = 20
    phase: PhaseType = 1
    nreps: int | None = None  # Defaults to MPI size
    restrains: RestrainType = "SCAT"
    restrain_hydrogens: bool = False
    no_b_bias: bool = False  # Disable linear (phi) bias updates
    no_c_bias: bool = False  # Disable coupling (psi) bias updates
    no_x_bias: bool = False
    no_s_bias: bool = False
    no_pka_bias: bool = False  # Disable pKa-based bias shifts
    auto_phase_switch: bool = False  # Enable automatic phase switching
    auto_stop: bool = False  # Enable automatic stop when converged in Phase 3
    convergence_mode: ConvergenceMode = "population"  # "population" or "rmsd"
    cleanup_old_analysis: bool = False  # Remove old analysis directories to save disk space
    generate_hh_plots: bool = False  # Generate Henderson-Hasselbalch plots
    cent_ncres: int | bool = False

    # Inter-site coupling
    coupling: Literal[0, 1, 2] = 0  # 0=none, 1=full c/x/s, 2=c-only
    coupling_profile: bool | None = None  # None=follow coupling, True/False=override

    # FNEX softmax constraint parameter (controls bias potential shape)
    fnex: float = 5.5

    # Lambda dynamics mass and friction (per-block via LDIN)
    # None = auto: HMR(4fs) → mass=12.0/fbeta=5.0; non-HMR(2fs) → mass=5.0/fbeta=7.0
    lambda_mass: float | None = None   # Lambda mass in amu·Å² (BIMLAM)
    lambda_fbeta: float | None = None  # Lambda Langevin friction in ps⁻¹ (BIBLAM)

    # Analysis method configuration
    analysis_method: AnalysisMethod = "wham"  # "wham" or "lmalf"
    lmalf_max_iter: int = 0  # Maximum L-BFGS iterations (0 = use default)
    lmalf_tolerance: float = 0.0  # Convergence tolerance (0 = use default)

    # Non-bonded parameters
    cutnb: float = 14.0
    ctofnb: float = 12.0
    ctonnb: float = 10.0
    elec_type: ElecType = "pmeex"
    vdw_type: VdwType | None = None
    gscale: float | None = None

    # Topology files (relative to toppar_dir)
    topology_files: list[str] = field(default_factory=lambda: [
        "top_all36_prot.rtf",
        "par_all36m_prot.prm",
        "top_all36_na.rtf",
        "par_all36_na.prm",
        "toppar_water_ions.str",
        "top_all36_cgenff.rtf",
        "par_all36_cgenff.prm",
        "my_files/titratable_residues.str",
        "my_files/nucleic_c36.str",
    ])

    # Extra topology/parameter files (absolute paths for custom ligands)
    extra_files: list[str | Path] = field(default_factory=list)

    # Prep format: "default" (CpHMD), "legacy" (msld-py-prep), or "auto" (detect)
    prep_format: PrepFormat = "auto"
    # Legacy setup script (auto-discovered from prep/*.inp if not set)
    legacy_setup_script: str | None = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.input_folder = Path(self.input_folder)
        self.toppar_dir = Path(self.toppar_dir)

        prep_dir = self.input_folder / "prep"

        # Auto-detect prep format
        if self.prep_format == "auto":
            has_patches = (prep_dir / "patches.dat").exists()
            has_alf_info = (prep_dir / "alf_info.py").exists()
            if has_patches:
                self.prep_format = "default"
            elif has_alf_info:
                self.prep_format = "legacy"
            else:
                raise FileNotFoundError(
                    f"Cannot detect prep format in {prep_dir}: "
                    f"expected patches.dat (default) or alf_info.py (legacy)"
                )

        # Validate based on format
        if self.prep_format == "default":
            for f in ["prep/system.psf", "prep/system.crd", "prep/patches.dat",
                       "prep/box.dat", "prep/fft.dat"]:
                path = self.input_folder / f
                if not path.exists():
                    raise FileNotFoundError(f"Required file not found: {path}")
        elif self.prep_format == "legacy":
            if not (prep_dir / "alf_info.py").exists():
                raise FileNotFoundError(f"Required file not found: {prep_dir / 'alf_info.py'}")
            # Auto-discover setup script if not specified
            if self.legacy_setup_script is None:
                self.legacy_setup_script = self._find_legacy_setup_script(prep_dir)

        # Resolve None sentinels based on prep format.
        # Legacy (msld-py-prep) systems: no HMR PSF, use vfswitch, low friction.
        is_legacy = self.prep_format == "legacy"
        if self.hmr is None:
            self.hmr = not is_legacy           # True for default, False for legacy
        if self.vdw_type is None:
            self.vdw_type = "vfswitch" if is_legacy else "vswitch"
        if self.gscale is None:
            self.gscale = 0.1 if is_legacy else 10.0

        # Lambda mass/friction: HMR(4fs) keeps heavy/gentle defaults;
        # non-HMR(2fs) uses lighter mass (Kramers ∝ 1/M) with scaled friction
        # γ(M) = γ₀·√(M₀/M) for near-critical damping.
        if self.lambda_mass is None:
            self.lambda_mass = 12.0 if self.hmr else 5.0
        if self.lambda_fbeta is None:
            self.lambda_fbeta = 5.0 if self.hmr else 7.0

    @staticmethod
    def _find_legacy_setup_script(prep_dir: Path) -> str:
        """Find the main MSLD setup .inp script in a legacy prep directory.

        Excludes known helper files (lpsites.inp, toppar.str, etc.).
        """
        exclude = {"lpsites.inp", "toppar.str"}
        candidates = [
            f.name for f in prep_dir.glob("*.inp")
            if f.name not in exclude
        ]
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) == 0:
            raise FileNotFoundError(
                f"No .inp setup script found in {prep_dir}. "
                f"Set legacy_setup_script in ALFConfig."
            )
        else:
            raise FileNotFoundError(
                f"Multiple .inp files in {prep_dir}: {candidates}. "
                f"Set legacy_setup_script in ALFConfig to specify which one."
            )


@dataclass
class SimulationState:
    """Runtime state for ALF simulation.

    Managed internally - not for direct user configuration.
    """
    rank: int = 0
    size: int = 1
    gpuid: int = 0
    phase: PhaseType = 1
    current_run: int = 1
    structure_loaded: bool = False  # Track if PSF/coordinates have been loaded
    box_size: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    box_angles: list[float] = field(default_factory=lambda: [90.0, 90.0, 90.0])
    crystal_type: str = ""
    restart_run: int | None = None

    # Patch information (loaded from patches.dat)
    patch_info: pd.DataFrame | None = None
    alf_info: dict[str, Any] | None = None

    # CpHMD parameters
    site_pH0: dict[str, float] = field(default_factory=dict)
    site_pKa_shifts: dict[str, dict] = field(default_factory=dict)
    delta_pKa: float = 0.0

    # Phase transition tracking
    phase2_start_run: int | None = None  # Run index when Phase 2 started

    # Stop criteria state
    converged: bool = False
    stop_reason: str = ""
    needs_confirmation: bool = False  # Flag to trigger confirmation repeat

    # Forced initial lambdas for unsampled states (set by _alf_analysis)
    forced_initial_lambdas: dict | None = None

    # RMSD convergence state (used when convergence_mode="rmsd")
    rmsd_state: Any = None  # RMSDState instance, lazy-initialized


class ALFSimulation:
    """Main ALF simulation orchestrator.

    This class manages the complete ALF/CpHMD simulation workflow:
    1. System initialization and MPI setup
    2. Phase-based dynamics execution
    3. ALF analysis and bias updates
    4. Henderson-Hasselbalch curve generation

    Example:
        >>> config = ALFConfig(input_folder="my_system", pH=7.0, nreps=8)
        >>> sim = ALFSimulation(config)
        >>> sim.run()
    """

    ALPHABET = "abcdefghijklmnopqrstuvwxyz"
    LOG_UNIT = 99  # File unit for CHARMM log redirection

    def __init__(self, config: ALFConfig):
        """Initialize ALF simulation.

        Args:
            config: ALFConfig with simulation parameters
        """
        self.config = config
        self.state = SimulationState()
        self.state.phase = config.phase  # Initialize phase from config
        self._comm = None
        self._initialized = False
        self._log_file = None  # Current CHARMM log file object

    def _ntersite(self) -> list[int]:
        """Compute [ms, msprof] from coupling config."""
        ms = self.config.coupling
        if self.config.coupling_profile is None:
            msprof = int(ms > 0)
        else:
            msprof = int(self.config.coupling_profile)
        return [ms, msprof]

    def _load_rmsd_state(self) -> None:
        """Load RMSD convergence state from disk for resume support."""
        from .rmsd_convergence import RMSDState
        state_file = self.config.input_folder / "rmsd_state.json"
        if state_file.exists():
            try:
                self.state.rmsd_state = RMSDState.load(state_file)
                print(f"Loaded RMSD state: {len(self.state.rmsd_state.rmsd_history)} entries")
            except Exception as e:
                print(f"Warning: could not load rmsd_state.json: {e}")
                self.state.rmsd_state = RMSDState()
        else:
            self.state.rmsd_state = RMSDState()

    def _init_mpi(self):
        """Initialize MPI and determine rank/size."""
        from mpi4py import MPI
        self._comm = MPI.COMM_WORLD
        self.state.rank = self._comm.Get_rank()
        self.state.size = self._comm.Get_size()

        # Set nreps to MPI size if not specified
        if self.config.nreps is None:
            self.config.nreps = self.state.size

        # Assign GPU based on local rank
        self.state.gpuid = self._get_gpu_id()

    def _get_gpu_id(self) -> int:
        """Determine GPU ID based on local MPI rank and CUDA_VISIBLE_DEVICES."""
        # Try various MPI implementations for local rank
        local_rank = None

        env_vars = [
            "OMPI_COMM_WORLD_LOCAL_RANK",  # OpenMPI
            "SLURM_LOCALID",               # SLURM
            "MPI_LOCALRANKID",             # Intel MPI
            "PMI_LOCAL_RANK",              # MPICH
        ]

        for env_var in env_vars:
            if env_var in os.environ:
                local_rank = int(os.environ[env_var])
                break

        if local_rank is None:
            # Fallback: use global rank modulo assumed GPUs per node
            local_rank = self.state.rank % 4

        # Check CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            available_gpus = [int(g.strip()) for g in cuda_visible.split(",") if g.strip()]
            if local_rank < len(available_gpus):
                return available_gpus[local_rank]

        return local_rank

    def _redirect_output(self, run_idx: int, k: int, replica_idx: int):
        """Redirect CHARMM output to a separate log file for this replica/repeat.

        Creates a log file at run{run_idx}/log.{k}.{replica_idx}.out and redirects
        CHARMM's output unit (OUTU) to write to it. This separates CHARMM output
        by replica and repeat for easier debugging.

        Args:
            run_idx: Current run number
            k: Repeat index within run
            replica_idx: MPI replica index
        """
        import pycharmm
        import pycharmm.lingo as lingo

        run_dir = self.config.input_folder / f"run{run_idx}"
        log_path = run_dir / f"log.{k}.{replica_idx}.out"

        # Create CharmmFile for log output
        self._log_file = pycharmm.CharmmFile(
            file_name=str(log_path),
            file_unit=self.LOG_UNIT,
            read_only=False,
            formatted=True,
        )

        # Redirect CHARMM output to this file unit
        lingo.charmm_script(f"OUTUnit {self.LOG_UNIT}")

    def _return_output(self):
        """Return CHARMM output to standard output (unit 6).

        Closes the current log file and restores CHARMM's output to stdout.
        """
        import pycharmm.lingo as lingo

        if self._log_file is not None:
            # Restore output to stdout (unit 6)
            lingo.charmm_script("OUTUnit 6")
            self._log_file.close()
            self._log_file = None

    def _init_charmm(self):
        """Initialize CHARMM and read topology files."""
        from .charmm_utils import read_topology_files

        # Verify CHARMM environment
        charmm_lib = os.environ.get("CHARMM_LIB_DIR")
        if not charmm_lib or not os.path.isdir(charmm_lib):
            raise RuntimeError("CHARMM_LIB_DIR not set or invalid")

        # Load topology files using utility function
        read_topology_files(
            self.config.toppar_dir,
            self.config.topology_files,
            verbose=False,
        )

        # Load extra files for custom ligands (absolute paths)
        if self.config.extra_files:
            self._load_extra_files()

    def _load_extra_files(self):
        """Load extra topology/parameter files for custom ligands.

        These are absolute paths to RTF, PRM, or STR files that supplement
        the standard CHARMM topology. Used for custom ligands with CGenFF
        parameters.
        """
        from pycharmm import lingo, read, settings

        # Suppress warnings during loading
        lingo.charmm_script("prnlev -1")
        lingo.charmm_script("bomblevel -2")
        settings.set_warn_level(-1)

        for file_path in self.config.extra_files:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Extra file not found: {file_path}")

            suffix = file_path.suffix.lower()
            if suffix == ".rtf":
                read.rtf(str(file_path), append=True)
            elif suffix == ".prm":
                read.prm(str(file_path), flex=True, append=True)
            elif suffix == ".str":
                lingo.charmm_script(f"stream {file_path}")
            else:
                raise ValueError(f"Unknown file type: {file_path}")

        # Restore settings
        settings.set_warn_level(5)
        lingo.charmm_script("bomblevel 0")
        lingo.charmm_script("prnlev 5")

    def _load_patch_info(self):
        """Load patch information from patches.dat."""
        patches_path = self.config.input_folder / "prep" / "patches.dat"
        self.state.patch_info = pd.read_csv(patches_path)

        # Extract site and subsite indices from SELECT column (e.g., "s1s1" -> site=1, sub=1)
        self.state.patch_info[["site", "sub"]] = (
            self.state.patch_info["SELECT"].str.extract(r"s(\d+)s(\d+)")
        )

    def _init_alf(self):
        """Initialize ALF and create initial variable files."""
        if self.state.patch_info is None:
            raise ValueError("patch_info must be loaded before ALF initialization")

        prep_dir = self.config.input_folder / "prep"

        # Build alf_info dictionary
        alf_info = {
            "name": self.config.input_folder.name,
            "nsubs": np.array([], dtype=int),
            "nblocks": 0,
            "nreps": self.config.nreps,
            "ncentral": self.config.nreps // 2,
            "nnodes": 1,
            "temp": self.config.temperature,
            "engine": "charmm",
            "ntersite": self._ntersite(),  # Intersite biases [ms, msprof]
            "fnex": self.config.fnex,
        }

        # Count blocks and subsites per site
        for site in self.state.patch_info["site"].unique():
            site_data = self.state.patch_info[self.state.patch_info["site"] == site]
            n_subsites = len(site_data["sub"].unique())
            alf_info["nblocks"] += n_subsites
            alf_info["nsubs"] = np.append(alf_info["nsubs"], n_subsites)

        self.state.alf_info = alf_info

        # Write ALF configuration files
        for key, value in alf_info.items():
            with open(prep_dir / key, "w") as f:
                if key == "nsubs":
                    f.write(" ".join(map(str, value)))
                else:
                    f.write(str(value))

        # Ensure G_imp entropy data is available (computed/cached locally)
        if self.state.rank == 0:
            self._ensure_g_imp()

        # Initialize ALF variables (creates analysis0/, variables1.inp)
        if self.state.rank == 0:
            init_vars(self.config.input_folder, alf_info)

    def _init_alf_legacy(self):
        """Initialize ALF from a legacy prep/alf_info.py file."""
        prep_dir = self.config.input_folder / "prep"
        alf_info_path = prep_dir / "alf_info.py"

        # Execute alf_info.py to extract the dictionary
        alf_info_ns = {"np": np}
        exec(alf_info_path.read_text(), alf_info_ns)

        if "alf_info" not in alf_info_ns:
            raise ValueError(f"alf_info.py must define 'alf_info' dict: {alf_info_path}")

        alf_info = alf_info_ns["alf_info"]

        # Validate required key
        if "box" not in alf_info:
            raise ValueError(
                f"Legacy prep format requires 'box' in alf_info.py "
                f"(cubic box edge length). Add: alf_info['box'] = <value>"
            )

        # Ensure nsubs is numpy array
        alf_info["nsubs"] = np.array(alf_info["nsubs"], dtype=int)
        alf_info["nblocks"] = int(np.sum(alf_info["nsubs"]))

        # Fill defaults for keys CpHMD expects
        alf_info.setdefault("engine", "charmm")
        alf_info.setdefault("nreps", self.config.nreps or 1)
        alf_info.setdefault("ncentral", alf_info["nreps"] // 2)
        alf_info.setdefault("nnodes", 1)
        alf_info.setdefault("temp", self.config.temperature)
        alf_info.setdefault("ntersite", self._ntersite())
        alf_info.setdefault("fnex", self.config.fnex)

        # Override nreps from config if user specified it
        if self.config.nreps is not None:
            alf_info["nreps"] = self.config.nreps

        self.state.alf_info = alf_info

        # Ensure G_imp entropy data is available
        if self.state.rank == 0:
            self._ensure_g_imp()

        # Initialize ALF variables (creates analysis0/, variables1.inp)
        if self.state.rank == 0:
            init_vars(self.config.input_folder, alf_info)

    def _ensure_g_imp(self):
        """Ensure G_imp entropy data is available for WHAM.

        Priority:
        1. Use existing G_imp/ in input folder (precomputed / previous run)
        2. Copy from bundled cphmd/data/G_imp/ (precomputed profiles)
        3. Compute via Monte Carlo and cache (expensive, last resort)
        """
        dst = self.config.input_folder / "G_imp"
        if dst.exists():
            print(f"Using existing G_imp: {dst}")
            return

        # Try bundled data first (avoids expensive MC computation)
        from cphmd import PACKAGE_DIR
        bundled = PACKAGE_DIR / "data" / "G_imp"
        if bundled.exists() and any(bundled.glob("G1_*.dat")):
            shutil.copytree(bundled, dst)
            print(f"Copied bundled G_imp to {dst}")
            return

        # Fall back to computing from scratch
        from cphmd.core.entropy import ensure_g_imp_available

        nsubs = self.state.alf_info["nsubs"]
        g_imp_dir = ensure_g_imp_available(
            constraint_type="fnex",
            nsubs=nsubs,
        )

        try:
            dst.symlink_to(g_imp_dir)
            print(f"Linked G_imp: {dst} → {g_imp_dir}")
        except OSError:
            shutil.copytree(g_imp_dir, dst, dirs_exist_ok=True)
            print(f"Copied G_imp to {dst}")

    def initialize(self):
        """Perform full initialization sequence."""
        self._init_mpi()
        self._comm.Barrier()

        self._init_charmm()

        if self.config.prep_format == "legacy":
            pass  # Legacy mode: no patches.dat to load
        else:
            self._load_patch_info()

        self._comm.Barrier()

        if self.config.prep_format == "legacy":
            self._init_alf_legacy()
        else:
            self._init_alf()

        self._comm.Barrier()

        self._initialized = True

    def run(self):
        """Execute the ALF simulation.

        This is the main entry point that orchestrates the complete simulation:
        1. Initialize MPI, CHARMM, and ALF
        2. Run dynamics for each iteration (start → end)
        3. Perform ALF analysis between iterations
        4. Generate Henderson-Hasselbalch curves
        """
        if not self._initialized:
            self.initialize()

        # Find last completed run and resume
        start_run = self._find_last_run()
        start_run = self._comm.bcast(start_run, root=0)

        # Load RMSD convergence state if resuming with rmsd mode
        if self.config.convergence_mode == "rmsd":
            self._load_rmsd_state()

        print(f"Rank {self.state.rank}: Starting from run {start_run}, "
              f"phase {self.state.phase}, nreps {self.config.nreps}")

        # Main simulation loop
        for run_idx in range(start_run, self.config.end + 1):
            self._execute_run(run_idx)

            # Check for early convergence (auto_stop)
            if self.state.converged:
                print(f"\nSimulation converged at run {run_idx}. Stopping early.")
                print(f"  {self.state.stop_reason}")
                break

        # Run final analysis after simulation completes (convergence or end of runs)
        final_run = run_idx if self.state.converged else self.config.end
        self._run_final_analysis(final_run)

    def _run_final_analysis(self, final_run: int) -> None:
        """Run final analysis after simulation completes or converges."""
        print(f"\n{'='*60}")
        print(f"Running final analysis (up to run {final_run})")
        print(f"{'='*60}")

        plots_dir = Path(self.config.input_folder) / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Energy Profile RMSD analysis
        try:
            from cphmd.analysis import EnergyProfileConfig, analyze_energy_profiles
            energy_config = EnergyProfileConfig(
                input_folder=self.config.input_folder,
                output_dir=plots_dir,
            )
            result = analyze_energy_profiles(energy_config)
            if result.rmsd_plot:
                print(f"  Energy RMSD plot: {result.rmsd_plot}")
            if result.profile_plot:
                print(f"  Energy profile plot: {result.profile_plot}")
        except Exception as e:
            print(f"  Energy profile analysis failed: {e}")

        # Population convergence plots
        try:
            from cphmd.analysis import generate_population_plots
            nsubs = self.state.alf_info.get("nsubs") if self.state.alf_info else None
            generate_population_plots(
                input_folder=self.config.input_folder,
                max_run=final_run,
                output_dir=plots_dir,
                nsubs=nsubs,
            )
            print(f"  Population convergence plots saved to {plots_dir}")
        except Exception as e:
            print(f"  Population convergence analysis failed: {e}")

        print(f"{'='*60}\n")

    def _find_last_run(self) -> int:
        """Find the last completed run to resume from.

        Also detects phase from log files if auto_phase_switch is enabled.
        Returns the run number to start/resume from.
        """
        found_run = None
        detected_phase = None

        for i in range(self.config.end + 1, self.config.start - 1, -1):
            var_file = self.config.input_folder / f"variables{i}.inp"
            if var_file.exists() and "set" in var_file.read_text():
                if "nan" in var_file.read_text().lower():
                    # Clean up corrupted run
                    self._cleanup_run(i)
                    continue

                # Found valid checkpoint
                found_run = i

                # Try to detect phase from previous run's logs
                if self.config.auto_phase_switch and i > 1:
                    phase_from_logs = self._detect_phase_from_logs(i - 1)
                    if phase_from_logs is not None:
                        detected_phase = phase_from_logs
                        print(f"Detected phase {detected_phase} from run {i-1} logs")

                    # Also check phase.dat in analysis directory
                    phase_file = self.config.input_folder / f"analysis{i-1}" / "phase.dat"
                    if phase_file.exists():
                        try:
                            saved_phase = int(np.loadtxt(phase_file))
                            if 1 <= saved_phase <= 3:
                                detected_phase = saved_phase
                                print(f"Loaded phase {detected_phase} from {phase_file}")
                        except Exception:
                            pass

                # Clean current run to restart
                self._cleanup_run(i)
                break

        # Update phase if detected
        if detected_phase is not None:
            self.state.phase = detected_phase

            # Detect when Phase 2 started (scan backwards for last Phase 1)
            if detected_phase >= 2 and found_run is not None:
                self.state.phase2_start_run = found_run  # conservative default
                for j in range(found_run - 1, 0, -1):
                    pf = self.config.input_folder / f"analysis{j}" / "phase.dat"
                    if pf.exists():
                        try:
                            p = int(np.loadtxt(pf))
                            if p == 1:
                                self.state.phase2_start_run = j + 1
                                break
                        except Exception:
                            pass

        return found_run if found_run else self.config.start

    def _cleanup_run(self, run_idx: int):
        """Remove run and analysis directories for a given run."""
        run_dir = self.config.input_folder / f"run{run_idx}"
        analysis_dir = self.config.input_folder / f"analysis{run_idx}"

        if run_dir.exists():
            shutil.rmtree(run_dir)
        if analysis_dir.exists():
            shutil.rmtree(analysis_dir)

    def _execute_run(self, run_idx: int):
        """Execute a single ALF run iteration.

        This includes:
        1. Setting up crystal/periodic boundaries
        2. Building BLOCK/MSLD commands
        3. Running equilibration + production dynamics
        4. ALF analysis (rank 0 only)
        5. Confirmation repeat if stop criteria met
        """
        # Reset confirmation flag at start of each run
        self.state.needs_confirmation = False

        # Broadcast current phase from rank 0
        self.state.phase = self._comm.bcast(self.state.phase, root=0)

        # Determine number of repeats based on phase
        repeats = 1 if self.state.phase == 1 else 2

        for k in range(repeats):
            start_time = time.time()

            self._comm.Barrier()

            # Get replica assignments for this rank
            my_replicas = self._get_replica_assignments()

            for replica_idx in my_replicas:
                letter = f"_{self.ALPHABET[replica_idx]}" if self.config.nreps > 1 else ""

                # Setup run directory and redirect CHARMM output to log file
                self._setup_run_directory(run_idx)
                self._redirect_output(run_idx, k, replica_idx)

                # Setup system (method depends on prep format)
                if self.config.prep_format == "legacy":
                    self._setup_legacy(run_idx, letter, k, replica_idx)
                else:
                    self._setup_crystal(run_idx, letter, k, replica_idx)
                    self._build_block_commands(run_idx, letter, k, replica_idx)
                    self._apply_restraints(run_idx, k)

                # Run minimization on first runs if needed
                if run_idx <= 5:
                    self._run_minimization(run_idx, replica_idx)

                self._run_dynamics(run_idx, letter, k, replica_idx)

                # Return CHARMM output to stdout
                self._return_output()

            self._comm.Barrier()
            elapsed = time.time() - start_time
            print(f"Run {run_idx}.{k} dynamics completed in {elapsed:.1f}s")

        # ALF analysis (rank 0 only)
        if self.state.rank == 0:
            start_time = time.time()
            self._alf_analysis(run_idx, repeats)
            elapsed = time.time() - start_time
            print(f"Analysis completed in {elapsed:.1f}s")

        self._comm.Barrier()

        # Check if confirmation repeat is needed (broadcast from rank 0)
        needs_confirmation = self._comm.bcast(self.state.needs_confirmation, root=0)

        if needs_confirmation:
            print(f"\nRunning confirmation repeat for run {run_idx}...")

            # Run one more repeat (k = repeats, since we already did 0..repeats-1)
            k = repeats
            start_time = time.time()

            self._comm.Barrier()

            # Get replica assignments for this rank
            my_replicas = self._get_replica_assignments()

            for replica_idx in my_replicas:
                letter = f"_{self.ALPHABET[replica_idx]}" if self.config.nreps > 1 else ""

                # Redirect output to log file for confirmation repeat
                self._redirect_output(run_idx, k, replica_idx)
                self._run_dynamics(run_idx, letter, k, replica_idx)
                self._return_output()

            self._comm.Barrier()
            elapsed = time.time() - start_time
            print(f"Run {run_idx}.{k} (confirmation) completed in {elapsed:.1f}s")

            # Re-run analysis with additional data
            if self.state.rank == 0:
                print("Re-analyzing with confirmation data...")
                self._alf_analysis(run_idx, repeats + 1, confirmation=True)

            self._comm.Barrier()

    def _get_replica_assignments(self) -> list[int]:
        """Get replica indices assigned to this MPI rank."""
        nreps = self.config.nreps
        size = self.state.size
        rank = self.state.rank

        if nreps > size:
            # Asynchronous: distribute replicas round-robin
            return [r for r in range(nreps) if r % size == rank]
        else:
            # Synchronous: one replica per rank (if rank < nreps)
            return [rank] if rank < nreps else []

    def _setup_run_directory(self, run_idx: int):
        """Create run directory structure."""
        if self.state.rank == 0:
            run_dir = self.config.input_folder / f"run{run_idx}"
            run_dir.mkdir(exist_ok=True)
            (run_dir / "res").mkdir(exist_ok=True)
        self._comm.Barrier()

    def _setup_crystal(self, run_idx: int, letter: str, k: int, replica_idx: int):
        """Setup crystal/periodic boundary conditions.

        Reads box parameters, loads structure, configures crystal images,
        and sets up non-bonded interactions.
        """
        from .charmm_utils import (
            BoxParameters,
            FFTParameters,
            NonBondedConfig,
            read_structure,
            setup_crystal,
            setup_nonbonded,
            define_selections,
            clear_block,
            clear_crystal,
        )
        import pycharmm.read as read
        import pycharmm.lingo as lingo
        import pycharmm.settings as settings
        import random

        prep_dir = self.config.input_folder / "prep"
        # Use structure_loaded flag to determine if this is the first run in this session
        # This is more reliable than checking run_idx == config.start because
        # _find_last_run() may return a different starting run
        is_first_run = not self.state.structure_loaded and k == 0

        # Skip setup for subsequent repeats (k > 0) within any run
        # Structure is already loaded from k=0, just need to run dynamics again
        if k > 0:
            return


        if is_first_run:
            # Read PSF file (suppress non-integer charge warning for MSLD systems)
            settings.set_bomb_level(-1)
            if self.config.hmr:
                psf_file = prep_dir / "system_hmr.psf"
                if not psf_file.exists():
                    # Generate HMR PSF from regular PSF
                    psf_file = prep_dir / "system.psf"
                    read.psf_card(str(psf_file))
                    import pycharmm.psf as pycharmm_psf
                    pycharmm_psf.hmr(newpsf=str(prep_dir / "system_hmr.psf"))
                else:
                    read.psf_card(str(psf_file))
            else:
                psf_file = prep_dir / "system.psf"
                read.psf_card(str(psf_file))
            settings.set_bomb_level(0)

            # Define selections for titratable groups
            if self.state.patch_info is not None:
                define_selections(self.state.patch_info)

            # Mark structure as loaded for this session
            self.state.structure_loaded = True
        else:
            # Clear previous setup for subsequent runs
            clear_block()
            if self.config.restrains == "NOE":
                from .charmm_utils import clear_noe
                clear_noe()
            clear_crystal()

        # Determine restart run for coordinate loading
        if run_idx > 5:
            self.state.restart_run = random.randint(run_idx - 5, run_idx - 1)
        else:
            self.state.restart_run = 1

        # Load box parameters
        box_params = BoxParameters.from_file(prep_dir / "box.dat")
        self.state.crystal_type = box_params.crystal_type
        self.state.box_size = box_params.dimensions
        self.state.box_angles = box_params.angles

        # Load coordinates
        if (prep_dir / "system_min.crd").exists():
            crd_file = prep_dir / "system_min.crd"
        elif self.config.hmr:
            crd_file = prep_dir / "system_hmr.crd"
        else:
            crd_file = prep_dir / "system.crd"

        read.coor_card(str(crd_file))

        # Try loading restart coordinates if not first run
        if self.state.restart_run != 1:
            restart_candidates = [
                f"run{self.state.restart_run}/prod.{k}.{replica_idx}.crd",
                f"run{self.state.restart_run}/prod.crd{letter}",
                f"run{self.state.restart_run}/prod.crd",
            ]
            for crd_name in restart_candidates:
                crd_path = self.config.input_folder / crd_name
                if crd_path.exists():
                    read.coor_card(str(crd_path))
                    break

        # Load FFT parameters
        fft_params = FFTParameters.from_file(prep_dir / "fft.dat")

        # Setup non-bonded configuration
        nb_config = NonBondedConfig(
            cutnb=self.config.cutnb,
            cutim=self.config.cutnb,
            ctofnb=self.config.ctofnb,
            ctonnb=self.config.ctonnb,
            elec_type=self.config.elec_type,
            vdw_type=self.config.vdw_type,
            fftx=fft_params.fftx,
            ffty=fft_params.ffty,
            fftz=fft_params.fftz,
        )

        # Setup crystal
        setup_crystal(box_params, nb_config, use_image_centering=not self.config.cent_ncres)

        # Setup non-bonded interactions
        setup_nonbonded(nb_config)

    def _setup_legacy(self, run_idx: int, letter: str, k: int, replica_idx: int):
        """Set up CHARMM session by streaming a legacy setup script.

        For legacy (msld-py-prep) prep folders:
        1. Stream variables file (sets bias CHARMM variables)
        2. Pre-process and stream setup script (topology, patches, crystal, BLOCK)
        3. Stream setup script
        4. Load minimized / restart coordinates
        5. Setup nonbonded parameters
        6. Stream user-provided restraints (prep/restrains.str)
        """
        from .charmm_utils import (
            NonBondedConfig,
            setup_nonbonded,
            clear_block,
            clear_crystal,
        )
        import pycharmm.lingo as lingo
        import pycharmm.settings as settings
        import random

        prep_dir = self.config.input_folder / "prep"
        is_first_run = not self.state.structure_loaded and k == 0

        # Skip setup for subsequent repeats (k > 0) within any run
        if k > 0:
            return

        if not is_first_run:
            # Clear previous CHARMM state before rebuilding
            clear_block()
            if self.config.restrains == "NOE":
                from .charmm_utils import clear_noe
                clear_noe()
            clear_crystal()
            # Delete all atoms so the setup script can regenerate segments
            lingo.charmm_script("delete atom sele all end")

        # Determine restart run for coordinate loading
        if run_idx > 5:
            self.state.restart_run = random.randint(run_idx - 5, run_idx - 1)
        else:
            self.state.restart_run = 1

        # 1. Stream variables file (sets @lam*, @c*, @x*, @s* CHARMM variables)
        var_file = self.config.input_folder / f"variables{run_idx}.inp"
        if not var_file.exists():
            raise FileNotFoundError(f"Variables file not found: {var_file}")
        lingo.charmm_script(f"stream {var_file}")

        # 2. Pre-process setup script: replace placeholder values with actual ones
        #    msld-py-prep generates "set box = ????" as a placeholder
        setup_script = prep_dir / self.config.legacy_setup_script
        script_text = setup_script.read_text()

        box = self.state.alf_info["box"]
        import re
        script_text = re.sub(
            r"(?i)set\s+box\s*=\s*\S+",
            f"set box = {box}",
            script_text,
        )
        script_text = re.sub(
            r"(?i)set\s+builddir\s*=\s*\S+",
            f"set builddir = {prep_dir}",
            script_text,
        )

        # Write processed script to a temp file and stream it
        processed_script = self.config.input_folder / f".setup_{run_idx}.inp"
        processed_script.write_text(script_text)

        # 3. Stream processed setup script (topology, patches, selections, crystal, BLOCK)
        settings.set_bomb_level(-2)
        lingo.charmm_script(f"stream {processed_script}")
        settings.set_bomb_level(0)

        # Clean up temp file
        processed_script.unlink(missing_ok=True)

        self.state.structure_loaded = True

        # 4. Load minimized / restart coordinates (if available)
        #    The setup script reads PDB fragments (un-minimized).
        #    On run 1, _run_minimization() will create system_min.crd.
        #    On run 2+, we must read it back to start from minimized state.
        import pycharmm.read as pycharmm_read

        min_crd = prep_dir / "system_min.crd"
        if min_crd.exists():
            pycharmm_read.coor_card(str(min_crd))

        # Try restart coordinates from a previous run
        if self.state.restart_run and self.state.restart_run != 1:
            restart_candidates = [
                f"run{self.state.restart_run}/prod.{k}.{replica_idx}.crd",
                f"run{self.state.restart_run}/prod.crd{letter}",
                f"run{self.state.restart_run}/prod.crd",
            ]
            for crd_name in restart_candidates:
                crd_path = self.config.input_folder / crd_name
                if crd_path.exists():
                    pycharmm_read.coor_card(str(crd_path))
                    break

        # 5. Setup nonbonded parameters (setup script does not handle this)

        nb_config = NonBondedConfig(
            cutnb=self.config.cutnb,
            cutim=self.config.cutnb,
            ctofnb=self.config.ctofnb,
            ctonnb=self.config.ctonnb,
            elec_type=self.config.elec_type,
            vdw_type=self.config.vdw_type,
        )
        setup_nonbonded(nb_config)

        # 6. Apply restraints (SCAT or NOE) from user-provided file
        #    Legacy (msld-py-prep) systems use globally unique atom names per
        #    substituent, so auto-generation from shared atom names cannot work.
        #    The user must provide prep/restrains.str with the restraint commands.
        self._apply_legacy_restraints()

    def _apply_legacy_restraints(self):
        """Stream user-provided restraints for legacy mode.

        msld-py-prep gives each atom a globally unique name (C164, C195, ...),
        so the standard auto-generation (which relies on shared atom names across
        subs) cannot produce valid restraints.  Instead, we stream the
        user-provided prep/restrains.str file directly.
        """
        import pycharmm.lingo as lingo
        import logging

        prep_dir = self.config.input_folder / "prep"
        restraint_file = prep_dir / "restrains.str"

        if restraint_file.exists():
            lingo.charmm_script(f"stream {restraint_file}")
            return

        if self.config.restrains:
            logging.getLogger(__name__).warning(
                "Restraints requested (restrains=%r) but prep/restrains.str not found. "
                "Legacy (msld-py-prep) systems require a hand-written restraints file. "
                "See the example in examples/88_marcella/prep/restrains.str.",
                self.config.restrains,
            )

    def _build_block_commands(self, run_idx: int, letter: str, k: int, replica_idx: int):
        """Build and execute BLOCK/MSLD commands for lambda dynamics."""
        from .block_builder import BlockConfig, build_block_command, read_variable_file
        from .charmm_utils import execute_block_command
        from .cphmd_params import (
            compute_all_site_parameters,
            compute_bias_shifts,
            write_bias_files,
            get_delta_pKa_for_phase,
        )

        # Skip BLOCK setup for subsequent repeats (k > 0)
        # BLOCK commands persist from k=0
        if k > 0:
            return

        if self.state.patch_info is None:
            raise ValueError("patch_info not loaded")

        # Read ALF variables file
        var_file = self.config.input_folder / f"variables{run_idx}.inp"
        variables = read_variable_file(var_file)

        # Compute CpHMD parameters if pH is specified
        effective_pH = self.config.pH
        delta_pKa = get_delta_pKa_for_phase(self.state.phase)

        if self.config.pH is not None:
            cphmd_params = compute_all_site_parameters(
                self.state.patch_info,
                self.config.temperature,
                self.config.pH,
            )
            effective_pH = cphmd_params.effective_pH

            # Compute and write bias shifts (or zeros if pKa bias disabled)
            if self.config.no_pka_bias:
                # Write zero bias shifts
                nblocks = self.state.alf_info["nblocks"]
                b_shift = np.zeros(nblocks)
                b_fix_shift = np.zeros(nblocks)
            else:
                b_shift, b_fix_shift = compute_bias_shifts(
                    cphmd_params,
                    self.state.patch_info,
                    delta_pKa,
                    replica_idx,
                )
            write_bias_files(self.config.input_folder, b_shift, b_fix_shift)

        # Build BLOCK configuration
        block_config = BlockConfig(
            temperature=self.config.temperature,
            pH=self.config.pH,
            effective_pH=effective_pH,
            delta_pKa=delta_pKa,
            use_cphmd=(self.config.pH is not None and delta_pKa != 0 and not self.config.no_pka_bias),
            initial_lambdas=self.state.forced_initial_lambdas,
            lambda_mass=self.config.lambda_mass,
            lambda_fbeta=self.config.lambda_fbeta,
        )

        # Generate and execute BLOCK command
        block_cmd = build_block_command(self.state.patch_info, variables, block_config)

        # Write block file for debugging
        block_file = self.config.input_folder / f"run{run_idx}" / f"block.{k}.{replica_idx}.str"
        block_file.write_text(block_cmd)

        # Execute BLOCK command
        execute_block_command(block_cmd)

    def _apply_restraints(self, run_idx: int, k: int = 0):
        """Apply SCAT or NOE restraints to titratable atoms."""
        from .restraints import generate_scat_restraints, generate_noe_restraints
        from .charmm_utils import execute_block_command
        import pycharmm.lingo as lingo

        # Skip restraint setup for subsequent repeats (k > 0)
        # Restraints persist from k=0
        if k > 0:
            return

        if self.state.patch_info is None:
            raise ValueError("patch_info not loaded")

        # Generate appropriate restraint command
        include_hydrogen = self.config.restrain_hydrogens

        if self.config.restrains == "NOE":
            restraint_cmd = generate_noe_restraints(
                self.state.patch_info, include_hydrogen
            )
        else:
            restraint_cmd = generate_scat_restraints(
                self.state.patch_info, include_hydrogen
            )

        # Write restraints file
        restraint_file = self.config.input_folder / "prep" / "restrains.str"
        restraint_file.write_text(restraint_cmd)

        # Execute restraint command
        lingo.charmm_script(restraint_cmd)

    def _run_minimization(self, run_idx: int, replica_idx: int):
        """Run energy minimization before dynamics.

        Only runs if no minimized coordinates exist.  Uses steepest-descent
        minimisation (BLADE supports SD but not ABNR).
        """
        import pycharmm.minimize as minimize
        import pycharmm.shake as shake
        import pycharmm.write as write
        import pycharmm.energy as energy

        min_crd = self.config.input_folder / "prep" / "system_min.crd"
        if min_crd.exists():
            return  # Already minimized

        print(f"Running minimization for run {run_idx}...")

        # Enable SHAKE for minimization
        shake.on(fast=True, bonh=True, param=True, tol=1e-7)

        # Steepest descent (250 steps, CPU)
        minimize.run_sd(nstep=250, nprint=50, step=0.005, tolenr=1e-3, tolgrd=1e-3)

        # Save minimized coordinates
        write.coor_card(str(min_crd))
        print(f"Minimized coordinates saved to {min_crd}")

        energy.show()

    def _run_dynamics(self, run_idx: int, letter: str, k: int, replica_idx: int):
        """Execute molecular dynamics with BLADE GPU acceleration.

        This method runs equilibration and production dynamics using pyCHARMM
        with BLADE GPU acceleration. Lambda dynamics files are written for
        subsequent ALF analysis.

        Args:
            run_idx: Current run number
            letter: Run letter suffix (for replica identification)
            k: Repeat index within run
            replica_idx: MPI replica index
        """
        import pycharmm
        import pycharmm.psf as psf
        import pycharmm.lingo as lingo
        import pycharmm.settings as settings

        # File units
        dcd_unit = 51
        rst_unit = 52
        lmd_unit = 53
        rpr_unit = 54

        # Dynamics parameters vary by phase
        if self.state.phase == 1:
            nsteps_eq = 0        # No equilibration in phase 1
            nsteps_prod = 40000  # 80 ps
            nsavc = 0            # No DCD saving by default
            nsavl = 1            # Save lambda every 2 fs
        elif self.state.phase == 2:
            nsteps_eq = 10000    # 20 ps equilibration
            nsteps_prod = 450000 # 900 ps
            nsavc = 0            # No DCD saving by default
            nsavl = 1            # Save lambda every 2 fs
        else:  # Phase 3
            nsteps_eq = 10000    # 20 ps equilibration
            nsteps_prod = 500000 # 1 ns
            nsavc = 0            # No DCD saving by default
            nsavl = 1            # Save lambda every 2 fs

        # HMR allows 4fs timestep
        timestep = 0.002
        if self.config.hmr:
            nsteps_eq //= 2
            nsteps_prod //= 2
            nsavc //= 2
            nsavl = max(nsavl, 1)
            timestep = 0.004

        # Legacy mode: match classic ALF lambda-save frequency
        if self.config.prep_format == "legacy":
            nsavl = 10

        # Initialize dynamics: SHAKE, FASTER, BLADE
        import pycharmm.shake as shake
        import pycharmm.dynamics as dyn

        # SHAKE for hydrogen constraints
        shake.on(fast=True, bonh=True, param=True, tol=1e-7)

        # Enable FASTER algorithm
        lingo.charmm_script("faster on")

        # Enable BLADE GPU
        gpuid = replica_idx % 8  # Assume max 8 GPUs
        lingo.charmm_script(f"blade on gpuid {gpuid}")

        # Set friction coefficients for Langevin dynamics
        dyn.set_fbetas(np.full(psf.get_natom(), self.config.gscale, dtype=float))

        # Initialize BLADE energy calculation (required before first dynamics step)
        lingo.charmm_script("energy blade")

        # Base dynamics parameters
        dyn_param = {
            "start": True,
            "restart": False,
            "blade": True,
            "prmc": True,
            "iprs": 100,
            "prdv": 100,
            "cpt": True,  # Constant pressure
            "timestep": timestep,
            "firstt": self.config.temperature,
            "finalt": self.config.temperature,
            "tstruc": self.config.temperature,
            "tbath": self.config.temperature,
            "ichecw": 0,  # Don't scale velocities
            "ihtfrq": 0,  # No heating
            "ieqfrq": 0,  # No velocity scaling
            "iasors": 1,  # Assign velocities during heating
            "iasvel": 1,  # Gaussian velocity distribution
            "iscvel": 0,
            "inbfrq": -1,  # Auto-update neighbor lists
            "ilbfrq": 0,
            "imgfrq": -1,  # Auto-update images
            "ntrfrq": 0,
            "echeck": -1,  # Disable energy check
            "iunldm": lmd_unit,
            "iunwri": rst_unit,
            "iuncrd": dcd_unit if nsavc > 0 else -1,  # -1 = don't write DCD
            "nsavc": nsavc,
            "nsavl": nsavl,
            "nprint": 10000,
            "iprfrq": 10000,
            "isvfrq": 0,  # Will be set to nstep before each run (save RST only at end)
        }

        # NPT ensemble settings
        dyn_param.update({
            "pconstant": True,
            "pmass": psf.get_natom() * 0.12,
            "pref": 1.0,
            "pgamma": 20.0,
            "hoover": True,
            "reft": self.config.temperature,
            "tmass": 1000,
        })

        name = self.config.input_folder.name
        run_dir = self.config.input_folder / f"run{run_idx}"
        res_dir = run_dir / "res"
        dcd_dir = run_dir / "dcd"
        if nsavc > 0:
            dcd_dir.mkdir(exist_ok=True)

        # === Equilibration Run ===
        if nsteps_eq > 0:
            rst_fn = str(res_dir / f"{name}_eq.{k}.{replica_idx}.rst")
            lmd_fn = str(res_dir / f"{name}_eq.{k}.{replica_idx}.lmd")

            # Only open DCD file if nsavc > 0
            dcd = None
            if nsavc > 0:
                dcd_fn = str(dcd_dir / f"{name}_eq.{k}.{replica_idx}.dcd")
                dcd = pycharmm.CharmmFile(file_name=dcd_fn, file_unit=dcd_unit,
                                          read_only=False, formatted=False)
            rst = pycharmm.CharmmFile(file_name=rst_fn, file_unit=rst_unit,
                                      read_only=False, formatted=True)
            lmd = pycharmm.CharmmFile(file_name=lmd_fn, file_unit=lmd_unit,
                                      read_only=False, formatted=False)

            dyn_param.update({"nstep": nsteps_eq, "isvfrq": nsteps_eq})
            pycharmm.DynamicsScript(**dyn_param).run()

            if dcd is not None:
                dcd.close()
            rst.close()
            lmd.close()

        lingo.charmm_script("energy blade")

        # === Production Run ===
        if nsteps_prod > 0:
            sim_type = "flat" if self.state.phase in (1, 2) else "prod"

            rst_fn = str(res_dir / f"{name}_{sim_type}.{k}.{replica_idx}.rst")
            lmd_fn = str(res_dir / f"{name}_{sim_type}.{k}.{replica_idx}.lmd")

            # Update for restart from equilibration
            dyn_param.update({"start": False, "restart": True, "iunrea": rpr_unit})

            # Find restart file
            rpr_fn = None
            if sim_type == "flat":
                candidates = [
                    res_dir / f"{name}_eq.{k}.{replica_idx}.rst",
                    res_dir / f"{name}_eq.rst",
                ]
            else:
                restart_run = run_idx - 1
                candidates = [
                    self.config.input_folder / f"run{restart_run}" / "res" / f"{name}_prod.{k}.{replica_idx}.rst",
                    self.config.input_folder / f"run{restart_run}" / "res" / f"{name}_flat.{k}.{replica_idx}.rst",
                ]

            for candidate in candidates:
                if candidate.exists():
                    rpr_fn = str(candidate)
                    break

            if rpr_fn is None and sim_type == "flat":
                # No restart file - start fresh
                dyn_param.update({"start": True, "restart": False})
                dyn_param.pop("iunrea", None)
                rpr = None
            elif rpr_fn is None:
                raise RuntimeError(f"No restart file found for production run {run_idx}")
            else:
                rpr = pycharmm.CharmmFile(file_name=rpr_fn, file_unit=rpr_unit,
                                          read_only=True, formatted=True)

            # Only open DCD file if nsavc > 0
            dcd = None
            if nsavc > 0:
                dcd_fn = str(dcd_dir / f"{name}_{sim_type}.{k}.{replica_idx}.dcd")
                dcd = pycharmm.CharmmFile(file_name=dcd_fn, file_unit=dcd_unit,
                                          read_only=False, formatted=False)
            rst = pycharmm.CharmmFile(file_name=rst_fn, file_unit=rst_unit,
                                      read_only=False, formatted=True)
            lmd = pycharmm.CharmmFile(file_name=lmd_fn, file_unit=lmd_unit,
                                      read_only=False, formatted=False)

            dyn_param.update({"nstep": nsteps_prod, "isvfrq": nsteps_prod})
            pycharmm.DynamicsScript(**dyn_param).run()

            if dcd is not None:
                dcd.close()
            rst.close()
            lmd.close()
            if rpr is not None:
                rpr.close()

        lingo.charmm_script("blade off")

    def _is_wham_output_invalid(self, analysis_dir: Path) -> bool:
        """Check for invalid WHAM output (all-zero, NaN, or Inf).

        Args:
            analysis_dir: Path to analysis directory containing b.dat, c.dat, etc.

        Returns:
            True if output is invalid and WHAM should be retried
        """
        files_to_check = ['b.dat', 'c.dat', 'b_sum.dat', 'c_sum.dat']

        for fname in files_to_check:
            fpath = analysis_dir / fname
            if not fpath.exists():
                continue
            try:
                data = np.loadtxt(fpath)
                if data.size == 0:
                    print(f"WHAM validation: {fname} is empty")
                    return True
                if np.all(data == 0):
                    print(f"WHAM validation: {fname} contains all zeros")
                    return True
                if np.any(np.isnan(data)):
                    print(f"WHAM validation: {fname} contains NaN")
                    return True
                if np.any(np.isinf(data)):
                    print(f"WHAM validation: {fname} contains Inf")
                    return True
            except Exception as e:
                print(f"WHAM validation: error reading {fname}: {e}")
                return True

        return False

    def _cleanup_invalid_wham(self, analysis_dir: Path) -> None:
        """Remove invalid WHAM output files for retry.

        Args:
            analysis_dir: Path to analysis directory
        """
        files_to_remove = [
            'b.dat', 'c.dat', 'x.dat', 's.dat',
            'b_sum.dat', 'c_sum.dat', 'x_sum.dat', 's_sum.dat',
        ]
        for fname in files_to_remove:
            fpath = analysis_dir / fname
            if fpath.exists():
                fpath.unlink()

        # Also remove multisite directory if present
        multisite_dir = analysis_dir / 'multisite'
        if multisite_dir.exists():
            shutil.rmtree(multisite_dir)

    # --- Adaptive cutoff helpers ---

    @staticmethod
    def _read_scaling_dat(scaling_file: Path) -> dict | None:
        """Read scaling diagnostics from a previous analysis step."""
        if not scaling_file.exists():
            return None
        try:
            with open(scaling_file) as f:
                header_fields = []
                for line in f:
                    if line.startswith("#"):
                        header_fields = line.strip().split()[1:]  # skip '#'
                        continue
                    parts = line.split()
                    if len(parts) >= 6:
                        result = {
                            "scaling": float(parts[0]),
                            "bottleneck": parts[1],
                            "cutb": float(parts[2]),
                            "cutc": float(parts[3]),
                            "cutx": float(parts[4]),
                            "cuts": float(parts[5]),
                        }
                        # Parse remaining fields based on header
                        for idx in range(6, len(parts)):
                            if idx < len(header_fields):
                                field_name = header_fields[idx]
                                try:
                                    result[field_name] = float(parts[idx])
                                except ValueError:
                                    result[field_name] = parts[idx]
                            elif idx == 6:
                                # Legacy format: 7th field is max_pop_diff
                                result["max_pop_diff"] = float(parts[idx])
                        return result
        except (ValueError, IndexError):
            return None
        return None

    def _adapt_cutoffs(self, run_idx: int) -> dict:
        """Compute adaptive cutoffs for Phase 1 based on previous scaling.

        Reads scaling.dat from the previous analysis directory. If scaling < 0.6,
        loosens the bottleneck cutoff. If scaling == 1.0 for 2 consecutive runs,
        tightens all cutoffs. Bounds: [1.0, 50.0].

        Returns:
            dict of cutoff/calc parameters to pass to get_free_energy5.
        """
        CUT_FLOOR = 1.0
        CUT_CEIL = 50.0

        # First 20 runs: aggressive b/c cutoffs to establish biases fast.
        # x and s excluded (0 = disable) — focus on linear/coupling terms first.
        # Per-parameter clipping ensures no single parameter overshoots.
        if run_idx < 20:
            cutb, cutc = 10.0, 20.0
            cutx, cuts = 0.0, 0.0
        else:
            cutb, cutc = 5.0, 8.0
            cutx, cuts = 1.0, 1.0

        # Try to read scaling from most recent analysis
        prev_scaling = None
        for prev_run in (run_idx, run_idx - 1):
            if prev_run < 0:
                break
            prev_file = self.config.input_folder / f"analysis{prev_run}" / "scaling.dat"
            prev_scaling = self._read_scaling_dat(prev_file)
            if prev_scaling is not None:
                break

        if prev_scaling is not None:
            # Start from previous cutoffs
            cutb = prev_scaling["cutb"]
            cutc = prev_scaling["cutc"]
            cutx = prev_scaling["cutx"]
            cuts = prev_scaling["cuts"]

            scaling = prev_scaling["scaling"]
            bottleneck = prev_scaling["bottleneck"]

            if scaling < 0.6:
                # Loosen the bottleneck parameter
                factor = min(1.5 / scaling, 3.0)
                cut_map = {"cutb": cutb, "cutc": cutc, "cutx": cutx, "cuts": cuts}
                if bottleneck in cut_map:
                    cut_map[bottleneck] = min(cut_map[bottleneck] * factor, CUT_CEIL)
                    cutb, cutc = cut_map["cutb"], cut_map["cutc"]
                    cutx, cuts = cut_map["cutx"], cut_map["cuts"]
                print(f"  Adaptive cutoffs: scaling={scaling:.2f} "
                      f"bottleneck={bottleneck} → loosened to "
                      f"cutb={cutb:.1f} cutc={cutc:.1f}")

            elif scaling >= 1.0:
                # Check for 2 consecutive scaling >= 1.0 → tighten
                consecutive_ones = 1
                for prev_run in range(run_idx - 1, max(run_idx - 2, 0), -1):
                    pf = self.config.input_folder / f"analysis{prev_run}" / "scaling.dat"
                    pd = self._read_scaling_dat(pf)
                    if pd and pd["scaling"] >= 1.0:
                        consecutive_ones += 1
                    else:
                        break
                if consecutive_ones >= 2:
                    cutb = max(cutb * 0.5, CUT_FLOOR)
                    cutc = max(cutc * 0.5, CUT_FLOOR)
                    cutx = max(cutx * 0.5, CUT_FLOOR)
                    cuts = max(cuts * 0.5, CUT_FLOOR)
                    print(f"  Adaptive cutoffs: {consecutive_ones}x scaling>=1.0 → tightened to "
                          f"cutb={cutb:.1f} cutc={cutc:.1f}")
            else:
                print(f"  Adaptive cutoffs: scaling={scaling:.2f} (in range, no change)")

            # If previous analysis had some transitions but poor connectivity,
            # dampen coupling cutoffs. Skip when connectivity=0 (normal for
            # Phase 1 short runs — no transitions expected at all).
            if ("connectivity" in prev_scaling
                    and 0 < prev_scaling["connectivity"] < 0.3):
                cutc = max(cutc * 0.5, CUT_FLOOR)
                cutx = max(cutx * 0.5, CUT_FLOOR)
                cuts = max(cuts * 0.5, CUT_FLOOR)
                print(f"  Low connectivity ({prev_scaling['connectivity']:.2f}) "
                      f"-> dampened coupling cutoffs")

        # Build cut_params dict with exclusion flags
        # A cutoff of 0 means "exclude this parameter type entirely"
        cut_params = {"cutb": cutb, "cutc": cutc, "cutx": cutx, "cuts": cuts}
        if self.config.no_b_bias or cutb == 0:
            cut_params["calc_phi"] = False
        if self.config.no_c_bias or cutc == 0:
            cut_params["calc_psi"] = False
        if self.config.no_x_bias or cutx == 0:
            cut_params["calc_chi"] = False
        if self.config.no_s_bias or cuts == 0:
            cut_params["calc_omega"] = False

        return cut_params

    def _rollback_biases(self, current_analysis: Path, prev_analysis: Path):
        """Revert bias files to previous analysis step."""
        for fname in ("b.dat", "c.dat", "x.dat", "s.dat"):
            src = prev_analysis / fname
            dst = current_analysis / fname
            if src.exists():
                shutil.copy2(src, dst)
        print(f"  Rolled back biases from {prev_analysis.name} to {current_analysis.name}")

    @staticmethod
    def _get_worst_site_diff(pop_strict, nsubs_eff):
        """Compute worst-site population imbalance metric."""
        worst = 0.0
        for start, end in _per_site_ranges(nsubs_eff):
            s = pop_strict[start:end]
            worst = max(worst, max(s) - min(s))
        return worst

    def _run_wham_with_retry(
        self,
        run_idx: int,
        nf: int,
        ms: int,
        msprof: int,
        cut_params: dict,
        max_attempts: int = 3,
    ) -> tuple[bool, str]:
        """Run analysis (WHAM or LMALF) with retry on failure or invalid output.

        Verbose WHAM output is redirected to analysis.log in the current directory.
        Only summary messages are returned to be printed to main output.

        Args:
            run_idx: Current run number
            nf: Number of files (N * nreps)
            ms: Multisite parameter
            msprof: Multisite profile parameter
            cut_params: Cutoff parameters (cutb, cutc, etc.)
            max_attempts: Maximum number of retry attempts

        Returns:
            Tuple of (success: bool, summary_message: str)
        """
        import contextlib

        analysis_dir = Path.cwd()  # We're already chdir'd to analysis dir
        method = self.config.analysis_method
        log_file = analysis_dir / "analysis.log"

        # Append WHAM verbose output to existing log file
        with open(log_file, "a") as log_f:
            for attempt in range(max_attempts):
                try:
                    log_f.write(f"{method.upper()} attempt {attempt + 1}/{max_attempts}...\n")
                    log_f.flush()

                    # Redirect stdout to log file during WHAM execution
                    with contextlib.redirect_stdout(log_f):
                        if method == "hybrid":
                            self._run_hybrid_analysis(nf, ms, msprof, cut_params)
                        elif method == "lmalf":
                            self._run_lmalf_analysis(nf, ms, msprof, cut_params)
                        else:
                            self._run_wham_analysis(nf, ms, msprof, cut_params)

                    # Validate output
                    if self._is_wham_output_invalid(analysis_dir):
                        msg = f"{method.upper()} output invalid on attempt {attempt + 1}"
                        log_f.write(f"{msg}\n")
                        log_f.flush()
                        self._cleanup_invalid_wham(analysis_dir)
                        continue

                    msg = f"{method.upper()} succeeded on attempt {attempt + 1}"
                    log_f.write(f"{msg}\n")
                    return True, msg

                except Exception as e:
                    msg = f"{method.upper()} attempt {attempt + 1} failed: {e}"
                    log_f.write(f"{msg}\n")
                    log_f.flush()
                    self._cleanup_invalid_wham(analysis_dir)

                    if attempt == max_attempts - 1:
                        msg = f"{method.upper()} failed after {max_attempts} attempts"
                        log_f.write(f"{msg}\n")
                        return False, msg

        return False, f"{method.upper()} failed after {max_attempts} attempts"

    def _run_wham_analysis(
        self, nf: int, ms: int, msprof: int, cut_params: dict
    ) -> None:
        """Run WHAM analysis using bundled GPU library.

        Args:
            nf: Number of simulation files
            ms: Multisite parameter
            msprof: Multisite profile parameter
            cut_params: Cutoff parameters
        """
        from cphmd.wham import run_wham

        nsubs = self.state.alf_info["nsubs"]

        run_wham(
            analysis_dir=Path.cwd(),
            nf=nf,
            temp=self.config.temperature,
            nts0=ms,
            nts1=msprof,
            use_gshift=False,
            nsubs=nsubs,
            g_imp_path="../G_imp",
            log_file="analysis.log",
            fnex=self.config.fnex,
        )
        get_free_energy5(
            self.state.alf_info, ms=ms, msprof=msprof, **cut_params
        )

    def _run_lmalf_analysis(
        self, nf: int, ms: int, msprof: int, cut_params: dict
    ) -> None:
        """Run LMALF analysis using bundled GPU library.

        LMALF expects:
        - Lambda.dat: Combined lambda trajectory
        - ensweight.dat: Ensemble weights (optional)

        Args:
            nf: Number of simulation files
            ms: Multisite parameter
            msprof: Multisite profile parameter
            cut_params: Cutoff parameters for GetFreeEnergyLM
        """
        from cphmd.wham import run_lmalf, prepare_lmalf_input
        from cphmd.core.alf_utils import get_free_energy_lm

        # LMALF needs a combined Lambda.dat file
        # Collect all data/Lambda.*.*.dat files
        import sys
        print("[LMALF] Starting analysis...", flush=True)
        sys.stdout.flush()
        data_dir = Path("data")
        lambda_files = sorted(data_dir.glob("Lambda.*.*.dat"))

        if not lambda_files:
            raise FileNotFoundError("No Lambda.*.*.dat files found in data/")

        print(f"[LMALF] Found {len(lambda_files)} lambda files")

        # Prepare combined Lambda.dat for LMALF
        print("[LMALF] Preparing input files...")
        prepare_lmalf_input(
            analysis_dir=Path.cwd(),
            lambda_files=lambda_files,
            weight_files=None,  # Use uniform weights
        )

        # Get nsubs from alf_info
        nsubs = self.state.alf_info["nsubs"]

        # Run LMALF optimization
        print("[LMALF] Running LMALF optimization...")
        run_lmalf(
            analysis_dir=Path.cwd(),
            nf=nf,
            temp=self.config.temperature,
            ms=ms,
            msprof=msprof,
            max_iter=self.config.lmalf_max_iter,
            tolerance=self.config.lmalf_tolerance,
            nsubs=nsubs,
            g_imp_path="../G_imp",
            log_file="analysis.log",
            fnex=self.config.fnex,
        )
        print("[LMALF] LMALF optimization finished")

        # Check if LMALF produced valid output (not all zeros)
        out_file = Path("OUT.dat")
        if out_file.exists():
            out_data = np.loadtxt(out_file)
            if np.all(out_data == 0):
                print("[LMALF] Warning: LMALF produced all zeros - falling back to WHAM")
                # Fall back to WHAM
                self._run_wham_analysis(nf, ms, msprof, cut_params)
                return

        # Convert OUT.dat to b/c/x/s.dat
        print("[LMALF] Converting OUT.dat to b/c/x/s.dat...")
        # Filter cut_params to only keys accepted by get_free_energy_lm
        lm_keys = {"cutb", "cutc", "cutx", "cuts", "cutc2", "cutx2", "cuts2"}
        lm_params = {k: v for k, v in cut_params.items() if k in lm_keys}
        get_free_energy_lm(
            self.state.alf_info,
            ms=ms,
            msprof=msprof,
            **lm_params,
        )
        print("[LMALF] Analysis complete")

    def _run_hybrid_analysis(
        self, nf: int, ms: int, msprof: int, cut_params: dict
    ) -> None:
        """Run WHAM followed by short LMALF refinement.

        Phase 1: WHAM -> LMALF (5 iterations) for better solutions.
        Phase 2/3: WHAM only (landscape is stable, refinement unnecessary).
        """
        # Step 1: Run WHAM to get initial C.dat/V.dat and solve
        self._run_wham_analysis(nf, ms, msprof, cut_params)

        # Step 2: If Phase 1, refine with short LMALF
        if self.state.phase == 1:
            from cphmd.wham import run_lmalf, prepare_lmalf_input
            from cphmd.core.alf_utils import get_free_energy_lm

            lambda_files = sorted(Path("data").glob("Lambda.*.*.dat"))
            if lambda_files:
                prepare_lmalf_input(
                    analysis_dir=Path.cwd(),
                    lambda_files=lambda_files,
                    weight_files=None,
                )
                nsubs = self.state.alf_info["nsubs"]
                print("[Hybrid] LMALF refinement (5 iterations)...")
                run_lmalf(
                    analysis_dir=Path.cwd(),
                    nf=nf,
                    temp=self.config.temperature,
                    ms=ms,
                    msprof=msprof,
                    max_iter=5,
                    nsubs=nsubs,
                    g_imp_path="../G_imp",
                    log_file="analysis.log",
                    fnex=self.config.fnex,
                )
                # Check if LMALF produced valid output before overwriting WHAM result
                out_file = Path("OUT.dat")
                if out_file.exists():
                    out_data = np.loadtxt(out_file)
                    if np.all(out_data == 0):
                        print("[Hybrid] LMALF produced all zeros - keeping WHAM output")
                        return

                # Re-run GetFreeEnergy on LMALF's output
                lm_keys = {"cutb", "cutc", "cutx", "cuts", "cutc2", "cutx2", "cuts2"}
                lm_params = {k: v for k, v in cut_params.items() if k in lm_keys}
                get_free_energy_lm(
                    self.state.alf_info, ms=ms, msprof=msprof, **lm_params
                )
                print("[Hybrid] LMALF refinement complete")

    def _detect_phase_from_logs(self, run_idx: int) -> int | None:
        """Detect simulation phase from NSTEP in log files.

        Phase detection based on typical NSTEP ranges:
        - NSTEP < 50,000 → Phase 1 (short equilibration runs)
        - NSTEP < 1,000,000 → Phase 2 (medium refinement runs)
        - NSTEP ≥ 1,000,000 → Phase 3 (production runs)

        Args:
            run_idx: Run number to check

        Returns:
            Detected phase (1, 2, or 3) or None if detection fails
        """
        import re

        run_dir = self.config.input_folder / f"run{run_idx}"
        if not run_dir.exists():
            return None

        # Look for log files
        log_patterns = [
            run_dir / "log.0.0.out",
            run_dir / "*.log",
        ]

        nstep_values = []
        nstep_pattern = re.compile(r"NSTEP\s*=\s*(\d+)", re.IGNORECASE)

        for pattern in log_patterns:
            if pattern.name.startswith("*"):
                # Glob pattern
                for log_file in run_dir.glob(pattern.name):
                    try:
                        text = log_file.read_text()
                        matches = nstep_pattern.findall(text)
                        nstep_values.extend(int(m) for m in matches)
                    except Exception:
                        continue
            elif pattern.exists():
                try:
                    text = pattern.read_text()
                    matches = nstep_pattern.findall(text)
                    nstep_values.extend(int(m) for m in matches)
                except Exception:
                    continue

        if not nstep_values:
            return None

        max_nstep = max(nstep_values)

        if max_nstep < 50000:
            return 1
        elif max_nstep < 1000000:
            return 2
        else:
            return 3

    def _alf_analysis(self, run_idx: int, repeats: int, confirmation: bool = False):
        """Perform ALF analysis and update biases.

        Steps:
        1. Extract lambda values from binary trajectory files
        2. Compute bias energies for WHAM reweighting
        3. Run WHAM/LMALF free energy analysis
        4. Update bias parameters for next iteration

        Args:
            run_idx: Current run number
            repeats: Number of dynamics repeats completed
            confirmation: If True, this is a confirmation re-analysis (skip bias update)
        """

        if self.state.alf_info is None:
            raise ValueError("alf_info not initialized")

        home_path = os.getcwd()

        try:
            os.chdir(self.config.input_folder)

            # Determine analysis window
            # Phase 1: wider window (15 runs) — short simulations benefit from more history
            # Phase 2/3: narrower window (5 runs) — longer simulations are self-sufficient
            if self.state.phase == 1:
                im5 = max(run_idx - 15, 1)
            else:
                im5 = max(run_idx - 5, 1)

            # Create analysis directory
            analysis_dir = Path(f"analysis{run_idx}")
            analysis_dir.mkdir(exist_ok=True)

            # Copy previous analysis results
            prev_analysis = Path(f"analysis{run_idx - 1}")
            if prev_analysis.exists():
                for fname in ["b_sum.dat", "c_sum.dat", "x_sum.dat", "s_sum.dat"]:
                    src = prev_analysis / fname
                    dst = analysis_dir / fname.replace("_sum", "_prev")
                    if src.exists():
                        shutil.copy(src, dst)

            # Process lambda files
            os.chdir(analysis_dir)
            self.state.alf_info["nreps"] = self.config.nreps
            (Path("data")).mkdir(exist_ok=True)

            name = self.config.input_folder.name

            # Redirect verbose ALF output to analysis.log
            import contextlib
            log_file = Path("analysis.log")

            with open(log_file, "w") as log_f:
                with contextlib.redirect_stdout(log_f):
                    # Process lambda files (verbose)
                    for j in range(self.config.nreps):
                        for kk in range(repeats):
                            if self.state.phase in [1, 2]:
                                fnmsin = [f"../run{run_idx}/res/{name}_flat.{kk}.{j}.lmd"]
                            else:
                                fnmsin = [f"../run{run_idx}/res/{name}_prod.{kk}.{j}.lmd"]

                            fnmout = f"data/Lambda.{kk}.{j}.dat"

                            if Path(fnmsin[0]).exists():
                                convert_lambda_binary_to_text(
                                    self.state.alf_info, fnmout, fnmsin
                                )

                    # Run energy calculation (verbose)
                    # Phase 2/3 generate many more samples per run, so we need to subsample
                    if self.state.phase == 1:
                        skipE = 10   # 40K samples → 4K per simulation
                    else:
                        skipE = 100  # 450K+ samples → 4.5K per simulation
                    nf = get_energy_from_analysis_dir(
                        self.state.alf_info, im5, run_idx, skipE=skipE
                    )

            # Write nsubs and nblocks files for WHAM library
            nsubs = self.state.alf_info["nsubs"]
            nblocks = self.state.alf_info["nblocks"]
            np.savetxt("nsubs", np.array(nsubs).reshape((1, -1)), fmt=" %d")
            np.savetxt("nblocks", np.array([nblocks]), fmt=" %d")

            # Run free energy analysis
            ntersite = self.state.alf_info.get("ntersite", self._ntersite())
            ms, msprof = ntersite[0], ntersite[1]

            # Copy nbshift folder to analysis directory (required for WHAM)
            nbshift_src = Path("..") / "nbshift"
            nbshift_dst = Path("nbshift")
            if nbshift_src.exists() and not nbshift_dst.exists():
                shutil.copytree(nbshift_src, nbshift_dst)

            # Determine cutoff parameters based on phase
            # Coupling cutoffs scale with system size: sqrt(2/max_nsubs)
            # Reference = 2 substates (typical titratable residue)
            max_nsubs = max(nsubs) if len(nsubs) > 0 else 2
            coupling_scale = (2.0 / max(max_nsubs, 2)) ** 0.5

            if self.state.phase == 1:
                # Adaptive cutoffs: reads previous scaling.dat, auto-tunes
                cut_params = self._adapt_cutoffs(run_idx)
            elif self.state.phase == 2:
                # Phase 2 warmup: exponential decay from start to target
                # over ~20 runs to avoid the Phase 1→2 cutoff cliff.
                warmup_runs = 20
                p2_start = self.state.phase2_start_run or run_idx
                runs_in_p2 = max(run_idx - p2_start, 0)
                decay = min(runs_in_p2 / warmup_runs, 1.0)

                # Start values (moderate) → Target values (tight)
                cutb_start, cutb_target = 1.0, 0.05
                cutc_start = 4.0 * coupling_scale
                cutc_target = 0.5 * coupling_scale

                # Log-space interpolation: start * (target/start)^decay
                cutb = cutb_start * (cutb_target / cutb_start) ** decay
                cutc = cutc_start * (cutc_target / cutc_start) ** decay
                cutx = cutc  # same as cutc
                cuts = cutc

                cut_params = {
                    "cutb": cutb,
                    "cutc": cutc,
                    "cutx": cutx,
                    "cuts": cuts,
                }
                if runs_in_p2 < warmup_runs:
                    print(f"  Phase 2 warmup ({runs_in_p2}/{warmup_runs}): "
                          f"cutb={cutb:.3f} cutc={cutc:.3f}")
            else:  # phase 3
                # Check if system needs recovery (very skewed populations)
                phase3_recovery = False
                prev_pop_file = (Path("..")
                                 / f"analysis{run_idx - 1}" / "pop_strict.dat")
                if prev_pop_file.exists():
                    try:
                        prev_pops = np.loadtxt(prev_pop_file)
                        nsubs_check = self.state.alf_info.get("nsubs", [len(prev_pops)])
                        col = 0
                        for n in nsubs_check:
                            site_pops = prev_pops[col:col + n]
                            if len(site_pops) > 0:
                                diff = site_pops.max() - site_pops.min()
                                if diff > 0.7:
                                    phase3_recovery = True
                                    break
                            col += n
                    except Exception:
                        pass

                if phase3_recovery:
                    # Use Phase 2 cutoffs to recover
                    cut_params = {
                        "cutb": 0.05,
                        "cutc": 0.5 * coupling_scale,
                        "cutx": 0.5 * coupling_scale,
                        "cuts": 0.5 * coupling_scale,
                    }
                    print(f"  Phase 3 recovery: pop diff > 70% → using Phase 2 cutoffs")
                else:
                    cut_params = {
                        "cutb": 0.02,
                        "cutc": 0.2 * coupling_scale,
                        "cutx": 0.1 * coupling_scale,
                        "cuts": 0.1 * coupling_scale,
                    }

            # Apply exclusion flags for phases 2/3
            if self.state.phase != 1:
                if self.config.no_b_bias:
                    cut_params["calc_phi"] = False
                if self.config.no_c_bias:
                    cut_params["calc_psi"] = False
                if self.config.no_x_bias:
                    cut_params["calc_chi"] = False
                if self.config.no_s_bias:
                    cut_params["calc_omega"] = False

            # Compute transition counts from Lambda data for regularization
            from cphmd.core.transitions import (
                compute_transition_matrix,
                transition_matrix_to_coupling_weights,
                save_transition_matrix,
                compute_connectivity_metric,
            )
            data_dir = Path("data")
            pre_lambda_data, _ = load_lambda_data(data_dir)
            trans_matrices = None
            if pre_lambda_data is not None:
                trans_matrices = compute_transition_matrix(
                    pre_lambda_data, nsubs
                )
                save_transition_matrix(trans_matrices, Path("transitions.dat"))
                connectivity, weak_pairs = compute_connectivity_metric(
                    trans_matrices
                )
                print(f"  Transition connectivity: {connectivity:.2f} "
                      f"(min-pair transitions: {int(connectivity * 50)})")
                if weak_pairs:
                    for site, si, sj in weak_pairs:
                        print(f"    Weakest: site {site}, states {si}<->{sj}")
                trans_weights = transition_matrix_to_coupling_weights(
                    trans_matrices, nsubs, ms=ms
                )
                cut_params["transition_weights"] = trans_weights
                cut_params["connectivity"] = connectivity

            # Run WHAM with retry (3 attempts) and validation
            # Verbose output appended to analysis.log, summary message returned
            # nf was set by get_energy_from_analysis_dir (actual count of ESim files)
            wham_success, wham_msg = self._run_wham_with_retry(
                run_idx=run_idx,
                nf=nf,
                ms=ms,
                msprof=msprof,
                cut_params=cut_params,
                max_attempts=3,
            )
            print(wham_msg)  # Print summary to main output

            if not wham_success:
                print("All WHAM attempts failed, using zero bias updates")
                np.savetxt("b.dat", np.zeros((1, nblocks)), fmt=" %10.5f")
                np.savetxt("c.dat", np.zeros((nblocks, nblocks)), fmt=" %10.5f")
                np.savetxt("x.dat", np.zeros((nblocks, nblocks)), fmt=" %10.5f")
                np.savetxt("s.dat", np.zeros((nblocks, nblocks)), fmt=" %10.5f")

            # Load lambda data for phase checking and population stats
            data_dir = Path("data")
            lambda_data, _ = load_lambda_data(data_dir)

            if lambda_data is not None:
                # Write population statistics at two thresholds
                nsubs = self.state.alf_info.get("nsubs")
                pop_data = calculate_populations(
                    lambda_data, thresholds=(0.8, 0.985), nsubs=nsubs,
                )
                write_populations_file(Path("populations.dat"), pop_data)

                # Print short population summary for λ > 0.985
                if pop_data:
                    pop_strict = pop_data.get("pop_strict_norm", [])
                    if len(pop_strict) > 0:
                        if nsubs is not None and len(nsubs) > 1:
                            # Per-site populations on separate lines
                            print("Populations (λ>0.985):")
                            site_diffs = []
                            for si, (start, end) in enumerate(_per_site_ranges(nsubs)):
                                s = pop_strict[start:end]
                                site_diffs.append((max(s) - min(s)) * 100)
                                site_str = ", ".join(f"{p:.1%}" for p in s)
                                print(f"  Site {si+1} ({nsubs[si]} subs): [{site_str}] "
                                      f"diff={site_diffs[-1]:.1f}%")
                            print(f"  Worst-site diff={max(site_diffs):.1f}%")
                        else:
                            pop_str = ", ".join(f"{p:.1%}" for p in pop_strict)
                            frac_diff = (max(pop_strict) - min(pop_strict)) * 100
                            print(f"Populations (λ>0.985): [{pop_str}] diff={frac_diff:.1f}%")

                        # Detect unsampled states using last 5 runs of population history
                        # A state must be consistently unsampled to trigger biased Dirichlet
                        nsubs_eff = nsubs if nsubs is not None else [len(pop_strict)]
                        if self.state.patch_info is not None:
                            sites = list(self.state.patch_info["site"].unique())
                        else:
                            sites = list(range(1, len(nsubs_eff) + 1))

                        # Collect population history from last 5 analysis dirs
                        pop_history = []  # list of pop_strict_norm arrays
                        for prev_r in range(max(run_idx - 4, 1), run_idx + 1):
                            prev_pop_file = (Path("..")
                                             / f"analysis{prev_r}" / "pop_strict.dat")
                            if prev_pop_file.exists():
                                try:
                                    prev_pops = np.loadtxt(prev_pop_file)
                                    if len(prev_pops) == len(pop_strict):
                                        pop_history.append(prev_pops)
                                except Exception:
                                    pass
                        # Always include current run
                        pop_history.append(np.array(pop_strict))

                        # Average over history window
                        avg_pops = np.mean(pop_history, axis=0)

                        biased_alphas = {}
                        for site_idx, (start, end) in enumerate(_per_site_ranges(nsubs_eff)):
                            site_pops = avg_pops[start:end]
                            # States with < 5% average population over window
                            low_mask = site_pops < 0.05
                            if low_mask.any() and not low_mask.all():
                                low_indices = np.where(low_mask)[0]
                                # Bias toward the least sampled state
                                worst = low_indices[np.argmin(site_pops[low_indices])]
                                alpha = np.ones(len(site_pops))
                                # Stronger bias in Phase 2/3 (longer sims need harder push)
                                alpha_val = 10.0 if self.state.phase >= 2 else 3.0
                                alpha[worst] = alpha_val
                                biased_alphas[sites[site_idx]] = alpha.tolist()
                                print(f"  Biased Dirichlet site {sites[site_idx]}: "
                                      f"alpha[{worst}]={alpha_val:.0f} (avg pop "
                                      f"{site_pops[worst]:.1%} over {len(pop_history)} runs)")
                        # If no population-based Dirichlet, check transitions
                        if not biased_alphas and trans_matrices is not None:
                            from cphmd.core.transitions import find_weakest_transitions
                            weak_transitions = find_weakest_transitions(
                                trans_matrices, nsubs_eff, threshold_count=5
                            )
                            for site_idx, (si, sj) in weak_transitions.items():
                                # Start near the transition boundary
                                alpha = np.ones(nsubs_eff[site_idx])
                                alpha[si] = 2.0
                                alpha[sj] = 2.0
                                biased_alphas[sites[site_idx]] = alpha.tolist()
                                print(f"  Transition Dirichlet site {sites[site_idx]}: "
                                      f"alpha[{si}]=alpha[{sj}]=2 "
                                      f"(weak {si}<->{sj} transition)")

                        if biased_alphas:
                            self.state.forced_initial_lambdas = biased_alphas
                        else:
                            self.state.forced_initial_lambdas = None

                        # Save pop_strict for future history lookups
                        np.savetxt("pop_strict.dat", np.array(pop_strict),
                                   fmt=" %.6f")

                        # Wrong-direction detection (Phase 1 only)
                        if self.state.phase == 1 and run_idx >= 2:
                            worst_diff_frac = self._get_worst_site_diff(
                                pop_strict, nsubs_eff
                            )
                            # Append max_pop_diff to scaling.dat
                            analysis_dir_local = Path("..") / f"analysis{run_idx}"
                            scaling_file = analysis_dir_local / "scaling.dat"
                            if scaling_file.exists():
                                content = scaling_file.read_text()
                                if "max_pop_diff" not in content:
                                    lines = content.strip().split("\n")
                                    new_lines = []
                                    for line in lines:
                                        if line.startswith("#"):
                                            new_lines.append(
                                                line.rstrip() + " max_pop_diff"
                                            )
                                        else:
                                            new_lines.append(
                                                f"{line.rstrip()} {worst_diff_frac:.6f}"
                                            )
                                    scaling_file.write_text(
                                        "\n".join(new_lines) + "\n"
                                    )

                            # Check for 2 consecutive increases in pop diff
                            prev_dir = (
                                Path("..") / f"analysis{run_idx - 1}"
                            )
                            prev_scaling = self._read_scaling_dat(
                                prev_dir / "scaling.dat"
                            )
                            if (
                                prev_scaling
                                and "max_pop_diff" in prev_scaling
                            ):
                                prev_diff = prev_scaling["max_pop_diff"]
                                prev2_dir = (
                                    Path("..")
                                    / f"analysis{run_idx - 2}"
                                )
                                prev2_scaling = self._read_scaling_dat(
                                    prev2_dir / "scaling.dat"
                                )
                                if (
                                    worst_diff_frac > prev_diff
                                    and prev2_scaling
                                    and "max_pop_diff" in prev2_scaling
                                    and prev_diff > prev2_scaling["max_pop_diff"]
                                ):
                                    print(
                                        f"  WRONG DIRECTION: pop diff "
                                        f"{worst_diff_frac:.3f} > "
                                        f"{prev_diff:.3f} > "
                                        f"{prev2_scaling['max_pop_diff']:.3f}"
                                    )
                                    self._rollback_biases(
                                        analysis_dir, prev_dir
                                    )
                                    # Halve cutoffs via scaling.dat override
                                    if prev_scaling:
                                        halved = {
                                            "scaling": prev_scaling["scaling"],
                                            "bottleneck": "rollback",
                                            "cutb": max(
                                                prev_scaling["cutb"] * 0.5, 1.0
                                            ),
                                            "cutc": max(
                                                prev_scaling["cutc"] * 0.5, 1.0
                                            ),
                                            "cutx": max(
                                                prev_scaling["cutx"] * 0.5, 1.0
                                            ),
                                            "cuts": max(
                                                prev_scaling["cuts"] * 0.5, 1.0
                                            ),
                                        }
                                        with open(scaling_file, "w") as f:
                                            conn = prev_scaling.get("connectivity")
                                            if conn is not None:
                                                f.write(
                                                    "# scaling bottleneck cutb "
                                                    "cutc cutx cuts connectivity "
                                                    "max_pop_diff\n"
                                                )
                                                f.write(
                                                    f"{halved['scaling']:.6f} "
                                                    f"{halved['bottleneck']} "
                                                    f"{halved['cutb']:.4f} "
                                                    f"{halved['cutc']:.4f} "
                                                    f"{halved['cutx']:.4f} "
                                                    f"{halved['cuts']:.4f} "
                                                    f"{conn:.4f} "
                                                    f"{worst_diff_frac:.6f}\n"
                                                )
                                            else:
                                                f.write(
                                                    "# scaling bottleneck cutb "
                                                    "cutc cutx cuts max_pop_diff\n"
                                                )
                                                f.write(
                                                    f"{halved['scaling']:.6f} "
                                                    f"{halved['bottleneck']} "
                                                    f"{halved['cutb']:.4f} "
                                                    f"{halved['cutc']:.4f} "
                                                    f"{halved['cutx']:.4f} "
                                                    f"{halved['cuts']:.4f} "
                                                    f"{worst_diff_frac:.6f}\n"
                                                )

                # Generate population convergence plots
                from cphmd.analysis.population_convergence import generate_population_plots
                generate_population_plots(
                    input_folder=Path(".."),
                    max_run=run_idx,
                    output_dir=Path(home_path) / "plots",
                    nsubs=self.state.alf_info.get("nsubs"),
                )

                # Generate RMSD convergence plot (works with both convergence modes)
                try:
                    if (Path("..") / f"analysis{run_idx}" / "multisite").is_dir():
                        from cphmd.analysis.rmsd_convergence_plot import generate_rmsd_convergence_plots
                        generate_rmsd_convergence_plots(
                            input_folder=Path(".."),
                            max_run=run_idx,
                            output_dir=Path(home_path) / "plots",
                            nsubs=list(nsubs),
                            rmsd_state=getattr(self.state, "rmsd_state", None),
                        )
                except Exception as e:
                    print(f"Warning: RMSD convergence plots failed: {e}")

                # Generate 1D energy profile plots in current analysis dir
                try:
                    from cphmd.analysis.energy_profiles import plot_1d_profiles
                    plot_1d_profiles(
                        analysis_dir=Path.cwd(),
                        nsubs=list(nsubs),
                    )
                except Exception as e:
                    print(f"Warning: 1D energy profile plots failed: {e}")

            # --- RMSD-based convergence check ---
            if (self.config.convergence_mode == "rmsd"
                    and self.config.auto_phase_switch):
                from .rmsd_convergence import (
                    RMSDState, RMSDConvergenceConfig,
                    compute_site_rmsd, check_rmsd_phase_transition,
                    check_rmsd_stop,
                )
                if self.state.rmsd_state is None:
                    self.state.rmsd_state = RMSDState()

                rmsd_cfg = RMSDConvergenceConfig()
                analysis_dir = Path(f"analysis{run_idx}")

                # Find reference analysis directory (lag iterations ago)
                ref_run = run_idx - rmsd_cfg.lag
                ref_dir = Path(f"analysis{ref_run}")

                if ref_dir.is_dir() and (ref_dir / "multisite").is_dir():
                    site_rmsds, site_coverages = compute_site_rmsd(
                        analysis_dir, ref_dir, list(nsubs),
                    )
                    self.state.rmsd_state.rmsd_history.append(site_rmsds)
                    self.state.rmsd_state.coverage_history.append(site_coverages)
                    self.state.rmsd_state.run_indices.append(run_idx)

                    rmsd_label = ", ".join(
                        f"s{i+1}={r:.2f}({c:.0%})"
                        for i, (r, c) in enumerate(zip(site_rmsds, site_coverages))
                    )
                    print(f"RMSD (vs run {ref_run}): {rmsd_label}")

                    # Phase transition check
                    new_phase, reason = check_rmsd_phase_transition(
                        self.state.phase, self.state.rmsd_state, rmsd_cfg,
                    )
                    if new_phase != self.state.phase:
                        print(f"PHASE TRANSITION: {self.state.phase} → {new_phase}")
                        print(f"  {reason}")
                        if new_phase == 2 and self.state.phase2_start_run is None:
                            self.state.phase2_start_run = run_idx
                        self.state.phase = new_phase
                    else:
                        print(f"Phase check: {reason}")

                    # Auto-stop check in Phase 3
                    if self.config.auto_stop and self.state.phase == 3:
                        should_stop, stop_reason = check_rmsd_stop(
                            self.state.rmsd_state, rmsd_cfg,
                        )
                        if confirmation:
                            if should_stop:
                                self.state.converged = True
                                self.state.stop_reason = stop_reason
                                self.state.needs_confirmation = False
                                print(f"\n{'='*60}")
                                print(f"RMSD CONVERGENCE CONFIRMED at run {run_idx}")
                                print(f"  {stop_reason}")
                                print(f"{'='*60}\n")
                                with open("CONVERGED", "w") as f:
                                    f.write(f"RMSD converged at run {run_idx}\n")
                                    f.write(f"{stop_reason}\n")
                            else:
                                self.state.needs_confirmation = False
                                print(f"RMSD convergence NOT confirmed: {stop_reason}")
                        elif should_stop:
                            self.state.needs_confirmation = True
                            print(f"\n{'='*60}")
                            print(f"RMSD CONVERGENCE CANDIDATE at run {run_idx}")
                            print(f"  {stop_reason}")
                            print(f"  Triggering confirmation repeat...")
                            print(f"{'='*60}\n")
                        else:
                            print(f"RMSD stop check: {stop_reason}")

                    # Persist RMSD state
                    self.state.rmsd_state.save(Path("..") / "rmsd_state.json")
                else:
                    print(f"RMSD: reference analysis{ref_run} not available yet (need {rmsd_cfg.lag} runs)")

            # --- Population-based phase transition check (default) ---
            elif self.config.auto_phase_switch and lambda_data is not None:
                # Get delta_pKa for current phase
                from .cphmd_params import get_delta_pKa_for_phase
                delta_pKa = get_delta_pKa_for_phase(self.state.phase)

                # Build CpHMD parameters if pH is set
                cphmd_kwargs = {}
                if (self.config.pH is not None and self.config.nreps > 3
                        and self.state.patch_info is not None):
                    from .cphmd_params import compute_all_site_parameters
                    cphmd_params = compute_all_site_parameters(
                        self.state.patch_info,
                        self.config.temperature,
                        self.config.pH,
                    )
                    cphmd_kwargs = {
                        "data_dir": data_dir,
                        "patch_info": self.state.patch_info,
                        "effective_pH": cphmd_params.effective_pH,
                        "delta_pKa": delta_pKa,
                        "nreps": self.config.nreps,
                    }

                new_phase, reason = check_phase_transition(
                    self.state.phase,
                    lambda_data,
                    **cphmd_kwargs,
                    nsubs=nsubs,
                    connectivity=cut_params.get("connectivity"),
                )

                if new_phase != self.state.phase:
                    print(f"PHASE TRANSITION: {self.state.phase} → {new_phase}")
                    print(f"  Reason: {reason}")
                    if new_phase == 2 and self.state.phase2_start_run is None:
                        self.state.phase2_start_run = run_idx
                    self.state.phase = new_phase
                else:
                    print(f"Phase check: {reason}")

            # Always save current phase to file (for resumption)
            np.savetxt("phase.dat", np.array([self.state.phase]), fmt="%d")

            # Check stop criteria if in Phase 3 and auto_stop is enabled (population mode only)
            if (self.config.auto_stop and self.state.phase == 3
                    and lambda_data is not None
                    and self.config.convergence_mode == "population"):
                # Build stop criteria config (step5-style)
                timestep_fs = 4.0 if self.config.hmr else 2.0
                stop_config = StopCriteriaConfig(
                    timestep_fs=timestep_fs,
                    max_frac_diff=0.02,  # 2% population difference threshold
                )

                should_stop, stop_reason, stop_result = check_phase3_stop(
                    lambda_data, stop_config, nsubs=nsubs,
                )

                if confirmation:
                    # This is the confirmation re-analysis
                    if should_stop:
                        # Confirmed! Stop criteria met with additional data
                        self.state.converged = True
                        self.state.stop_reason = stop_reason
                        self.state.needs_confirmation = False
                        print(f"\n{'='*60}")
                        print(f"CONVERGENCE CONFIRMED at run {run_idx}")
                        print(f"  {stop_reason}")
                        print(f"  Fractions (λ>{stop_config.threshold_strict}): {stop_result.fractions}")
                        print(f"  Fraction diff: {stop_result.frac_diff_pct:.2f}%")
                        print(f"  Score: {stop_result.score:.4f}")
                        print(f"{'='*60}\n")

                        # Write convergence marker file
                        with open("CONVERGED", "w") as f:
                            f.write(f"Converged at run {run_idx}\n")
                            f.write(f"{stop_reason}\n")
                            f.write(f"Fractions: {stop_result.fractions}\n")
                            f.write(f"Score: {stop_result.score:.4f}\n")
                    else:
                        # Confirmation failed - continue search
                        self.state.needs_confirmation = False
                        print(f"\n{'='*60}")
                        print(f"CONVERGENCE NOT CONFIRMED at run {run_idx}")
                        print(f"  With additional data: {stop_reason}")
                        print(f"  Continuing optimization...")
                        print(f"{'='*60}\n")
                else:
                    # Normal check (first pass)
                    if should_stop:
                        # Trigger confirmation repeat
                        self.state.needs_confirmation = True
                        print(f"\n{'='*60}")
                        print(f"CONVERGENCE CANDIDATE at run {run_idx}")
                        print(f"  {stop_reason}")
                        print(f"  Fractions: {stop_result.fractions}")
                        print(f"  Fraction diff: {stop_result.frac_diff_pct:.2f}%")
                        print(f"  Triggering confirmation repeat...")
                        print(f"{'='*60}\n")
                    else:
                        print(f"Stop check: {stop_reason}")

            # Generate Henderson-Hasselbalch plots if enabled
            # Guard: require nreps > 3 for meaningful multi-point HH fitting
            if (self.config.generate_hh_plots and
                self.config.pH is not None and
                self.config.nreps > 3 and
                self.state.patch_info is not None):
                from cphmd.analysis.henderson_hasselbalch import generate_hh_analysis
                from .cphmd_params import compute_all_site_parameters

                delta_pKa = get_delta_pKa_for_phase(self.state.phase)
                cphmd_params = compute_all_site_parameters(
                    self.state.patch_info,
                    self.config.temperature,
                    self.config.pH,
                )

                generate_hh_analysis(
                    run_idx=run_idx,
                    data_dir=Path("data"),
                    patch_info=self.state.patch_info,
                    pH=cphmd_params.effective_pH,
                    delta_pKa=delta_pKa,
                    nreps=self.config.nreps,
                    output_dir=Path(home_path) / "plots",
                    ncentral=self.state.alf_info.get("ncentral", self.config.nreps // 2),
                )

            # Update variables for next run (skip during confirmation)
            if not confirmation:
                set_vars_from_analysis_dir(
                    Path.cwd(), self.state.alf_info, step=run_idx + 1
                )

            # Cleanup old analysis directories to save disk space
            # Keep enough history for the analysis window (15 for Phase 1, 5 for Phase 2/3)
            keep_window = 15 if self.state.phase == 1 else 5
            if self.config.cleanup_old_analysis and run_idx > keep_window:
                old_analysis = Path(home_path) / f"analysis{run_idx - keep_window}"
                if old_analysis.exists():
                    shutil.rmtree(old_analysis)
                    print(f"Cleaned up {old_analysis}")

            # Clean up large intermediate files (ESim/Lambda consumed by WHAM)
            for subdir in ("Energy", "Lambda"):
                p = Path(subdir)
                if p.exists():
                    shutil.rmtree(p)

            print(f"ALF analysis complete for run {run_idx}")

        finally:
            os.chdir(home_path)


def run_alf_simulation(config: ALFConfig) -> None:
    """Convenience function to run ALF simulation.

    Args:
        config: ALFConfig with simulation parameters

    Example:
        >>> config = ALFConfig(
        ...     input_folder="my_system",
        ...     pH=7.0,
        ...     temperature=298.15,
        ...     nreps=8,
        ...     start=1,
        ...     end=20,
        ... )
        >>> run_alf_simulation(config)
    """
    sim = ALFSimulation(config)
    sim.run()


