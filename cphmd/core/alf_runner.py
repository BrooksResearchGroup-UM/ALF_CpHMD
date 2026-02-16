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

import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from cphmd.core import ElecType, PhaseType, VdwType
from cphmd.core.alf_utils import (
    convert_lambda_binary_to_parquet,
    init_vars,
    set_vars_from_analysis_dir,
)
from cphmd.core.bias_analyzer import BiasAnalyzer
from cphmd.core.convergence_tracker import ConvergenceTracker
from cphmd.core.dynamics_runner import DynamicsRunner
from cphmd.core.g_imp_provisioner import GImpProvisioner
from cphmd.core.phase_switcher import (
    PhaseTransitionConfig,
    load_lambda_data,
)

# Type aliases
RestrainType = Literal["SCAT", "NOE"]
AnalysisMethod = Literal["wham", "lmalf", "hybrid", "nonlinear"]
ConvergenceMode = Literal["population", "rmsd"]
PrepFormat = Literal["default", "legacy", "auto"]


@dataclass
class ALFConfig:
    """Configuration for ALF simulation.

    Attributes:
        input_folder: Path to the prepared system folder (contains prep/ subdirectory)
        toppar_dir: Path to topology/parameter files
        temperature: Simulation temperature in Kelvin
        pH: Enable CpHMD pH coupling (effective_pH auto-computed from macro-pKa)
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
    pH: bool = False
    hmr: bool | None = None
    start: int = 1
    end: int = 20
    phase: PhaseType = 1
    nreps: int | None = None  # Defaults to MPI size
    restrains: RestrainType = "SCAT"
    restrain_hydrogens: bool = False
    # Bias type: clean interface for enabling/disabling parameter types.
    # Each letter enables that bias type: b(linear), c(coupling), x(skew), s(endpoint),
    # t(opposite-endpoint), u(endpoint-cubed). None = use individual no_*_bias flags.
    bias_type: str | None = None  # e.g. "bc", "bcx", "bcxs", "bcxst", "bcxstu"

    no_b_bias: bool = False  # Disable linear (phi) bias updates
    no_c_bias: bool = False  # Disable coupling (psi) bias updates
    no_x_bias: bool = False
    no_s_bias: bool = False
    no_t_bias: bool = True   # Disable t-term bias updates (bcxstu)
    no_u_bias: bool = True   # Disable u-term bias updates (bcxstu)
    no_pka_bias: bool = False  # Disable pKa-based bias shifts
    auto_phase_switch: bool = False  # Enable automatic phase switching
    auto_stop: bool = False  # Enable automatic stop when converged in Phase 3
    phase_transition: PhaseTransitionConfig = field(default_factory=PhaseTransitionConfig)
    convergence_mode: ConvergenceMode = "population"  # "population" or "rmsd"
    cleanup_old_analysis: bool = False  # Remove old analysis directories to save disk space
    generate_hh_plots: bool = True  # Generate Henderson-Hasselbalch plots
    cent_ncres: int | bool = False

    # Preset biases (converged single-site biases from cubic box simulations)
    use_presets: bool = False  # Use preset biases for initial ALF parameters

    # Inter-site coupling
    coupling: Literal[0, 1, 2] = 0  # 0=none, 1=full c/x/s, 2=c-only
    coupling_profile: bool | None = None  # None=follow coupling, True/False=override

    # Phase 1 x/s coverage gate: enable coupling when all states are visited
    min_xs_coverage_runs: int = 3  # consecutive covered runs before Phase 1 x/s enable
    phase1_xs_cutoff: float = 2.0  # cutx/cuts value when x/s enabled in Phase 1

    # FNEX softmax constraint parameter (controls bias potential shape)
    fnex: float = 5.5
    # Optional overrides for bias potential shape constants (None = derive from fnex)
    chi_offset: float | None = None  # s-term sigmoid offset (default: 4*exp(-fnex))
    omega_decay: float | None = None  # x-term exponential decay (default: fnex)
    chi_offset_t: float = 0.012  # t-term sigmoid offset (independent of FNEX)
    chi_offset_u: float = 0.012  # u-term offset (independent of FNEX)

    # Lambda dynamics mass and friction (per-block via LDIN)
    # None = auto: HMR(4fs) → mass=12.0/fbeta=5.0; non-HMR(2fs) → mass=5.0/fbeta=7.0
    lambda_mass: float | None = None   # Lambda mass in amu·Å² (BIMLAM)
    lambda_fbeta: float | None = None  # Lambda Langevin friction in ps⁻¹ (BIBLAM)

    # G_imp entropy profile bins (2D=N×N, 1D=N²)
    # None = use bundled/existing G_imp as-is; set explicitly to validate & regenerate
    # int = same bins for all phases; list[int] = per-phase [phase1, phase2, phase3]
    g_imp_bins: int | list[int] | None = None
    cutlsum: float = 0.8  # G12 conditional threshold (λ_i + λ_j > cutlsum)

    # WHAM endpoint bin weight: prevents solver from sacrificing the last histogram bin.
    # float = same weight for all phases; list[float] = per-phase [phase1, phase2, phase3]
    # Phase 1 (far from solution): low weight avoids sign-switching.
    # Phase 3 (near solution): high weight locks endpoint accuracy.
    endpoint_weight: float | list[float] = 100.0
    # Exponential ramp decay rate for endpoint weighting.
    # Controls how many bins near each endpoint are upweighted:
    # w(k) = 1 + (W-1)*exp(-α*min(k, N-1-k)), where α = endpoint_decay.
    # At α=2: weights ~100, 14, 2.8, 1.0 (3-bin effective ramp).
    # float = same for all phases; list[float] = per-phase [phase1, phase2, phase3].
    endpoint_decay: float | list[float] = 2.0

    # gshift correction in WHAM Stage 2 profile extraction
    use_gshift: bool = True  # Apply G_imp entropy shifts (recommended; see docs/WHAM/gshift_theory.md)

    # Analysis method configuration
    analysis_method: AnalysisMethod = "wham"  # "wham", "lmalf", "hybrid", or "nonlinear"
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

    # Replica exchange (pH-ladder swaps between dynamics segments)
    replica_exchange: Any = field(default=None)  # ReplicaExchangeConfig or None

    # Prep format: "default" (CpHMD), "legacy" (msld-py-prep), or "auto" (detect)
    prep_format: PrepFormat = "auto"
    # Legacy setup script (auto-discovered from prep/*.inp if not set)
    legacy_setup_script: str | None = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.input_folder = Path(self.input_folder).resolve()
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
            self.lambda_mass = 7.0 if self.hmr else 4.0
        if self.lambda_fbeta is None:
            self.lambda_fbeta = 7.0 if self.hmr else 10.0

        # bias_type enum overrides individual no_*_bias flags
        if self.bias_type is not None:
            bt = self.bias_type.lower()
            valid = {"bc", "bcx", "bcxs", "bcxst", "bcxstu"}
            if bt not in valid:
                raise ValueError(
                    f"Invalid bias_type={bt!r}. Must be one of: {', '.join(sorted(valid))}"
                )
            self.no_b_bias = "b" not in bt
            self.no_c_bias = "c" not in bt
            self.no_x_bias = "x" not in bt
            self.no_s_bias = "s" not in bt
            self.no_t_bias = "t" not in bt
            self.no_u_bias = "u" not in bt

        # Hierarchy enforcement: ntriangle encoding requires c → x → s → t → u.
        # Disabling an earlier type forces all later types off.
        if self.no_x_bias:
            self.no_s_bias = True
        if self.no_x_bias or self.no_s_bias:
            self.no_t_bias = True
            self.no_u_bias = True

        # Warn when experimental t/u terms are enabled
        if not self.no_t_bias or not self.no_u_bias:
            import warnings
            warnings.warn(
                "t/u bias terms are experimental and not yet validated. "
                "Use bias_type='bcxstu' only for testing.",
                stacklevel=2,
            )

        # Normalize replica_exchange: None → disabled config
        if self.replica_exchange is None:
            from cphmd.core.replica_exchange import ReplicaExchangeConfig
            self.replica_exchange = ReplicaExchangeConfig()
        elif isinstance(self.replica_exchange, dict):
            from cphmd.core.replica_exchange import ReplicaExchangeConfig
            self.replica_exchange = ReplicaExchangeConfig(**self.replica_exchange)
        elif isinstance(self.replica_exchange, bool):
            from cphmd.core.replica_exchange import ReplicaExchangeConfig
            self.replica_exchange = ReplicaExchangeConfig(enabled=self.replica_exchange)

    @property
    def ntriangle(self) -> int:
        """CUDA ntriangle encoding: 1(c) + 2(x) + 2(s) + 2(t) + 2(u).

        Controls which pair-interaction parameter types are computed in WHAM/LMALF/nonlinear
        CUDA kernels. Higher values include more bias term types.

        Returns:
            1 (bc), 3 (bcx), 5 (bcxs), 7 (bcxst), or 9 (bcxstu).
        """
        return (
            1
            + (0 if self.no_x_bias else 2)
            + (0 if self.no_s_bias else 2)
            + (0 if self.no_t_bias else 2)
            + (0 if self.no_u_bias else 2)
        )

    @staticmethod
    def resolve_g_imp_bins(
        g_imp_bins: "int | list[int] | None", phase: int
    ) -> int | None:
        """Resolve g_imp_bins for a specific phase.

        Args:
            g_imp_bins: Single int (all phases), list of 3 ints (per-phase), or None.
            phase: Phase number (1, 2, or 3).

        Returns:
            Resolved bin count for this phase, or None if g_imp_bins is None.
        """
        if g_imp_bins is None:
            return None
        if isinstance(g_imp_bins, int):
            return g_imp_bins
        # list[int] — index by phase (1-indexed → 0-indexed)
        return g_imp_bins[phase - 1]

    @staticmethod
    def resolve_endpoint_weight(
        endpoint_weight: "float | list[float]", phase: int
    ) -> float:
        """Resolve endpoint_weight for a specific phase.

        Args:
            endpoint_weight: Single float (all phases) or list of 3 floats (per-phase).
            phase: Phase number (1, 2, or 3).

        Returns:
            Resolved endpoint weight for this phase.
        """
        if isinstance(endpoint_weight, (int, float)):
            return float(endpoint_weight)
        return float(endpoint_weight[phase - 1])

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

    # EWBS (Energy-Weighted Bias Stability) state — tracks per-type
    # smoothed RMS bias changes for convergence detection
    ewbs_state: Any = None  # EWBSState instance, lazy-initialized

    # Phase 2→1 regression tracking
    stuck_phase2_count: int = 0  # consecutive stuck runs in Phase 2
    phase_regression_count: int = 0  # number of Phase 2→1 regressions performed

    # Phase 1 x/s coverage gate: consecutive runs with all states visited per site
    xs_coverage_count: int = 0

    # Replica exchange state (persisted alongside ewbs_state)
    exchange_state: Any = None  # ExchangeState instance, lazy-initialized


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
        self._log_file = None  # Current CHARMM log file object (kept for compat)
        self._python_log = None  # Per-rank Python output file
        self._bias_analyzer = BiasAnalyzer(config)
        self._convergence = ConvergenceTracker(config, self.state)
        self._dynamics = DynamicsRunner(config, self.state)
        self._exchanger = None  # ReplicaExchanger, created in initialize() if enabled

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

    def _redirect_python_output(self):
        """Redirect Python stdout to per-rank log files.

        Must be called AFTER _init_mpi() (need rank) and BEFORE _init_charmm()
        (pyCHARMM's Fortran engine may hijack fd 1).  This matches the legacy
        script pattern that redirects sys.stdout before ``import pycharmm``.

        stderr is kept on the real stderr so that Python tracebacks and errors
        remain visible in the SLURM output.
        """
        log_path = self.config.input_folder / f"python_log_rank{self.state.rank}.out"
        self._python_log = open(log_path, "w")
        sys.stdout = self._python_log
        # Keep sys.stderr pointing to real stderr — errors must stay visible

    def _redirect_output(self, run_idx: int, k: int, replica_idx: int):
        """Redirect CHARMM output to a separate log file."""
        self._dynamics.redirect_output(run_idx, k, replica_idx)

    def _return_output(self):
        """Return CHARMM output to standard output."""
        self._dynamics.return_output()

    def _init_charmm(self):
        """Initialize CHARMM and read topology files."""
        from .charmm_utils import read_topology_files

        # Verify CHARMM environment
        charmm_lib = os.environ.get("CHARMM_LIB_DIR")
        if not charmm_lib or not Path(charmm_lib).is_dir():
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
            "g_imp_bins": ALFConfig.resolve_g_imp_bins(self.config.g_imp_bins, self.state.phase) or 32,
            "cutlsum": self.config.cutlsum,
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
            if self.config.use_presets:
                site_res_types = self._get_site_residue_types()
                preset_cfg = self._resolve_preset_config()
                init_vars(
                    self.config.input_folder, alf_info,
                    use_presets=True, preset_config=preset_cfg,
                    site_residue_types=site_res_types,
                )
            else:
                init_vars(self.config.input_folder, alf_info)


    # Patch name prefix → canonical residue type for presets
    _PATCH_TO_RESTYPE = {
        "ASP": "ASP", "ASH": "ASP", "ASPO": "ASP",
        "GLU": "GLU", "GLH": "GLU", "GLUO": "GLU",
        "HSP": "HSP", "HSD": "HSP", "HSE": "HSP",
        "HSPO": "HSP", "HSPD": "HSP", "HSPE": "HSP",
        "LYS": "LYS", "LYSO": "LYS", "LYSU": "LYS",
        "TYR": "TYR", "TYRO": "TYR", "TYRU": "TYR",
        "ARG": "ARG", "ARGO": "ARG", "ARU1": "ARG", "ARU2": "ARG",
        "SER": "SER", "SERO": "SER", "SERD": "SER",
        "CYS": "CYS", "CYSO": "CYS", "CYSD": "CYS",
        "THR": "THR", "THRO": "THR", "THRD": "THR",
        "NRED": "NRED", "NREDO": "NRED", "NRD2": "NRED", "NRDU": "NRED",
    }

    def _get_site_residue_types(self) -> list[str]:
        """Extract canonical residue type for each titratable site from patches.dat.

        Reads the PATCH column for the first subsite (s1) of each site and maps
        the patch name to a canonical residue type (ASP, GLU, HSP, LYS, TYR, etc.).

        Returns:
            List of residue type names, one per site (in site order).
        """
        df = self.state.patch_info
        res_types = []
        for site in sorted(df["site"].unique(), key=int):
            site_rows = df[df["site"] == site]
            # Use first row's PATCH name to identify residue type
            patch_name = site_rows.iloc[0]["PATCH"]
            # Try exact match first, then prefix matching
            restype = self._PATCH_TO_RESTYPE.get(patch_name)
            if restype is None:
                # Try progressively shorter prefixes (e.g., "ASH1" → "ASH" → "AS")
                for length in range(len(patch_name), 1, -1):
                    restype = self._PATCH_TO_RESTYPE.get(patch_name[:length])
                    if restype:
                        break
            if restype is None:
                restype = "UNKNOWN"
            res_types.append(restype)
        return res_types

    def _resolve_preset_config(self) -> str | None:
        """Resolve preset configuration name from elec_type + vdw_type.

        Mapping: pmeex→pme_ex, pmeon→pme_on, pmenn→pme_nn,
                 fshift→fshift, fswitch→fswitch; vswitch/vfswitch.

        Returns:
            Preset config name (e.g., "pme_ex_vswitch") or None if types unknown.
        """
        elec_map = {
            "pmeex": "pme_ex", "pmeon": "pme_on", "pmenn": "pme_nn",
            "fshift": "fshift", "fswitch": "fswitch",
        }
        elec = elec_map.get(self.config.elec_type)
        vdw = self.config.vdw_type  # already "vswitch" or "vfswitch"
        if elec and vdw:
            return f"{elec}_{vdw}"
        return None

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
                "Legacy prep format requires 'box' in alf_info.py "
                "(cubic box edge length). Add: alf_info['box'] = <value>"
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
        """Ensure G_imp entropy data is available for WHAM."""
        nsubs = (
            self.state.alf_info.get("nsubs", [])
            if self.state.alf_info
            else []
        )
        expected_bins = ALFConfig.resolve_g_imp_bins(
            self.config.g_imp_bins, self.state.phase
        )
        provisioner = GImpProvisioner(
            input_folder=self.config.input_folder,
            nsubs=nsubs,
            fnex=self.config.fnex,
            cutlsum=self.config.cutlsum,
            g_imp_bins=expected_bins,
        )
        provisioner.ensure_available()

    def _regenerate_g_imp_if_needed(self, old_phase: int, new_phase: int):
        """Regenerate G_imp if the new phase requires different bins."""
        nsubs = (
            self.state.alf_info.get("nsubs", [])
            if self.state.alf_info
            else []
        )
        new_bins = ALFConfig.resolve_g_imp_bins(self.config.g_imp_bins, new_phase)
        provisioner = GImpProvisioner(
            input_folder=self.config.input_folder,
            nsubs=nsubs,
            fnex=self.config.fnex,
            cutlsum=self.config.cutlsum,
            g_imp_bins=new_bins,
        )
        provisioner.regenerate_if_needed(
            old_phase, new_phase,
            resolve_bins=lambda p: ALFConfig.resolve_g_imp_bins(self.config.g_imp_bins, p),
            alf_info=self.state.alf_info,
        )

    def _supplement_g_imp(self, g_imp_dir: Path, nsubs, bins: int | None):
        """Ensure G1, G12 and cross-site files exist, computing if missing."""
        provisioner = GImpProvisioner(
            input_folder=self.config.input_folder,
            nsubs=nsubs,
            fnex=self.config.fnex,
            cutlsum=self.config.cutlsum,
        )
        provisioner.supplement(g_imp_dir, bins)

    @staticmethod
    def _detect_g_imp_bins(g_imp_dir: Path) -> int | None:
        """Detect bin count from existing G_imp files."""
        return GImpProvisioner.detect_bins(g_imp_dir)

    def _run_bias_guessing_phase(self):
        """Run initial bias guessing before the main simulation loop.

        Evaluates solvated-vacuum energy differences to estimate initial
        linear (b) and barrier (c) biases.  Only runs on rank 0 when
        presets are not in use and ``.biases_guessed`` marker is absent.

        Supports both default format (uses setup_crystal + patches.dat)
        and legacy format (uses setup_legacy + siteX_subY named selections).

        After guessing, clears the CHARMM session so the main loop
        starts with a fresh PSF load.
        """
        if self.config.use_presets:
            return
        marker = self.config.input_folder / "analysis0" / ".biases_guessed"
        if marker.exists():
            return

        is_legacy = self.config.prep_format == "legacy"

        if self.state.rank == 0:
            import pycharmm.lingo as lingo

            from .alf_utils import init_vars
            from .bias_guesser import guess_initial_biases_combined
            from .charmm_utils import clear_block, clear_crystal

            nsubs = self.state.alf_info["nsubs"].astype(int).tolist()

            print("Guessing initial biases from solvated-vacuum energy difference...")
            sys.stdout.flush()

            # Suppress CHARMM output during bias guessing (many BLOCK/energy
            # evaluations produce huge output for large systems)
            lingo.charmm_script("prnlev 0")

            # Load PSF, coordinates, crystal, and nonbonded for energy evaluation
            if is_legacy:
                self._dynamics.setup_legacy(
                    run_idx=self.config.start, letter="", k=0, replica_idx=0
                )
            else:
                self._dynamics.setup_crystal(
                    run_idx=self.config.start, letter="", k=0, replica_idx=0
                )

            try:
                if is_legacy:
                    # Legacy MSLD: estimate c only from solvated energies.
                    # Vacuum step (delete water + CUTNB=999) crashes CHARMM
                    # with "MAKINB: Too many bonds" for large block counts.
                    # Without ΔΔE_solvation, b includes solvent screening
                    # and can have wrong sign — keep b=0 (default).
                    from .bias_guesser import guess_initial_biases
                    _, c = guess_initial_biases(
                        self.state.patch_info, nsubs, fnex=self.config.fnex,
                        legacy=True,
                    )
                    c *= 5.0
                    b = np.zeros((1, sum(nsubs)))
                else:
                    b, c = guess_initial_biases_combined(
                        self.state.patch_info, nsubs, fnex=self.config.fnex,
                    )
            except Exception as e:
                print(f"  Bias guessing failed: {e}. Using zero biases.")
                b, c = None, None

            if b is not None:
                init_vars(
                    self.config.input_folder, self.state.alf_info,
                    b_init=b, c_init=c,
                )
                print(
                    f"  Initial bias guess applied: "
                    f"b range [{b.min():.2f}, {b.max():.2f}], "
                    f"c range [{c.min():.2f}, {c.max():.2f}]"
                )

            marker.touch()

            # Restore CHARMM output level
            lingo.charmm_script("prnlev 5")

            # Clear CHARMM session — main loop will reload fresh
            clear_block()
            clear_crystal()
            lingo.charmm_script("delete atom sele all end")
            self.state.structure_loaded = False

            sys.stdout.flush()

        self._comm.Barrier()

    def initialize(self):
        """Perform full initialization sequence."""
        self._init_mpi()
        # Wire MPI into bias analyzer for distributed WHAM
        self._bias_analyzer.set_mpi(self._comm, self.state.rank, self.state.size)
        self._redirect_python_output()

        # Save real stderr for error reporting (sys.stdout is now redirected)
        real_stderr = sys.__stderr__

        try:
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

            # Create replica exchanger if enabled (version check runs here)
            if self.config.replica_exchange.enabled:
                from cphmd.core.replica_exchange import ReplicaExchanger

                self._exchanger = ReplicaExchanger(
                    self.config.replica_exchange,
                    self._comm,
                    self.state.rank,
                    self.state.size,
                )
                # Load persisted exchange state
                exchange_state_path = self.config.input_folder / "exchange_state.json"
                self._exchanger.load_state(exchange_state_path)

            self._initialized = True
        except Exception:
            # Print the error to real stderr so it appears in SLURM output
            import traceback
            print(
                f"\n[RANK {self.state.rank}] INITIALIZATION FAILED:\n"
                f"{traceback.format_exc()}",
                file=real_stderr,
                flush=True,
            )
            raise

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

        # Phase 0: Initial bias guessing (before main loop)
        self._run_bias_guessing_phase()

        # Find last completed run and resume (rank 0 only — cleanup is not MPI-safe)
        if self.state.rank == 0:
            start_run = self._find_last_run()
        else:
            start_run = None
        start_run = self._comm.bcast(start_run, root=0)
        # Sync phase/phase2_start_run detected by rank 0
        self.state.phase = self._comm.bcast(self.state.phase, root=0)
        self.state.phase2_start_run = self._comm.bcast(
            self.state.phase2_start_run, root=0
        )

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

                # Setup system (method depends on prep format).
                # With exchange, force full setup for k>0 so each repeat
                # gets fresh crystal/BLOCK state (exchange homogenizes
                # all replicas into one microstate by end of k=0).
                force_setup = (k > 0 and self._exchanger is not None)
                if self.config.prep_format == "legacy":
                    self._setup_legacy(run_idx, letter, k, replica_idx)
                else:
                    self._setup_crystal(run_idx, letter, k, replica_idx,
                                        force=force_setup)
                    self._build_block_commands(run_idx, letter, k, replica_idx,
                                              force=force_setup)
                    self._apply_restraints(run_idx, k, force=force_setup)

                # Run minimization on first runs if needed
                if run_idx <= 5:
                    self._run_minimization(run_idx, replica_idx)

                # Dispatch to segmented or regular dynamics
                if self._exchanger is not None:
                    self._run_dynamics_with_exchange(run_idx, letter, k, replica_idx)
                else:
                    self._run_dynamics(run_idx, letter, k, replica_idx)

                # Return CHARMM output to stdout
                self._return_output()

            self._comm.Barrier()
            elapsed = time.time() - start_time
            print(f"Run {run_idx}.{k} dynamics completed in {elapsed:.1f}s")
            sys.stdout.flush()

        # ALF analysis — all ranks participate for distributed WHAM
        start_time = time.time()
        self._alf_analysis(run_idx, repeats)
        elapsed = time.time() - start_time
        if self.state.rank == 0:
            print(f"Analysis completed in {elapsed:.1f}s")
            sys.stdout.flush()

        # Save exchange state after analysis
        if self._exchanger is not None and self.state.rank == 0:
            exchange_state_path = self.config.input_folder / "exchange_state.json"
            self._exchanger.save_state(exchange_state_path)
            run_dir = self.config.input_folder / f"run{run_idx}"
            self._exchanger.write_exchange_log(run_dir / "exchange_log.txt", run_idx)

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
            sys.stdout.flush()

            # Re-run analysis with additional data (all ranks for distributed WHAM)
            if self.state.rank == 0:
                print("Re-analyzing with confirmation data...")
                sys.stdout.flush()
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

    def _setup_crystal(self, run_idx: int, letter: str, k: int, replica_idx: int,
                       force: bool = False):
        """Setup crystal/periodic boundary conditions."""
        self._dynamics.setup_crystal(run_idx, letter, k, replica_idx, force=force)

    def _setup_legacy(self, run_idx: int, letter: str, k: int, replica_idx: int):
        """Set up CHARMM session by streaming a legacy setup script."""
        self._dynamics.setup_legacy(run_idx, letter, k, replica_idx)

    def _apply_legacy_restraints(self):
        """Stream user-provided restraints for legacy mode."""
        self._dynamics.apply_legacy_restraints()

    def _build_block_commands(self, run_idx: int, letter: str, k: int, replica_idx: int,
                              force: bool = False):
        """Build and execute BLOCK/MSLD commands for lambda dynamics."""
        self._dynamics.build_block_commands(run_idx, letter, k, replica_idx, force=force)

    def _apply_restraints(self, run_idx: int, k: int = 0, force: bool = False):
        """Apply SCAT or NOE restraints to titratable atoms."""
        self._dynamics.apply_restraints(run_idx, k, force=force)

    def _run_minimization(self, run_idx: int, replica_idx: int):
        """Run energy minimization before dynamics."""
        self._dynamics.run_minimization(run_idx, replica_idx)

    def _run_dynamics(self, run_idx: int, letter: str, k: int, replica_idx: int):
        """Execute molecular dynamics with BLADE GPU acceleration."""
        self._dynamics.run_dynamics(run_idx, letter, k, replica_idx)

    def _run_dynamics_with_exchange(
        self, run_idx: int, letter: str, k: int, replica_idx: int,
    ):
        """Run segmented dynamics with replica exchange attempts between segments.

        Splits production dynamics into segments of ``exchange_freq`` steps,
        with exchange attempts between adjacent pH replicas after each segment.
        Equilibration (if any) runs as a non-segmented prefix (no exchanges
        during eq — biases haven't stabilized yet).

        Requires one replica per MPI rank (synchronous mode: nreps == size).
        """
        from cphmd.core.cphmd_params import get_delta_pKa_for_phase

        _, nsteps_prod, _, _ = self._dynamics.get_production_steps()
        n_segments = self._exchanger.compute_n_segments(nsteps_prod)
        seg_steps = self.config.replica_exchange.exchange_freq

        # Reset permutation to identity at the start of each run.
        # The exchange log captures the final permutation per run.
        npairs = self.state.size - 1
        self._exchanger.state.permutation = list(range(npairs + 1))

        nblocks = self.state.alf_info["nblocks"]
        delta_pKa = get_delta_pKa_for_phase(self.state.phase)

        if n_segments == 0:
            # exchange_freq > nsteps_prod: fall back to regular dynamics
            self._dynamics.run_dynamics(run_idx, letter, k, replica_idx)
            return

        # For k>0: restart equilibration from previous run's production restart
        # to avoid inheriting k=0's exchange-homogenized lambda state.
        # k=0 exchange funnels all replicas into one microstate — restarting
        # from the previous run gives diverse, independent starting conditions.
        prev_rst = None
        if k > 0 and run_idx > 0:
            prev_rst = self._find_exchange_restart(run_idx, replica_idx)

        # Run equilibration if needed (Phase 2: 10000 steps).
        # No exchanges during eq — biases aren't stable yet.
        eq_rst = self._dynamics.run_equilibration(
            run_idx, k, replica_idx, restart_from=prev_rst,
        )

        # First segment reads from eq restart (if produced) or starts fresh
        restart_override = eq_rst

        for seg in range(n_segments):
            is_first = (seg == 0)

            rst_path, lmd_path = self._dynamics.run_dynamics_segment(
                run_idx, letter, k, replica_idx,
                segment_idx=seg,
                nsteps=seg_steps,
                restart_from=restart_override,
                is_first_segment=is_first,
                blade_ready=(eq_rst is not None),
            )

            # All ranks synchronize before exchange attempt
            self._comm.Barrier()

            # Attempt exchange
            partner_rst = self._exchanger.attempt_exchange(
                segment_idx=seg,
                run_idx=run_idx,
                lmd_path=lmd_path,
                rst_path=rst_path,
                nblocks=nblocks,
                patch_info=self.state.patch_info,
                delta_pKa=delta_pKa,
                temperature=self.config.temperature,
            )

            # Ensure all restart files are accessible before next segment
            self._comm.Barrier()

            # Next segment reads partner's restart if swapped, else own restart
            restart_override = partner_rst if partner_rst is not None else rst_path

        # Turn off BLADE after all segments
        self._dynamics.finish_blade()

    def _find_exchange_restart(self, run_idx: int, replica_idx: int) -> Path | None:
        """Find the previous run's last segment restart for this replica.

        Looks for the last k=0 production segment restart in run{run_idx-1}.
        Falls back to non-segmented restart if no segments found.

        Returns:
            Path to restart file, or None if not found.
        """
        import glob as glob_mod

        prev_run = run_idx - 1
        name = self.config.input_folder.name
        prev_res = self.config.input_folder / f"run{prev_run}" / "res"
        sim_type = "flat" if self.state.phase in (1, 2) else "prod"

        # Look for segmented restart files from k=0 of previous run
        seg_pattern = str(prev_res / f"{name}_{sim_type}.0.{replica_idx}.seg*.rst")
        seg_files = sorted(glob_mod.glob(seg_pattern))
        if seg_files:
            return Path(seg_files[-1])

        # Fallback: non-segmented restart
        non_seg = prev_res / f"{name}_{sim_type}.0.{replica_idx}.rst"
        if non_seg.exists():
            return non_seg

        return None

    def _is_wham_output_invalid(self, analysis_dir: Path, cut_params: dict | None = None) -> bool:
        """Check for invalid WHAM output (all-zero, NaN, or Inf)."""
        return BiasAnalyzer.is_output_invalid(analysis_dir, cut_params)

    def _cleanup_invalid_wham(self, analysis_dir: Path) -> None:
        """Remove invalid WHAM output files for retry."""
        BiasAnalyzer.cleanup_invalid(analysis_dir)

    # --- Adaptive cutoff helpers ---

    def _adapt_cutoffs(self, run_idx: int, xs_enabled: bool = False) -> dict:
        """Compute fixed staged cutoffs for Phase 1."""
        return self._bias_analyzer.compute_cutoffs(
            phase=1, run_idx=run_idx, coupling_scale=0.0,
            phase2_start_run=None, alf_info=self.state.alf_info,
            input_folder=self.config.input_folder,
            xs_enabled=xs_enabled,
        )

    def _phase2_cutoffs(self, run_idx: int, coupling_scale: float) -> dict:
        """Compute Phase 2 warmup cutoffs (log-space decay over 20 runs)."""
        return self._bias_analyzer.compute_cutoffs(
            phase=2, run_idx=run_idx, coupling_scale=coupling_scale,
            phase2_start_run=self.state.phase2_start_run,
            alf_info=self.state.alf_info,
            input_folder=self.config.input_folder,
        )

    def _phase3_cutoffs(self, run_idx: int, coupling_scale: float) -> dict:
        """Compute Phase 3 cutoffs (tight, with recovery for skewed populations)."""
        return self._bias_analyzer.compute_cutoffs(
            phase=3, run_idx=run_idx, coupling_scale=coupling_scale,
            phase2_start_run=self.state.phase2_start_run,
            alf_info=self.state.alf_info,
            input_folder=self.config.input_folder,
        )

    def _run_wham_with_retry(
        self,
        run_idx: int,
        nf: int,
        ms: int,
        msprof: int,
        cut_params: dict,
        max_attempts: int = 3,
    ) -> tuple[bool, str]:
        """Run analysis with retry on failure or invalid output."""
        return self._bias_analyzer.run_with_retry(
            run_idx=run_idx, nf=nf, ms=ms, msprof=msprof,
            cut_params=cut_params, alf_info=self.state.alf_info,
            phase=self.state.phase, max_attempts=max_attempts,
        )

    def _run_wham_analysis(
        self, nf: int, ms: int, msprof: int, cut_params: dict
    ) -> None:
        """Run WHAM analysis using bundled GPU library."""
        self._bias_analyzer._run_wham(nf, ms, msprof, cut_params, self.state.alf_info)

    def _invoke_lmalf(
        self,
        nf: int,
        ms: int,
        msprof: int,
        *,
        max_iter: int | None = None,
        tolerance: float | None = None,
    ) -> bool:
        """Run LMALF optimization."""
        return self._bias_analyzer._invoke_lmalf(
            nf, ms, msprof, self.state.alf_info,
            max_iter=max_iter, tolerance=tolerance,
        )

    def _run_lmalf_analysis(
        self, nf: int, ms: int, msprof: int, cut_params: dict
    ) -> None:
        """Run LMALF analysis using bundled GPU library."""
        self._bias_analyzer._run_lmalf(nf, ms, msprof, cut_params, self.state.alf_info)

    def _run_hybrid_analysis(
        self, nf: int, ms: int, msprof: int, cut_params: dict
    ) -> None:
        """Run WHAM followed by short LMALF refinement."""
        self._bias_analyzer._run_hybrid(
            nf, ms, msprof, cut_params, self.state.alf_info, self.state.phase,
        )

    def _generate_analysis_plots(
        self, run_idx: int, nsubs, msprof: int
    ) -> None:
        """Generate convergence and diagnostic plots after analysis."""
        self._convergence.generate_plots(
            run_idx, nsubs, msprof, self.config.input_folder,
        )

    def _detect_forced_initial_lambdas(
        self, run_idx: int, pop_strict: list, nsubs, trans_matrices,
    ) -> None:
        """Detect unsampled states and set biased Dirichlet initial lambdas."""
        self._convergence.detect_forced_lambdas(
            run_idx, pop_strict, nsubs, trans_matrices, self.config.input_folder,
        )

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

        When MPI is active (size > 1), all ranks participate in data preparation
        and WHAM computation. Only rank 0 handles file I/O and post-WHAM processing.
        """
        if self.state.alf_info is None:
            raise ValueError("alf_info not initialized")

        is_rank0 = (self.state.rank == 0)
        is_distributed = (self.state.size > 1 and self._comm is not None)
        home_path = os.getcwd()

        try:
            os.chdir(self.config.input_folder)

            # Analysis window (consistent on all ranks)
            if self.state.phase == 1:
                im5 = max(run_idx - 15, 1)
            else:
                im5 = max(run_idx - 5, 1)

            analysis_dir = Path(f"analysis{run_idx}")

            # === Rank 0: setup + lambda processing ===
            if is_rank0:
                analysis_dir.mkdir(exist_ok=True)

                # Copy previous analysis results
                prev_analysis = Path(f"analysis{run_idx - 1}")
                if prev_analysis.exists():
                    for fname in ["b_sum.dat", "c_sum.dat", "x_sum.dat", "s_sum.dat",
                                  "t_sum.dat", "u_sum.dat"]:
                        src = prev_analysis / fname
                        dst = analysis_dir / fname.replace("_sum", "_prev")
                        if src.exists():
                            shutil.copy(src, dst)

                # Process lambda files
                os.chdir(analysis_dir)
                self.state.alf_info["nreps"] = self.config.nreps
                (Path("data")).mkdir(exist_ok=True)

                name = self.config.input_folder.name

                import contextlib
                log_file = Path("analysis.log")

                with open(log_file, "w") as log_f:
                    with contextlib.redirect_stdout(log_f):
                        for j in range(self.config.nreps):
                            for kk in range(repeats):
                                sim_type = "flat" if self.state.phase in [1, 2] else "prod"

                                if self._exchanger is not None:
                                    # Segmented LMD files from replica exchange
                                    import glob as glob_mod

                                    seg_pattern = (
                                        f"../run{run_idx}/res/"
                                        f"{name}_{sim_type}.{kk}.{j}.seg*.lmd"
                                    )
                                    fnmsin = sorted(glob_mod.glob(seg_pattern))
                                else:
                                    fnmsin = [
                                        f"../run{run_idx}/res/"
                                        f"{name}_{sim_type}.{kk}.{j}.lmd"
                                    ]

                                fnmout = f"data/Lambda.{kk}.{j}.parquet"

                                if fnmsin and Path(fnmsin[0]).exists():
                                    convert_lambda_binary_to_parquet(
                                        self.state.alf_info, fnmout, fnmsin
                                    )
                                    # Delete source .lmd to save disk space
                                    for lmd_path in fnmsin:
                                        p = Path(lmd_path)
                                        if p.exists() and p.suffix == '.lmd':
                                            p.unlink()

                                    # For exchange mode: also clean up intermediate
                                    # segment restart files (keep only the final one)
                                    if self._exchanger is not None:
                                        seg_rst_pattern = (
                                            f"../run{run_idx}/res/"
                                            f"{name}_{sim_type}.{kk}.{j}.seg*.rst"
                                        )
                                        seg_rsts = sorted(glob_mod.glob(seg_rst_pattern))
                                        # Keep the last segment's restart, delete the rest
                                        for rst_p in seg_rsts[:-1]:
                                            Path(rst_p).unlink(missing_ok=True)

            # === Barrier: rank 0 done with file processing ===
            if is_distributed:
                self._comm.Barrier()
                if not is_rank0:
                    os.chdir(analysis_dir)

            # === All ranks: compute WHAM inputs (distributed cross-energy) ===
            nf = self._bias_analyzer.prepare_data(
                self.state.alf_info, run_idx, self.state.phase,
            )

            # === Rank 0: metadata, cutoffs, transitions ===
            nsubs = ms = msprof = None
            cut_params = {}

            if is_rank0:
                nsubs = self.state.alf_info["nsubs"]
                nblocks = self.state.alf_info["nblocks"]
                np.savetxt("nsubs", np.array(nsubs).reshape((1, -1)), fmt=" %d")
                np.savetxt("nblocks", np.array([nblocks]), fmt=" %d")

                ntersite = self.state.alf_info.get("ntersite", self._ntersite())
                ms, msprof = ntersite[0], ntersite[1]

                nbshift_src = Path("..") / "nbshift"
                nbshift_dst = Path("nbshift")
                if nbshift_src.exists() and not nbshift_dst.exists():
                    shutil.copytree(nbshift_src, nbshift_dst)

                max_nsubs = max(nsubs) if len(nsubs) > 0 else 2
                coupling_scale = (2.0 / max(max_nsubs, 2)) ** 0.5

                if self.state.phase == 1:
                    xs_ok = (self.state.xs_coverage_count
                             >= self.config.min_xs_coverage_runs)
                    cut_params = self._adapt_cutoffs(run_idx, xs_enabled=xs_ok)
                elif self.state.phase == 2:
                    cut_params = self._phase2_cutoffs(run_idx, coupling_scale)
                else:
                    cut_params = self._phase3_cutoffs(run_idx, coupling_scale)

                if self.state.phase != 1:
                    if self.config.no_b_bias:
                        cut_params["calc_phi"] = False
                    if self.config.no_c_bias:
                        cut_params["calc_psi"] = False
                    if self.config.no_x_bias:
                        cut_params["calc_chi"] = False
                    if self.config.no_s_bias:
                        cut_params["calc_omega"] = False
                    if self.config.no_t_bias:
                        cut_params["calc_omega2"] = False
                    if self.config.no_u_bias:
                        cut_params["calc_omega3"] = False

                # Compute transition counts
                from cphmd.core.transitions import (
                    compute_connectivity_metric,
                    compute_transition_matrix,
                    save_transition_matrix,
                    transition_matrix_to_coupling_weights,
                )
                data_dir = Path("data")
                pre_lambda_data, _ = load_lambda_data(data_dir)
                trans_matrices = None
                if pre_lambda_data is not None:
                    trans_matrices = compute_transition_matrix(
                        pre_lambda_data, nsubs
                    )
                    save_transition_matrix(trans_matrices, Path("transitions.dat"))

                    from cphmd.core.phase_switcher import _per_site_ranges
                    col_fracs = (pre_lambda_data > 0.8).mean(axis=0)
                    visited_per_site = []
                    for start, end in _per_site_ranges(nsubs):
                        visited_per_site.append(col_fracs[start:end] > 0.01)

                    # Phase 1 x/s coverage gate: track consecutive all-visited runs
                    if self.state.phase == 1 and run_idx >= 20:
                        all_visited = all(v.all() for v in visited_per_site)
                        if all_visited:
                            self.state.xs_coverage_count += 1
                        else:
                            self.state.xs_coverage_count = 0

                    connectivity, weak_pairs = compute_connectivity_metric(
                        trans_matrices, visited_per_site=visited_per_site,
                    )
                    print(f"  Transition connectivity: {connectivity:.2f} "
                          f"(min-pair transitions: {int(connectivity * 50)})")
                    if weak_pairs:
                        for site, si, sj in weak_pairs:
                            print(f"    Weakest: site {site + 1}, states {si + 1}<->{sj + 1}")
                    trans_weights = transition_matrix_to_coupling_weights(
                        trans_matrices, nsubs, ms=ms
                    )
                    cut_params["transition_weights"] = trans_weights
                    cut_params["connectivity"] = connectivity

            # === Broadcast WHAM parameters to all ranks ===
            if is_distributed:
                nf = self._comm.bcast(nf, root=0)
                ms = self._comm.bcast(ms, root=0)
                msprof = self._comm.bcast(msprof, root=0)
                cut_params = self._comm.bcast(cut_params, root=0)

            # === All ranks: WHAM with retry ===
            wham_success, wham_msg = self._run_wham_with_retry(
                run_idx=run_idx,
                nf=nf,
                ms=ms,
                msprof=msprof,
                cut_params=cut_params,
                max_attempts=3,
            )

            # Free analysis data to reduce memory between runs
            self._bias_analyzer._packed_D = None
            self._bias_analyzer._packed_sim_indices = None
            self._bias_analyzer._packed_frame_counts = None
            self._bias_analyzer._wham_lambda = None

            # === Non-rank-0 returns after WHAM ===
            if not is_rank0:
                return

            # === Rank 0: post-WHAM processing ===
            print(wham_msg)

            if not wham_success:
                print("All WHAM attempts failed, using zero bias updates")
                nblocks = self.state.alf_info["nblocks"]
                np.savetxt("b.dat", np.zeros((1, nblocks)), fmt=" %10.5f")
                np.savetxt("c.dat", np.zeros((nblocks, nblocks)), fmt=" %10.5f")
                np.savetxt("x.dat", np.zeros((nblocks, nblocks)), fmt=" %10.5f")
                np.savetxt("s.dat", np.zeros((nblocks, nblocks)), fmt=" %10.5f")

            # Update EWBS
            self._convergence.update_ewbs()

            # Compute populations
            lambda_data, pop_data, pop_strict = (
                self._convergence.compute_populations(nsubs)
            )

            if pop_strict is not None and len(pop_strict) > 0:
                self._convergence.detect_forced_lambdas(
                    run_idx, pop_strict, nsubs, trans_matrices,
                    self.config.input_folder,
                )
                np.savetxt("pop_strict.dat", np.array(pop_strict), fmt=" %.6f")

            if lambda_data is not None:
                self._convergence.generate_plots(
                    run_idx, nsubs, msprof, self.config.input_folder,
                )

            self._convergence.check_and_update_phase(
                run_idx, lambda_data, nsubs, cut_params, trans_matrices,
                self._regenerate_g_imp_if_needed,
            )

            np.savetxt("phase.dat", np.array([self.state.phase]), fmt="%d")

            self._convergence.check_stop(run_idx, lambda_data, nsubs, confirmation)

            if (self.config.generate_hh_plots and
                self.config.pH and
                self.config.nreps > 3 and
                self.state.patch_info is not None):
                from cphmd.analysis.henderson_hasselbalch import generate_hh_analysis

                from .cphmd_params import compute_all_site_parameters, get_delta_pKa_for_phase

                delta_pKa = get_delta_pKa_for_phase(self.state.phase)
                cphmd_params = compute_all_site_parameters(
                    self.state.patch_info,
                    self.config.temperature,
                )

                generate_hh_analysis(
                    run_idx=run_idx,
                    data_dir=Path("data"),
                    patch_info=self.state.patch_info,
                    pH=cphmd_params.effective_pH,
                    delta_pKa=delta_pKa,
                    nreps=self.config.nreps,
                    output_dir=Path(self.config.input_folder) / "plots",
                    ncentral=self.state.alf_info.get("ncentral", self.config.nreps // 2),
                )

            if not confirmation:
                set_vars_from_analysis_dir(
                    Path.cwd(), self.state.alf_info, step=run_idx + 1
                )

            keep_window = 15 if self.state.phase == 1 else 5
            if self.config.cleanup_old_analysis and run_idx > keep_window:
                old_analysis = Path(self.config.input_folder) / f"analysis{run_idx - keep_window}"
                if old_analysis.exists():
                    shutil.rmtree(old_analysis)
                    print(f"Cleaned up {old_analysis}")

            for subdir in ("Energy", "Lambda"):
                p = Path(subdir)
                if p.exists():
                    shutil.rmtree(p)

            print(f"ALF analysis complete for run {run_idx}")
            sys.stdout.flush()

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


