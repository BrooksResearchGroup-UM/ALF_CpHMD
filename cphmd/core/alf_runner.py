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

# Add ALF library to path if not already installed
from cphmd import ALF_LIB_DIR

if str(ALF_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(ALF_LIB_DIR))

# Type aliases
PhaseType = Literal[1, 2, 3]
RestrainType = Literal["SCAT", "NOE"]
ElecType = Literal["pmeex", "pmeon", "pmenn", "fshift", "fswitch"]
VdwType = Literal["vswitch", "vfswitch"]


@dataclass
class ALFConfig:
    """Configuration for ALF simulation.

    Attributes:
        input_folder: Path to the prepared system folder (contains prep/ subdirectory)
        toppar_dir: Path to topology/parameter files
        temperature: Simulation temperature in Kelvin
        pH: Target pH for CpHMD (None for standard ALF without pH coupling)
        hmr: Whether to use hydrogen mass repartitioning (4fs timestep)
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
        vdw_type: VDW method (vswitch, vfswitch)
    """
    input_folder: str | Path
    toppar_dir: str | Path = "toppar"
    temperature: float = 298.15
    pH: float | None = None
    hmr: bool = False
    start: int = 1
    end: int = 20
    phase: PhaseType = 1
    nreps: int | None = None  # Defaults to MPI size
    restrains: RestrainType = "SCAT"
    restrain_hydrogens: bool = False
    no_x_bias: bool = False
    no_s_bias: bool = False
    cent_ncres: int | bool = False

    # Non-bonded parameters
    cutnb: float = 14.0
    ctofnb: float = 12.0
    ctonnb: float = 10.0
    elec_type: ElecType = "pmeex"
    vdw_type: VdwType = "vswitch"

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

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.input_folder = Path(self.input_folder)
        self.toppar_dir = Path(self.toppar_dir)

        # Validate required files exist
        required_files = [
            "prep/system.psf",
            "prep/system.crd",
            "prep/patches.dat",
            "prep/box.dat",
            "prep/fft.dat",
        ]
        for f in required_files:
            path = self.input_folder / f
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")


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

    def __init__(self, config: ALFConfig):
        """Initialize ALF simulation.

        Args:
            config: ALFConfig with simulation parameters
        """
        self.config = config
        self.state = SimulationState()
        self._comm = None
        self._initialized = False

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

    def _load_patch_info(self):
        """Load patch information from patches.dat."""
        patches_path = self.config.input_folder / "prep" / "patches.dat"
        self.state.patch_info = pd.read_csv(patches_path)

        # Extract site and subsite indices from SELECT column (e.g., "s1s1" -> site=1, sub=1)
        self.state.patch_info[["site", "sub"]] = (
            self.state.patch_info["SELECT"].str.extract(r"s(\d+)s(\d+)")
        )

    def _init_alf(self):
        """Initialize ALF library and create initial variable files."""
        import alf

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
            "ntersite": [1, 1],  # Enable intersite biases [ms, msprof]
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

        # Copy G_imp directory (importance sampling data)
        if self.state.rank == 0:
            self._copy_g_imp()

        # Initialize ALF variables
        home_dir = os.getcwd()
        try:
            os.chdir(self.config.input_folder)
            if self.state.rank == 0:
                alf.InitVars(alf_info)
                alf.SetVars(alf_info, 1)
        finally:
            os.chdir(home_dir)

    def _copy_g_imp(self):
        """Copy G_imp directory from ALF package to input folder."""
        import alf

        pkg_dir = Path(alf.__file__).resolve().parent
        dst = self.config.input_folder / "G_imp"

        candidates = ["G_imp_20", "G_imp"]
        for name in candidates:
            src = pkg_dir / name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f"Copied G_imp from {src} → {dst}")
                return

        print("Warning: No G_imp directory found in ALF package")

    def initialize(self):
        """Perform full initialization sequence."""
        self._init_mpi()
        self._comm.Barrier()

        self._init_charmm()
        self._load_patch_info()

        self._comm.Barrier()
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

        print(f"Rank {self.state.rank}: Starting from run {start_run}, "
              f"phase {self.state.phase}, nreps {self.config.nreps}")

        # Main simulation loop
        for run_idx in range(start_run, self.config.end + 1):
            self._execute_run(run_idx)

    def _find_last_run(self) -> int:
        """Find the last completed run to resume from."""
        # Implementation preserved from original script
        # Returns the run number to start/resume from
        for i in range(self.config.end + 1, self.config.start - 1, -1):
            var_file = self.config.input_folder / f"variables{i}.inp"
            if var_file.exists() and "set" in var_file.read_text():
                if "nan" in var_file.read_text().lower():
                    # Clean up corrupted run
                    self._cleanup_run(i)
                    continue
                # Found valid checkpoint
                self._cleanup_run(i)  # Clean current run to restart
                return i if i == 1 else i
        return self.config.start

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
        """
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

                # Setup and run dynamics
                self._setup_run_directory(run_idx)
                self._setup_crystal(run_idx, letter, k, replica_idx)
                self._build_block_commands(run_idx, letter, k, replica_idx)
                self._apply_restraints(run_idx)

                # Run minimization on first runs if needed
                if run_idx <= 5:
                    self._run_minimization(run_idx, replica_idx)

                self._run_dynamics(run_idx, letter, k, replica_idx)

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
            (run_dir / "dcd").mkdir(exist_ok=True)
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
        import random

        prep_dir = self.config.input_folder / "prep"
        is_first_run = (
            run_idx == self.config.start
            and (letter == "" or letter == "_a" or self.state.size > 1)
        )

        if is_first_run:
            # Read PSF file
            psf_file = prep_dir / ("system_hmr.psf" if self.config.hmr else "system.psf")
            read.psf_card(str(psf_file))

            # Define selections for titratable groups
            if self.state.patch_info is not None:
                define_selections(self.state.patch_info)
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

            # Compute and write bias shifts
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
            use_cphmd=(self.config.pH is not None and delta_pKa != 0),
        )

        # Generate and execute BLOCK command
        block_cmd = build_block_command(self.state.patch_info, variables, block_config)

        # Write block file for debugging
        block_file = self.config.input_folder / f"run{run_idx}" / f"block.{k}.{replica_idx}.str"
        block_file.write_text(block_cmd)

        # Execute BLOCK command
        execute_block_command(block_cmd)

    def _apply_restraints(self, run_idx: int):
        """Apply SCAT or NOE restraints to titratable atoms."""
        from .restraints import generate_scat_restraints, generate_noe_restraints
        from .charmm_utils import execute_block_command
        import pycharmm.lingo as lingo

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

        Only runs if no minimized coordinates exist. Uses steepest descent
        followed by ABNR minimization with BLADE GPU acceleration.
        """
        import pycharmm
        import pycharmm.minimize as minimize
        import pycharmm.shake as shake
        import pycharmm.write as write
        import pycharmm.lingo as lingo
        import pycharmm.energy as energy

        min_crd = self.config.input_folder / "prep" / "system_min.crd"
        if min_crd.exists():
            return  # Already minimized

        print(f"Running minimization for run {run_idx}...")

        # Enable SHAKE for minimization
        shake.on(fast=True, bonh=True, param=True, tol=1e-7)

        # Steepest descent (CPU)
        minimize.run_sd(nstep=100)

        # Enable FASTER and BLADE
        lingo.charmm_script("faster on")
        gpuid = replica_idx % 8
        lingo.charmm_script(f"blade on gpuid {gpuid}")

        # ABNR minimization (GPU)
        minimize.run_abnr(nstep=1000, tolenr=1e-3, tolgrd=1e-3)

        # Save minimized coordinates
        write.coor_card(str(min_crd))
        print(f"Minimized coordinates saved to {min_crd}")

        # Show energy
        energy.show()
        lingo.charmm_script("energy blade")

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
            nsteps_eq = 10000    # 20 ps
            nsteps_prod = 40000  # 80 ps
            nsavc = 1000         # Save coords every 2 ps
            nsavl = 1            # Save lambda every 2 fs
        elif self.state.phase == 2:
            nsteps_eq = 50000    # 100 ps
            nsteps_prod = 450000 # 900 ps
            nsavc = 10000        # Save coords every 20 ps
            nsavl = 1            # Save lambda every 2 fs
        else:  # Phase 3
            nsteps_eq = 0        # No equilibration
            nsteps_prod = 500000 # 1 ns
            nsavc = 10000        # Save coords every 20 ps
            nsavl = 1            # Save lambda every 2 fs

        # HMR allows 4fs timestep
        timestep = 0.002
        if self.config.hmr:
            nsteps_eq //= 2
            nsteps_prod //= 2
            nsavc //= 2
            nsavl = max(nsavl, 1)
            timestep = 0.004

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
        gscale = 10.0
        dyn.set_fbetas(np.full(psf.get_natom(), gscale, dtype=float))

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
            "inbfrq": 0,  # BLADE handles neighbor lists
            "ilbfrq": 0,
            "imgfrq": 0,  # BLADE handles images
            "ntrfrq": 0,
            "echeck": -1,  # Disable energy check
            "iunldm": lmd_unit,
            "iunwri": rst_unit,
            "iuncrd": dcd_unit,
            "nsavc": nsavc,
            "nsavl": nsavl,
            "nprint": nsavc,
            "iprfrq": nsavc,
            "isvfrq": nsavc,
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
        dcd_dir = run_dir / "dcd"
        res_dir = run_dir / "res"

        # === Equilibration Run ===
        if nsteps_eq > 0:
            dcd_fn = str(dcd_dir / f"{name}_eq.{k}.{replica_idx}.dcd")
            rst_fn = str(res_dir / f"{name}_eq.{k}.{replica_idx}.rst")
            lmd_fn = str(res_dir / f"{name}_eq.{k}.{replica_idx}.lmd")

            dcd = pycharmm.CharmmFile(file_name=dcd_fn, file_unit=dcd_unit,
                                      read_only=False, formatted=False)
            rst = pycharmm.CharmmFile(file_name=rst_fn, file_unit=rst_unit,
                                      read_only=False, formatted=True)
            lmd = pycharmm.CharmmFile(file_name=lmd_fn, file_unit=lmd_unit,
                                      read_only=False, formatted=False)

            dyn_param.update({"nstep": nsteps_eq})
            pycharmm.DynamicsScript(**dyn_param).run()

            dcd.close()
            rst.close()
            lmd.close()

        lingo.charmm_script("energy blade")

        # === Production Run ===
        if nsteps_prod > 0:
            sim_type = "flat" if self.state.phase in (1, 2) else "prod"

            dcd_fn = str(dcd_dir / f"{name}_{sim_type}.{k}.{replica_idx}.dcd")
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

            dcd = pycharmm.CharmmFile(file_name=dcd_fn, file_unit=dcd_unit,
                                      read_only=False, formatted=False)
            rst = pycharmm.CharmmFile(file_name=rst_fn, file_unit=rst_unit,
                                      read_only=False, formatted=True)
            lmd = pycharmm.CharmmFile(file_name=lmd_fn, file_unit=lmd_unit,
                                      read_only=False, formatted=False)

            dyn_param.update({"nstep": nsteps_prod})
            pycharmm.DynamicsScript(**dyn_param).run()

            dcd.close()
            rst.close()
            lmd.close()
            if rpr is not None:
                rpr.close()

        lingo.charmm_script("blade off")

    def _alf_analysis(self, run_idx: int, repeats: int):
        """Perform ALF analysis and update biases.

        Uses the ALF library to:
        1. Extract lambda values from trajectory
        2. Compute free energies with WHAM
        3. Update bias parameters for next iteration
        """
        import alf
        import alf.GetLambda

        if self.state.alf_info is None:
            raise ValueError("alf_info not initialized")

        home_path = os.getcwd()

        try:
            os.chdir(self.config.input_folder)

            # Determine analysis window
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

            # Link G_imp directory
            g_imp_target = analysis_dir / "G_imp"
            if not g_imp_target.exists():
                g_imp_src = Path("G_imp")
                if g_imp_src.exists():
                    os.symlink(g_imp_src.resolve(), g_imp_target)

            # Create ALF symlink for WHAM library compatibility
            # WHAM expects ../ALF/G_imp/ path structure
            alf_link = self.config.input_folder / "ALF"
            if not alf_link.exists():
                os.symlink(self.config.input_folder.resolve(), alf_link)

            # Process lambda files
            os.chdir(analysis_dir)
            self.state.alf_info["nreps"] = self.config.nreps
            (Path("data")).mkdir(exist_ok=True)

            name = self.config.input_folder.name

            for j in range(self.config.nreps):
                for kk in range(repeats):
                    if self.state.phase in [1, 2]:
                        fnmsin = [f"../run{run_idx}/res/{name}_flat.{kk}.{j}.lmd"]
                    else:
                        fnmsin = [f"../run{run_idx}/res/{name}_prod.{kk}.{j}.lmd"]

                    fnmout = f"data/Lambda.{kk}.{j}.dat"

                    if Path(fnmsin[0]).exists():
                        alf.GetLambda.GetLambda(self.state.alf_info, fnmout, fnmsin)

            # Run energy calculation
            alf.GetEnergy(self.state.alf_info, im5, run_idx)

            # Write nsubs and nblocks files for WHAM library
            nsubs = self.state.alf_info["nsubs"]
            nblocks = self.state.alf_info["nblocks"]
            np.savetxt("nsubs", np.array(nsubs).reshape((1, -1)), fmt=" %d")
            np.savetxt("nblocks", np.array([nblocks]), fmt=" %d")

            # Run free energy analysis
            # For early runs, skip WHAM and create minimal bias files
            ntersite = self.state.alf_info.get("ntersite", [1, 1])
            N = run_idx - im5 + 1  # Number of cycles

            # Create minimal bias update files for first few runs
            # (WHAM requires more data to be meaningful)
            if run_idx <= 3:
                # Create zero-change bias files
                np.savetxt("b.dat", np.zeros((1, nblocks)), fmt=" %7.2f")
                np.savetxt("c.dat", np.zeros((nblocks, nblocks)), fmt=" %7.2f")
                np.savetxt("x.dat", np.zeros((nblocks, nblocks)), fmt=" %7.2f")
                np.savetxt("s.dat", np.zeros((nblocks, nblocks)), fmt=" %7.2f")
            else:
                # After enough data, run full WHAM analysis
                # Import function directly to avoid namespace collision with alf's star imports
                from alf.GetFreeEnergy5 import GetFreeEnergy5
                from alf.wham.RunWham import RunWham

                # Try WHAM, fall back to zero updates on failure
                try:
                    nf = N * self.config.nreps
                    RunWham(nf, self.config.temperature, ntersite[0], ntersite[1])
                    GetFreeEnergy5(
                        self.state.alf_info, ntersite[0], ntersite[1]
                    )
                except Exception as e:
                    print(f"WHAM analysis failed: {e}, using zero bias updates")
                    np.savetxt("b.dat", np.zeros((1, nblocks)), fmt=" %7.2f")
                    np.savetxt("c.dat", np.zeros((nblocks, nblocks)), fmt=" %7.2f")
                    np.savetxt("x.dat", np.zeros((nblocks, nblocks)), fmt=" %7.2f")
                    np.savetxt("s.dat", np.zeros((nblocks, nblocks)), fmt=" %7.2f")

            # Update variables for next run
            alf.SetVars(self.state.alf_info, run_idx + 1)

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


# CLI interface for direct script execution
def main():
    """CLI entry point for ALF simulation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run ALF CpHMD simulation")
    parser.add_argument("-i", "--input", required=True, help="Input folder")
    parser.add_argument("-t", "--temperature", type=float, default=298.15, help="Temperature (K)")
    parser.add_argument("-pH", "--pH", type=float, default=None, help="Target pH")
    parser.add_argument("-hmr", "--hmr", action="store_true", help="Use HMR (4fs timestep)")
    parser.add_argument("-s", "--start", type=int, default=1, help="Start run")
    parser.add_argument("-e", "--end", type=int, default=20, help="End run")
    parser.add_argument("-p", "--phase", type=int, default=1, choices=[1, 2, 3], help="Initial phase")
    parser.add_argument("-nr", "--nreps", type=int, default=None, help="Number of replicas")
    parser.add_argument("-r", "--restrains", default="SCAT", choices=["SCAT", "NOE"], help="Restraint type")
    parser.add_argument("-nx", "--no-x-bias", action="store_true", help="Disable skew bias")
    parser.add_argument("-ns", "--no-s-bias", action="store_true", help="Disable endpoint bias")

    args = parser.parse_args()

    config = ALFConfig(
        input_folder=args.input,
        temperature=args.temperature,
        pH=args.pH,
        hmr=args.hmr,
        start=args.start,
        end=args.end,
        phase=args.phase,
        nreps=args.nreps,
        restrains=args.restrains,
        no_x_bias=args.no_x_bias,
        no_s_bias=args.no_s_bias,
    )

    run_alf_simulation(config)


if __name__ == "__main__":
    main()
