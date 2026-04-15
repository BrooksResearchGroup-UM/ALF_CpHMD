"""
pyCHARMM utility functions for ALF/CpHMD simulations.

This module provides helper functions for common CHARMM operations
that are used throughout the CpHMD workflow:
- Topology/parameter file loading
- PSF/CRD structure reading
- Crystal/periodic boundary setup
- Non-bonded parameter configuration
- Selection definitions for titratable groups
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from cphmd.core import ElecType, VdwType
from cphmd.utils.charmm_path import qpath

# pyCHARMM imports are deferred to function bodies so that mpi4py can
# initialize MPI first (pyCHARMM also calls MPI_Init on import).


@dataclass
class BoxParameters:
    """Crystal box parameters.

    Attributes:
        crystal_type: Crystal system type (CUBIC, OCTAHEDRAL, etc.)
        dimensions: Box dimensions [A, B, C] in Angstroms
        angles: Box angles [alpha, beta, gamma] in degrees
    """
    crystal_type: str
    dimensions: list[float]
    angles: list[float] = field(default_factory=lambda: [90.0, 90.0, 90.0])

    @classmethod
    def from_file(cls, box_file: Path | str) -> "BoxParameters":
        """Load box parameters from box.dat file.

        File format:
            Line 1: Crystal type (e.g., OCTAHEDRAL)
            Line 2: A B C (dimensions)
            Line 3: alpha beta gamma (angles)
        """
        with open(box_file) as f:
            lines = f.readlines()

        crystal_type = lines[0].strip()
        dimensions = list(map(float, lines[1].strip().split()))
        angles = list(map(float, lines[2].strip().split()))

        return cls(crystal_type=crystal_type, dimensions=dimensions, angles=angles)


@dataclass
class FFTParameters:
    """FFT grid parameters for PME.

    Attributes:
        fftx, ffty, fftz: Grid dimensions
    """
    fftx: int
    ffty: int
    fftz: int

    @classmethod
    def from_file(cls, fft_file: Path | str) -> "FFTParameters":
        """Load FFT parameters from fft.dat file."""
        with open(fft_file) as f:
            values = f.read().strip().split()

        return cls(
            fftx=int(values[0]),
            ffty=int(values[1]),
            fftz=int(values[2]),
        )



@dataclass
class NonBondedConfig:
    """Non-bonded interaction configuration.

    Attributes:
        cutnb: Non-bonded cutoff distance
        cutim: Image cutoff distance
        ctofnb: Outer switching cutoff
        ctonnb: Inner switching cutoff
        elec_type: Electrostatics method (pmeex, pmeon, pmenn, fshift, fswitch)
        vdw_type: VDW method (vswitch, vfswitch)
        kappa: Ewald screening parameter
        order: PME interpolation order

    Electrostatics methods (performance relative to pmeex):
        - pmeex: PME with exclusions - baseline, best accuracy
        - pmeon: PME on - +0.5%, excellent
        - pmenn: PME no exclusions - +7%, good
        - fshift: Force shift - +20%, fair (no long-range)
        - fswitch: Force switch - +40%, poor (no long-range)

    VDW methods:
        - vswitch: Potential switch - baseline
        - vfswitch: Force switch - +4%
    """
    cutnb: float = 14.0
    cutim: float = 14.0
    ctofnb: float = 12.0
    ctonnb: float = 10.0
    elec_type: ElecType = "pmeex"
    vdw_type: VdwType = "vswitch"
    kappa: float = 0.320
    order: int = 6
    fftx: int | None = None
    ffty: int | None = None
    fftz: int | None = None

    # Legacy compatibility
    @property
    def use_pme(self) -> bool:
        return self.elec_type in ("pmeex", "pmeon", "pmenn")

    def to_dict(self) -> dict:
        """Convert to dictionary for NonBondedScript."""
        params = {
            "elec": True,
            "atom": True,
            "cdie": True,
            "eps": 1,
            "cutnb": self.cutnb,
            "cutim": self.cutim,
            "ctofnb": self.ctofnb,
            "ctonnb": self.ctonnb,
            "inbfrq": -1,
            "imgfrq": -1,
            "nbxmod": 5,
        }

        # VDW switching
        if self.vdw_type == "vswitch":
            params["vswitch"] = True
        else:  # vfswitch
            params["vfswitch"] = True

        # Electrostatics method
        if self.elec_type in ("pmeex", "pmeon", "pmenn"):
            # PME methods
            params.update({
                "switch": True,
                "ewald": True,
                "pmewald": True,
                "kappa": self.kappa,
                "order": self.order,
            })
            if self.fftx is not None:
                params["fftx"] = self.fftx
                params["ffty"] = self.ffty
                params["fftz"] = self.fftz
        elif self.elec_type == "fshift":
            # Force shift (no PME)
            params["fshift"] = True
        else:  # fswitch
            # Force switch (no PME) — fswitch alone; do NOT add switch
            # (CHARMM nbutil.F90:870 treats FSWI+SWIT as a conflict)
            params["fswitch"] = True

        return params


def read_topology_files(
    toppar_dir: Path | str,
    topology_files: list[str],
    verbose: bool = False,
) -> None:
    """Read CHARMM topology and parameter files.

    Args:
        toppar_dir: Directory containing topology files
        topology_files: List of file names relative to toppar_dir
        verbose: Whether to print verbose output
    """
    import pycharmm.lingo as lingo
    import pycharmm.read as read
    import pycharmm.settings as settings

    toppar_dir = Path(toppar_dir)

    if not verbose:
        lingo.charmm_script("prnlev -1")

    # Categorize files
    rtf_files = [f for f in topology_files if f.endswith(".rtf")]
    prm_files = [f for f in topology_files if f.endswith(".prm")]
    str_files = [f for f in topology_files if f.endswith(".str")]

    # Set permissive error handling
    lingo.charmm_script("bomblevel -2")
    settings.set_warn_level(-1)

    # Load RTF files
    if rtf_files:
        read.rtf(qpath(toppar_dir / rtf_files[0]))
        for f in rtf_files[1:]:
            read.rtf(qpath(toppar_dir / f), append=True)

    # Load PRM files
    if prm_files:
        read.prm(qpath(toppar_dir / prm_files[0]), flex=True)
        for f in prm_files[1:]:
            read.prm(qpath(toppar_dir / f), flex=True, append=True)

    # Stream STR files
    for f in str_files:
        lingo.charmm_script(f"stream {qpath(toppar_dir / f)}")

    # Restore settings
    settings.set_warn_level(5)
    lingo.charmm_script("bomblevel 0")
    if not verbose:
        lingo.charmm_script("prnlev 5")

    lingo.charmm_script("IOFOrmat EXTEnded")


def read_structure(
    psf_file: Path | str,
    crd_file: Path | str,
) -> None:
    """Read PSF and coordinate files.

    Args:
        psf_file: Path to PSF file
        crd_file: Path to CRD file
    """
    import pycharmm.read as read
    read.psf_card(qpath(psf_file))
    read.coor_card(qpath(crd_file))


def setup_crystal(
    box_params: BoxParameters,
    nb_config: NonBondedConfig,
    use_image_centering: bool = True,
) -> None:
    """Set up crystal/periodic boundary conditions.

    Args:
        box_params: Box parameters (type, dimensions, angles)
        nb_config: Non-bonded configuration
        use_image_centering: Apply image centering for solvent/ions
    """
    import pycharmm.crystal as crystal
    import pycharmm.lingo as lingo

    crystal_type = box_params.crystal_type.strip().upper()
    a, b, c = box_params.dimensions
    alpha, beta, gamma = box_params.angles

    crystal.free()
    if crystal_type in {"CUBIC", "CUBI"}:
        crystal.define_cubic(a)
    elif crystal_type in {"TETRAGONAL", "TETR"}:
        crystal.define_tetra(a, c)
    elif crystal_type in {"ORTHORHOMBIC", "ORTH"}:
        crystal.define_ortho(a, b, c)
    elif crystal_type in {"MONOCLINIC", "MONO"}:
        crystal.define_mono(a, b, c, beta)
    elif crystal_type in {"TRICLINIC", "TRIC"}:
        crystal.define_tri(a, b, c, alpha, beta, gamma)
    elif crystal_type in {"HEXAGONAL", "HEXA"}:
        crystal.define_hexa(a, c)
    elif crystal_type in {"RHOMBOHEDRAL", "RHOM"}:
        crystal.define_rhombo(a, alpha)
    elif crystal_type in {"OCTAHEDRAL", "OCTA"}:
        crystal.define_octa(a)
    elif crystal_type == "RHDO":
        crystal.define_rhdo(a)
    else:
        raise ValueError(f"Unsupported crystal type for CHARMM setup: {box_params.crystal_type!r}")

    # Build crystal images
    crystal.build(nb_config.cutim)

    # Set up image centering
    if use_image_centering:
        lingo.charmm_script("IMAGE BYRESid SELE segid SOLV .or. segid IONS END")
        lingo.charmm_script("IMAGE BYSEGid SELE .not. (segid SOLV .or. segid IONS) END")


def setup_nonbonded(nb_config: NonBondedConfig) -> None:
    """Configure non-bonded interactions.

    Args:
        nb_config: Non-bonded configuration
    """
    import pycharmm
    params = nb_config.to_dict()
    pycharmm.NonBondedScript(**params).run()


def define_selections(patch_info: pd.DataFrame) -> None:
    """Define CHARMM selections for titratable groups.

    Creates named selections based on the patches.dat information.
    Each selection identifies atoms in a specific protonation state.

    Args:
        patch_info: DataFrame from patches.dat with columns:
            SELECT, SEGID, RESID, PATCH, ATOMS
    """
    import pycharmm.lingo as lingo
    for _, row in patch_info.iterrows():
        name = row["SELECT"]
        segid = row["SEGID"]
        resid = row["RESID"]
        resname = row["PATCH"]

        # Build atom type selection
        atoms = row["ATOMS"].split()
        atom_clause = " -\n .or. type ".join(atoms)
        atom_selection = f"-\n(type {atom_clause})"

        # Create the selection
        cmd = (
            f"DEFine {name} SELEction "
            f"SEGID {segid} .AND. RESId {resid} .AND. RESName {resname} "
            f".AND. {atom_selection} END"
        )
        lingo.charmm_script(cmd)


def execute_block_command(block_cmd: str) -> None:
    """Execute a BLOCK command string.

    Args:
        block_cmd: Complete BLOCK command string
    """
    import pycharmm
    pycharmm.charmm_script(block_cmd)


def clear_block() -> None:
    """Clear existing BLOCK setup."""
    import pycharmm.lingo as lingo
    lingo.charmm_script("BLOCK\n CLEAR\n END")


def clear_crystal() -> None:
    """Free crystal setup."""
    import pycharmm.crystal as crystal

    crystal.free()


def clear_noe() -> None:
    """Reset NOE restraints."""
    import pycharmm.lingo as lingo
    lingo.charmm_script("NOE\n RESET\n END")


def reset_io_unit(unit: int = 91) -> None:
    """Reset Fortran I/O unit to clear EOF marker.

    After sequential write operations, the unit stays at EOF and subsequent
    reads fail with "Sequential READ not allowed after EOF marker".
    Force-reset by opening/closing with /dev/null.
    """
    import pycharmm.lingo as lingo
    lingo.charmm_script(f"open unit {unit} write form name /dev/null")
    lingo.charmm_script(f"close unit {unit}")


def get_natom() -> int:
    """Get number of atoms in PSF."""
    import pycharmm.psf as psf
    return psf.get_natom()


def show_energy() -> None:
    """Display current energy."""
    import pycharmm.energy as energy
    energy.show()


def setup_shake(fast: bool = True, bonh: bool = True, tol: float = 1e-7) -> None:
    """Enable SHAKE for hydrogen constraints.

    Args:
        fast: Use fast SHAKE algorithm
        bonh: Constrain bonds with hydrogens
        tol: Convergence tolerance
    """
    import pycharmm.shake as shake
    shake.on(fast=fast, bonh=bonh, param=True, tol=tol)


def get_gpu_id(rank: int) -> int:
    """Return the CUDA virtual device index for an MPI rank.

    CUDA remaps CUDA_VISIBLE_DEVICES to 0-based virtual indices, so
    ``blade on gpuid`` takes the virtual index = local_rank % n_visible.
    Detects local rank from MPI environment variables (OpenMPI, SLURM,
    Intel MPI, MPICH); falls back to global rank.
    """
    import os

    local_rank = None
    for env_var in [
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "MPI_LOCALRANKID",
        "PMI_LOCAL_RANK",
    ]:
        if env_var in os.environ:
            local_rank = int(os.environ[env_var])
            break

    if local_rank is None:
        local_rank = rank

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        n_visible = len([g for g in cuda_visible.split(",") if g.strip()])
        if n_visible > 0:
            return local_rank % n_visible

    return local_rank


def enable_blade(gpuid: int = 0) -> None:
    """Enable BLADE GPU acceleration.

    Args:
        gpuid: GPU device ID
    """
    import pycharmm.lingo as lingo
    lingo.charmm_script("faster on")
    lingo.charmm_script(f"blade on gpuid {gpuid}")


class CHARMMSession:
    """Context manager for CHARMM session setup.

    Handles initialization and cleanup of CHARMM state for ALF simulations.

    Example:
        >>> with CHARMMSession(toppar_dir, topology_files) as session:
        ...     session.read_structure(psf_file, crd_file)
        ...     session.setup_crystal(box_params, nb_config)
    """

    def __init__(
        self,
        toppar_dir: Path | str,
        topology_files: list[str],
        verbose: bool = False,
    ):
        self.toppar_dir = Path(toppar_dir)
        self.topology_files = topology_files
        self.verbose = verbose
        self._initialized = False

    def __enter__(self):
        read_topology_files(self.toppar_dir, self.topology_files, self.verbose)
        self._initialized = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass

    def read_structure(self, psf_file: Path | str, crd_file: Path | str) -> None:
        """Read PSF and coordinate files."""
        read_structure(psf_file, crd_file)

    def setup_crystal(
        self,
        box_params: BoxParameters,
        nb_config: NonBondedConfig,
        use_image_centering: bool = True,
    ) -> None:
        """Set up crystal and non-bonded interactions."""
        setup_crystal(box_params, nb_config, use_image_centering)
        setup_nonbonded(nb_config)

    def define_selections(self, patch_info: pd.DataFrame) -> None:
        """Define selections for titratable groups."""
        define_selections(patch_info)

    def execute_block(self, block_cmd: str) -> None:
        """Execute BLOCK command."""
        execute_block_command(block_cmd)
