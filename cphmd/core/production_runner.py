"""Production CpHMD runner configuration.

ProductionConfig defines parameters for production CpHMD runs with fixed
converged biases (no ALF training loop).  It shares field conventions with
ALFConfig but strips out all training-specific knobs.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from cphmd.core import ElecType, PrepFormat, RestrainType, VdwType
from cphmd.core.replica_exchange import ReplicaExchangeConfig


def build_nsubsites_str(nsubs: list[int]) -> str:
    """Build isitemld-format nsubsites string for parquet metadata.

    Maps nsubs (per-site substate counts) to the block-to-site mapping array.
    Index 0 is the environment block, values are 1-based site indices.
    Example: nsubs=[3,2] -> "[0 1 1 1 2 2]"
    """
    isitemld = [0] + [i + 1 for i, n in enumerate(nsubs) for _ in range(n)]
    return "[" + " ".join(map(str, isitemld)) + "]"


def build_parquet_metadata(
    nsubs: list[int],
    temperature: float,
    nsavl: int,
    delta_t: float,
    npriv: int,
    actual_steps: int,
    pH: float,
    prod_id: int,
    name: str,
) -> dict[str, str]:
    """Build metadata dict for production parquet files.

    All values are strings (parquet schema metadata requirement).
    """
    nblocks = 1 + sum(nsubs)
    nsites = len(nsubs)
    time_step = delta_t * nsavl
    time_start = npriv * delta_t
    time_end = time_start + (actual_steps - 1) * time_step

    return {
        "Title": "CpHMD production",
        "Temperature": f"{temperature:.2f}",
        "Time Step": f"{time_step:.6f}".rstrip("0").rstrip("."),
        "Time Start": f"{time_start:.3f}",
        "Time End": f"{time_end:.3f}",
        "Save Frequency": str(nsavl),
        "nblocks": str(nblocks),
        "nsites": str(nsites),
        "nsubsites": build_nsubsites_str(nsubs),
        "Start Step": str(npriv),
        "Total Steps": str(actual_steps),
        "End Step": str(npriv + actual_steps - nsavl),
        "pH": str(pH),
        "Simulation": f"prod_{prod_id}",
        "Name": name,
    }


def find_resume_point(lambdas_dir: Path, n_chunks: int, nreps: int) -> int:
    """Scan for existing valid parquets and return first incomplete chunk.

    Validates each parquet with ``pq.read_metadata()``.  Invalid files are
    deleted so that a subsequent run can regenerate them cleanly.

    Returns
    -------
    int
        1-based chunk number to start/resume from.  Returns ``n_chunks + 1``
        when every chunk already has valid parquets for all replicas.
    """
    import pyarrow.parquet as pq

    for chunk in range(1, n_chunks + 1):
        all_valid = True
        for rep in range(nreps):
            parquet_path = lambdas_dir / f"{chunk:03d}_{rep:02d}.parquet"
            if not parquet_path.exists():
                all_valid = False
                break
            try:
                pq.read_metadata(str(parquet_path))
            except Exception:
                parquet_path.unlink(missing_ok=True)
                all_valid = False
                break
        if not all_valid:
            return chunk
    return n_chunks + 1


def find_restart_for_chunk(
    res_dir: Path,
    iteration: int,
    replica: int,
    exchange: bool,
) -> Path:
    """Find restart file from the previous chunk for continuation.

    Looks for ``prod.{prev:03d}.{replica:02d}.rst`` first.  When *exchange*
    is ``True`` and the normal restart is missing, falls back to the
    highest-numbered segment restart (``*.seg*.rst``).

    Raises
    ------
    FileNotFoundError
        If no suitable restart file exists.
    """
    prev = iteration - 1

    rst = res_dir / f"prod.{prev:03d}.{replica:02d}.rst"
    if rst.exists():
        return rst

    if exchange:
        import glob as glob_mod

        pattern = str(res_dir / f"prod.{prev:03d}.{replica:02d}.seg*.rst")
        seg_files = sorted(glob_mod.glob(pattern))
        if seg_files:
            return Path(seg_files[-1])

    raise FileNotFoundError(
        f"No restart file found for chunk {iteration}, replica {replica} "
        f"in {res_dir}. Expected prod.{prev:03d}.{replica:02d}.rst"
    )


@dataclass
class ProductionConfig:
    """Configuration for a production CpHMD simulation with fixed biases.

    At least one bias source must be provided: ``use_presets``,
    ``variables_dir``, or ``variables_files``.  When multiple sources are
    set, per-residue precedence is: files > dir > presets.

    Attributes:
        input_folder: System folder (contains ``prep/`` subdirectory).
        toppar_dir: Path to topology/parameter directory.
        prod_id: Production run identity; creates ``prod_{id}/``.
        seed: RNG seed for reproducibility (``None`` = derive from *prod_id*).
        ns: Total nanoseconds of production dynamics.
        ns_per_chunk: Nanoseconds per iteration chunk (last chunk handles remainder).
        temperature: Simulation temperature in Kelvin.
        pH_start: Starting (or only) pH value.
        pH_end: Ending pH value (ignored when *nreps* == 1).
        nreps: Number of pH replicas.
        use_presets: Use converged single-site preset biases.
        variables_dir: Directory containing ``var-{resname}.inp`` files.
        variables_files: Explicit per-residue variable file mapping.
        elec_type: Electrostatics method.
        vdw_type: VDW method (``None`` = auto-detect from prep format).
        hmr: Hydrogen mass repartitioning (``None`` = auto-detect).
        restrains: Restraint method for titratable atoms.
        scat_force_constant: SCAT restraint force constant.
        fnex: FNEX softmax constraint parameter.
        chi_offset: LDBV class 8 REF (s-term).
        omega_decay: LDBV class 10 REF (x-term, negative).
        lambda_mass: Lambda mass in amu*A^2 (``None`` = auto).
        lambda_fbeta: Lambda Langevin friction in ps^-1 (``None`` = auto).
        nsavc: Coordinate save frequency (frames).
        nsavl: Lambda save frequency.
        cutnb: Non-bonded list cutoff.
        ctofnb: Outer non-bonded cutoff.
        ctonnb: Inner non-bonded cutoff.
        gscale: Langevin friction coefficient (``None`` = auto).
        topology_files: Topology/parameter files relative to *toppar_dir*.
        extra_files: Additional topology/parameter files (absolute paths).
        replica_exchange: Replica exchange configuration (``None`` = disabled).
        prep_format: Prep format (``"default"``, ``"legacy"``, or ``"auto"``).
        cent_ncres: Number of residues for recentering (``False`` to disable).
        debug: Keep full CHARMM verbosity.
    """

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------
    input_folder: str | Path
    toppar_dir: str | Path = "toppar"

    # Production run identity
    prod_id: int = 1
    seed: int | None = None

    # Duration
    ns: float = 10.0
    ns_per_chunk: float = 1.0

    # Temperature
    temperature: float = 298.15

    # pH replicas
    pH_start: float = 7.0
    pH_end: float = 7.0
    nreps: int = 1

    # ------------------------------------------------------------------
    # Bias source (at least one required)
    # ------------------------------------------------------------------
    use_presets: bool = False
    variables_dir: str | Path | None = None
    variables_files: dict[str, str | Path] | None = None

    # ------------------------------------------------------------------
    # Electrostatics / nonbonded
    # ------------------------------------------------------------------
    elec_type: ElecType = "pmeex"
    vdw_type: VdwType | None = None
    hmr: bool | None = None
    restrains: RestrainType = "SCAT"
    scat_force_constant: float = 300.0
    fnex: float = 5.5

    # Bias shape constants
    chi_offset: float | None = None
    omega_decay: float | None = None

    # Lambda dynamics
    lambda_mass: float | None = None
    lambda_fbeta: float | None = None

    # Output frequencies
    nsavc: int = 500
    nsavl: int = 1

    # Nonbonded
    cutnb: float = 14.0
    ctofnb: float = 12.0
    ctonnb: float = 10.0
    gscale: float | None = None

    # Topology files (relative to toppar_dir)
    topology_files: list[str] = field(
        default_factory=lambda: [
            "top_all36_prot.rtf",
            "par_all36m_prot.prm",
            "top_all36_na.rtf",
            "par_all36_na.prm",
            "toppar_water_ions.str",
            "top_all36_cgenff.rtf",
            "par_all36_cgenff.prm",
            "my_files/titratable_residues.str",
            "my_files/nucleic_c36.str",
        ]
    )

    # Extra topology/parameter files (absolute paths for custom ligands)
    extra_files: list[str | Path] = field(default_factory=list)

    # Replica exchange
    replica_exchange: ReplicaExchangeConfig | None = None

    # Prep format
    prep_format: PrepFormat = "auto"

    # Recentering
    cent_ncres: int | bool = False

    # Debug
    debug: bool = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # --- Path conversions ---
        self.input_folder = Path(self.input_folder).resolve()
        self.toppar_dir = Path(self.toppar_dir)
        if self.variables_dir is not None:
            self.variables_dir = Path(self.variables_dir)

        # --- Prep format auto-detection ---
        prep_dir = self.input_folder / "prep"
        if self.prep_format == "auto":
            if (prep_dir / "patches.dat").exists():
                self.prep_format = "default"
            elif (prep_dir / "alf_info.py").exists():
                self.prep_format = "legacy"
            else:
                raise FileNotFoundError(
                    f"Cannot detect prep format in {prep_dir}: "
                    f"expected patches.dat (default) or alf_info.py (legacy)"
                )

        # --- Required files ---
        if self.prep_format == "default":
            for f in [
                "prep/system.psf",
                "prep/system.crd",
                "prep/patches.dat",
                "prep/box.dat",
                "prep/fft.dat",
            ]:
                path = self.input_folder / f
                if not path.exists():
                    raise FileNotFoundError(f"Required file not found: {path}")

        # --- Bias source validation ---
        has_bias = self.use_presets or self.variables_dir is not None or self.variables_files
        if not has_bias:
            raise ValueError(
                "At least one bias source must be set: "
                "use_presets=True, variables_dir, or variables_files"
            )

        # --- Preset requires explicit vdw_type ---
        if self.use_presets and self.vdw_type is None:
            raise ValueError(
                "Preset resolution requires explicit vdw_type " "(e.g., vdw_type='vswitch')"
            )

        # --- ns_per_chunk ---
        if self.ns_per_chunk <= 0:
            raise ValueError("ns_per_chunk must be positive")

        # --- Replica exchange requires 2+ replicas ---
        if self.replica_exchange is not None and self.replica_exchange.enabled and self.nreps < 2:
            raise ValueError(
                "Replica exchange requires 2 or more replicas " f"(nreps={self.nreps})"
            )

        # --- pH warnings ---
        if self.nreps > 1 and self.pH_start == self.pH_end:
            warnings.warn(
                f"nreps={self.nreps} but all replicas at same pH " f"({self.pH_start})",
                UserWarning,
                stacklevel=2,
            )
        if self.nreps == 1 and self.pH_start != self.pH_end:
            warnings.warn(
                f"pH_end ignored when nreps=1 " f"(pH_start={self.pH_start}, pH_end={self.pH_end})",
                UserWarning,
                stacklevel=2,
            )

        # --- Auto-detect sentinels from prep format ---
        is_legacy = self.prep_format == "legacy"
        if self.hmr is None:
            self.hmr = not is_legacy
        if self.vdw_type is None:
            self.vdw_type = "vfswitch" if is_legacy else "vswitch"
        if self.gscale is None:
            self.gscale = 0.1 if is_legacy else 10.0

        # Lambda mass/friction: HMR(4fs) → heavy/gentle; non-HMR(2fs) → lighter
        if self.lambda_mass is None:
            self.lambda_mass = 12.0 if self.hmr else 5.0
        if self.lambda_fbeta is None:
            self.lambda_fbeta = 5.0 if self.hmr else 7.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_chunks(self) -> int:
        """Number of iteration chunks (last chunk handles remainder)."""
        return math.ceil(self.ns / self.ns_per_chunk)

    @property
    def delta_pKa(self) -> float:
        """pH spacing between replicas (0.0 when single replica)."""
        if self.nreps > 1:
            return (self.pH_end - self.pH_start) / (self.nreps - 1)
        return 0.0

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    # Patch name → canonical residue type for variable resolution.
    # Known patches map to their standard residue name; unknown patches
    # fall back to progressively shorter prefixes then first-3-chars.
    _PATCH_TO_RESTYPE = {
        "ASP": "ASP",
        "ASH": "ASP",
        "ASPO": "ASP",
        "GLU": "GLU",
        "GLH": "GLU",
        "GLUO": "GLU",
        "HSP": "HSP",
        "HSD": "HSP",
        "HSE": "HSP",
        "HSPO": "HSP",
        "HSPD": "HSP",
        "HSPE": "HSP",
        "LYS": "LYS",
        "LYSO": "LYS",
        "LYSU": "LYS",
        "TYR": "TYR",
        "TYRO": "TYR",
        "TYRU": "TYR",
        "ARG": "ARG",
        "ARGO": "ARG",
        "ARU1": "ARG",
        "ARU2": "ARG",
        "SER": "SER",
        "SERO": "SER",
        "SERD": "SER",
        "CYS": "CYS",
        "CYSO": "CYS",
        "CYSD": "CYS",
        "THR": "THR",
        "THRO": "THR",
        "THRD": "THR",
        "NR": "NR",
        "NRED": "NR",
        "NREDO": "NR",
        "NRDU": "NR",
        "NRD2": "NR",
    }

    def get_seed_for_chunk(self, iteration: int) -> int:
        """Return a deterministic seed for the given chunk iteration.

        When ``seed`` is ``None``, derives from *prod_id* so that runs with
        different ``prod_id`` values get distinct seed sequences.
        """
        base = self.seed if self.seed is not None else self.prod_id * 10000
        return base + iteration

    def _resolve_preset_config(self) -> str | None:
        """Resolve preset configuration name from elec_type + vdw_type.

        Mapping: pmeex -> pme_ex, pmeon -> pme_on, pmenn -> pme_nn,
                 fshift -> fshift, fswitch -> fswitch; combined with vdw_type.

        Returns:
            Preset config name (e.g., "pme_ex_vswitch") or None if types unknown.
        """
        elec_map = {
            "pmeex": "pme_ex",
            "pmeon": "pme_on",
            "pmenn": "pme_nn",
            "fshift": "fshift",
            "fswitch": "fswitch",
        }
        elec = elec_map.get(self.elec_type)
        vdw = self.vdw_type
        if elec and vdw:
            return f"{elec}_{vdw}"
        return None

    def _patch_to_restype(self, patch_name: str) -> str:
        """Map a patch name to its canonical residue type.

        Tries exact match in ``_PATCH_TO_RESTYPE``, then progressively shorter
        prefixes, then falls back to stripping a trailing O/U/D suffix.
        """
        # Exact match
        restype = self._PATCH_TO_RESTYPE.get(patch_name)
        if restype is not None:
            return restype
        # Progressively shorter prefixes
        for length in range(len(patch_name) - 1, 1, -1):
            restype = self._PATCH_TO_RESTYPE.get(patch_name[:length])
            if restype is not None:
                return restype
        # Fallback: strip trailing protonation suffix (O/U/D) if present
        if len(patch_name) > 2 and patch_name[-1] in "OUD":
            return patch_name[:-1]
        # Last resort: first 3 chars
        return patch_name[:3] if len(patch_name) >= 3 else patch_name

    def _get_site_residue_types(self) -> dict[str, str]:
        """Extract canonical residue type for each titratable site.

        Reads patches.dat via ``_load_patches``, groups by site, and maps
        the first subsite's PATCH name to a canonical type.

        Returns:
            Dict mapping canonical residue type to itself (deduplicated).
            The keys are the unique residue types found across all sites.
        """
        from cphmd.core.generate_block import _load_patches

        df = _load_patches(self.input_folder)
        restypes: dict[str, str] = {}
        for site in sorted(df["site"].unique(), key=int):
            site_rows = df[df["site"] == site]
            patch_name = site_rows.iloc[0]["PATCH"]
            restype = self._patch_to_restype(patch_name)
            restypes[restype] = restype
        return restypes

    def _resolve_variables(self) -> dict[str, dict[str, float]]:
        """Resolve per-residue bias parameters from multiple sources.

        For each unique residue type found in patches.dat, looks up bias
        variables in precedence order:

        1. ``variables_files[restype]`` — explicit per-residue file path
        2. ``variables_dir / "var-{restype}.inp"`` — directory-based lookup
        3. ``use_presets`` — converged single-site preset biases
        4. None found — collected into missing list

        Raises:
            ValueError: If any residue types cannot be resolved from any source.

        Returns:
            Mapping of residue type to variable dict (e.g., {"lams1s1": 0.0, ...}).
        """
        import tempfile

        from cphmd.core.generate_block import _load_variables
        from cphmd.presets import list_presets, write_preset_variables

        site_types = self._get_site_residue_types()
        variables: dict[str, dict[str, float]] = {}
        missing: list[str] = []

        # Normalize variables_files keys to uppercase for matching
        files_map: dict[str, Path] = {}
        if self.variables_files:
            files_map = {k.upper(): Path(v) for k, v in self.variables_files.items()}

        preset_config = self._resolve_preset_config() if self.use_presets else None
        available_presets = list_presets(preset_config) if self.use_presets else []

        for restype in sorted(site_types):
            # 1. Explicit file override
            if restype in files_map:
                var_file = files_map[restype]
                variables[restype] = self._read_variable_file(var_file)
                continue

            # 2. Variables directory
            if self.variables_dir is not None:
                try:
                    variables[restype] = _load_variables(restype, self.variables_dir)
                    continue
                except FileNotFoundError:
                    pass

            # 3. Presets
            if self.use_presets and restype in available_presets:
                # Write preset to temp file, then read back as variable dict
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir) / f"var-{restype.lower()}.inp"
                    write_preset_variables(restype, str(tmp_path), config=preset_config)
                    variables[restype] = _load_variables(restype, Path(tmpdir))
                continue

            # 4. Not found
            missing.append(restype)

        if missing:
            raise ValueError(
                f"No bias variables found for residue types: {', '.join(sorted(missing))}. "
                f"Provide via variables_files, variables_dir, or enable use_presets."
            )

        return variables

    @staticmethod
    def _read_variable_file(path: Path) -> dict[str, float]:
        """Read a CHARMM-style variable file (``set varname = value``)."""
        variables: dict[str, float] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("set"):
                    parts = line.replace("set", "", 1).strip().split("=")
                    if len(parts) == 2:
                        var_name = parts[0].strip()
                        try:
                            var_value = float(parts[1].strip())
                        except ValueError:
                            continue
                        variables[var_name] = var_value
        return variables


class ProductionRunner:
    """Runs CpHMD production dynamics with fixed converged biases."""

    def __init__(self, config: ProductionConfig, comm=None):
        self.config = config
        self._comm = comm
        self._prod_dir = config.input_folder / f"prod_{config.prod_id}"
        self._blade_ready = False
        self._gpuid = 0  # Will be set from MPI rank or CUDA_VISIBLE_DEVICES
        self._resumed = False
        self._nsubs: list[int] | None = None  # Set during initialize from patches.dat

    def _create_prod_directory(self) -> None:
        """Create production directory with required subdirectories.

        Creates ``prod_{id}/`` under ``input_folder`` with subdirectories:
        ``dcd``, ``lambdas``, ``res``, ``logs``, ``prep``.
        Idempotent — safe to call multiple times.
        """
        for subdir in ("dcd", "lambdas", "res", "logs", "prep"):
            (self._prod_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _generate_production_files(self) -> None:
        """Generate block.str and restrains.str in the production prep directory.

        Resolves per-residue bias variables (from presets, directory, or explicit
        files), writes them to a temporary directory, copies ``patches.dat``
        into ``prod_{id}/prep/``, and calls ``generate_block_files`` to produce
        ``block.str`` and ``restrains.str``.
        """
        import shutil
        import tempfile

        from cphmd.core.generate_block import BlockGeneratorConfig, generate_block_files

        # 1. Resolve per-residue bias variable dicts
        variables = self.config._resolve_variables()

        # 2. Write variable files to a temp directory
        temp_dir = Path(tempfile.mkdtemp())
        try:
            for restype, var_dict in variables.items():
                var_file = temp_dir / f"var-{restype.lower()}.inp"
                lines = [f"* Variables for {restype}", "*", ""]
                for var_name, var_value in var_dict.items():
                    lines.append(f"set {var_name} = {var_value:10.3f}")
                lines.append("")
                var_file.write_text("\n".join(lines))

            # 3. Copy patches.dat into prod prep directory
            src_patches = self.config.input_folder / "prep" / "patches.dat"
            dst_patches = self._prod_dir / "prep" / "patches.dat"
            shutil.copy2(str(src_patches), str(dst_patches))

            # 4. Generate block.str and restrains.str
            block_config = BlockGeneratorConfig(
                input_folder=str(self._prod_dir),
                variables_dir=str(temp_dir),
                restrain_type=self.config.restrains,
                electrostatics=self.config.elec_type,
                temperature=self.config.temperature,
                pH=self.config.pH_start,
                lambda_mass=self.config.lambda_mass,
                lambda_fbeta=self.config.lambda_fbeta,
                chi_offset=self.config.chi_offset,
                omega_decay=self.config.omega_decay,
            )
            generate_block_files(block_config)
        finally:
            # 5. Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Set up for production: create directories, generate block files.

        Creates the ``prod_{id}/`` directory tree, generates ``block.str`` and
        ``restrains.str`` from resolved bias variables, and reads ``patches.dat``
        to determine per-site substate counts (``_nsubs``).
        """
        self._create_prod_directory()
        self._generate_production_files()

        # Extract nsubs from patches.dat
        from cphmd.core.generate_block import _load_patches

        df = _load_patches(self._prod_dir)
        self._nsubs = [len(g) for _, g in df.groupby("site", sort=True)]

    # ------------------------------------------------------------------
    # MPI replica assignment
    # ------------------------------------------------------------------

    def _get_replica_assignments(self) -> list[int]:
        """Return list of replica indices this MPI rank is responsible for.

        When running without MPI (``comm`` is ``None``), all replicas are
        assigned to this (single) process.  With MPI, replicas are distributed
        round-robin across ranks.
        """
        if self._comm is None:
            return list(range(self.config.nreps))
        rank = self._comm.Get_rank()
        size = self._comm.Get_size()
        return [r for r in range(self.config.nreps) if r % size == rank]

    # ------------------------------------------------------------------
    # Non-exchange dynamics
    # ------------------------------------------------------------------

    def _run_chunk(self, iteration: int, replica: int) -> tuple[Path, Path]:
        """Run one chunk of production dynamics (non-exchange).

        Executes BLaDE-accelerated Langevin dynamics for a single chunk,
        using the CHARMM dynamics engine via pyCHARMM.  On the first chunk
        SHAKE, BLaDE, and friction coefficients are initialized; subsequent
        chunks skip that setup.

        Parameters
        ----------
        iteration : int
            1-based chunk number.
        replica : int
            0-based replica index.

        Returns
        -------
        tuple[Path, Path]
            ``(rst_path, lmd_path)`` for the restart and lambda files written.
        """
        import numpy as np
        import pycharmm
        import pycharmm.dynamics as dyn
        import pycharmm.lingo as lingo
        import pycharmm.psf as psf
        import pycharmm.shake as shake

        from cphmd.utils.charmm_path import qpath

        config = self.config
        prod_dir = self._prod_dir
        res_dir = prod_dir / "res"
        dcd_dir = prod_dir / "dcd"

        # Compute step counts
        timestep = 0.004 if config.hmr else 0.002
        nsteps = int(config.ns_per_chunk * 1e6 / (timestep * 1e3))  # ns -> ps -> steps
        # Handle remainder for last chunk
        if iteration == config.n_chunks:
            total_steps = int(config.ns * 1e6 / (timestep * 1e3))
            completed_steps = nsteps * (iteration - 1)
            nsteps = total_steps - completed_steps

        nsavl = config.nsavl
        nsavc = config.nsavc

        rst_fn = res_dir / f"prod.{iteration:03d}.{replica:02d}.rst"
        lmd_fn = res_dir / f"prod.{iteration:03d}.{replica:02d}.lmd"
        dcd_fn = dcd_dir / f"prod.{iteration:03d}.{replica:02d}.dcd" if nsavc > 0 else None

        # File units
        rst_unit = 52
        lmd_unit = 53
        dcd_unit = 51
        rpr_unit = 54

        # Setup BLADE and shake on first chunk
        if not self._blade_ready:
            shake.on(fast=True, bonh=True, param=True, tol=1e-7)
            lingo.charmm_script("faster on")
            lingo.charmm_script(f"blade on gpuid {self._gpuid}")
            dyn.set_fbetas(np.full(psf.get_natom(), config.gscale, dtype=float))
            lingo.charmm_script("energy blade")
            self._blade_ready = True

        # Build dynamics parameters
        seed = config.get_seed_for_chunk(iteration)
        dyn_param: dict = {
            "blade": True,
            "prmc": True,
            "iprs": 100,
            "prdv": 100,
            "cpt": True,
            "timestep": timestep,
            "firstt": config.temperature,
            "finalt": config.temperature,
            "tstruc": config.temperature,
            "tbath": config.temperature,
            "ichecw": 0,
            "ihtfrq": 0,
            "ieqfrq": 0,
            "iasors": 1,
            "iasvel": 1,
            "iscvel": 0,
            "inbfrq": -1,
            "ilbfrq": 0,
            "imgfrq": -1,
            "ntrfrq": 0,
            "echeck": -1,
            "iunldm": lmd_unit,
            "iunwri": rst_unit,
            "iuncrd": dcd_unit if dcd_fn else -1,
            "nsavc": nsavc if dcd_fn else 0,
            "nsavl": nsavl,
            "nprint": min(nsteps, 50000),
            "iprfrq": min(nsteps, 50000),
            "nstep": nsteps,
            "isvfrq": nsteps,
            "iseed": seed,
            "pconstant": True,
            "pmass": psf.get_natom() * 0.12,
            "pref": 1.0,
            "pgamma": 20.0,
            "hoover": True,
            "reft": config.temperature,
            "tmass": 1000,
        }

        # Start vs restart
        rpr = None
        if iteration > 1 or self._resumed:
            dyn_param["start"] = False
            dyn_param["restart"] = True
            restart_file = find_restart_for_chunk(res_dir, iteration, replica, exchange=False)
            dyn_param["iunrea"] = rpr_unit
            rpr = pycharmm.CharmmFile(
                file_name=qpath(restart_file),
                file_unit=rpr_unit,
                read_only=True,
                formatted=True,
            )
        else:
            dyn_param["start"] = True
            dyn_param["restart"] = False

        # Open output files
        rst = pycharmm.CharmmFile(
            file_name=qpath(rst_fn),
            file_unit=rst_unit,
            read_only=False,
            formatted=True,
        )
        lmd = pycharmm.CharmmFile(
            file_name=qpath(lmd_fn),
            file_unit=lmd_unit,
            read_only=False,
            formatted=False,
        )
        dcd = None
        if dcd_fn:
            dcd = pycharmm.CharmmFile(
                file_name=qpath(dcd_fn),
                file_unit=dcd_unit,
                read_only=False,
                formatted=False,
            )

        pycharmm.DynamicsScript(**dyn_param).run()

        # Close files
        rst.close()
        lmd.close()
        if dcd:
            dcd.close()
        if rpr:
            rpr.close()

        return (rst_fn, lmd_fn)

    def _convert_lambda(self, iteration: int, replica: int) -> Path:
        """Convert ``.lmd`` binary to ``.parquet`` with production metadata.

        Reads the binary lambda file produced by ``_run_chunk``, writes a
        parquet file into the ``lambdas/`` directory with full provenance
        metadata, validates the written file, and deletes the source ``.lmd``.

        Parameters
        ----------
        iteration : int
            1-based chunk number.
        replica : int
            0-based replica index.

        Returns
        -------
        Path
            Path to the written parquet file.
        """
        import pyarrow.parquet as pq

        from cphmd.utils.lambda_io import read_lambda_binary, write_lambda_parquet

        res_dir = self._prod_dir / "res"
        lambdas_dir = self._prod_dir / "lambdas"

        lmd_fn = res_dir / f"prod.{iteration:03d}.{replica:02d}.lmd"
        parquet_fn = lambdas_dir / f"{iteration:03d}_{replica:02d}.parquet"

        data, meta = read_lambda_binary(lmd_fn)

        # Build metadata
        pH = self._get_replica_pH(replica)
        metadata = build_parquet_metadata(
            nsubs=self._nsubs,
            temperature=self.config.temperature,
            nsavl=meta.nsavl,
            delta_t=meta.delta_t,
            npriv=meta.npriv,
            actual_steps=len(data),
            pH=pH,
            prod_id=self.config.prod_id,
            name=self.config.input_folder.name,
        )

        write_lambda_parquet(parquet_fn, data, nsubs=self._nsubs, metadata=metadata)

        # Verify before deleting source
        pq.read_metadata(str(parquet_fn))
        lmd_fn.unlink()

        return parquet_fn

    def _get_replica_pH(self, replica: int) -> float:
        """Compute pH value for a given replica index.

        For a single replica, returns ``pH_start``.  For multiple replicas,
        linearly interpolates between ``pH_start`` and ``pH_end``.
        """
        if self.config.nreps == 1:
            return self.config.pH_start
        return self.config.pH_start + replica * self.config.delta_pKa
