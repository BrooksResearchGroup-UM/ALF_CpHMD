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
