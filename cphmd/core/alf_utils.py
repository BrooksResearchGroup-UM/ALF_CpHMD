"""ALF (Adaptive Landscape Flattening) utility functions.

This module provides integrated versions of key ALF functions for use
with the cphmd package. WHAM analysis remains in the external ALF library.

Key Functions:
- init_vars(): Initialize analysis/0/ with zero biases
- set_vars(): Generate variables/{N}.py for pyCHARMM
- get_energy(): Compute bias energies from lambda trajectories

Directory Structure:
    work_dir/
    ├── prep/           # System files
    ├── runs/           # Dynamics runs (runs/1/, runs/2/, etc.)
    ├── analysis/       # ALF analysis (analysis/0/, analysis/1/, etc.)
    ├── variables/      # Variable files (variables/1.py, etc.)
    ├── nbshift/        # pH shift biases
    └── G_imp/          # Importance sampling data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _clean_negzero(arr: np.ndarray) -> np.ndarray:
    """Replace near-zero values with exact 0.0 to avoid -0.00000 in output."""
    arr = arr.copy()
    arr[np.abs(arr) < 5e-6] = 0.0
    return arr


def _fmtval(v: float) -> float:
    """Clean a single float: near-zero → 0.0."""
    return 0.0 if abs(v) < 5e-6 else v


from .bias_constants import CHI_OFFSET, OMEGA_DECAY


@dataclass
class ALFInfo:
    """ALF simulation information.

    Attributes:
        name: System name
        engine: MD engine (pycharmm, charmm, blade)
        nblocks: Total number of blocks (env + subsites)
        nsubs: Array of subsites per site
        nreps: Number of replicas
        ncentral: Central replica index (for replica exchange)
        temp: Temperature in Kelvin
        nnodes: Number of compute nodes
        ntersite: Terminal site indices for free energy calculation
        constraint_type: Implicit constraint type ("fnex" or "fpie")
        fnex: FNEX parameter value (when constraint_type="fnex")
        fpie_width: FPIE flat-bottom well width (when constraint_type="fpie")
        fpie_force: FPIE flat-bottom force constant (when constraint_type="fpie")
        g_imp_bins: Bin size for G_imp histograms (2D=N×N, 1D=N²)
    """
    name: str
    engine: str = "pycharmm"
    nblocks: int = 0
    nsubs: np.ndarray = None
    nreps: int = 1
    ncentral: int = 0
    temp: float = 298.15
    nnodes: int = 1
    ntersite: tuple[int, int] = (1, 1)
    # Constraint configuration
    constraint_type: str = "fnex"
    fnex: float = 5.5
    fpie_width: float = 1.0
    fpie_force: float = 100.0
    g_imp_bins: int = 32
    cutlsum: float = 0.8

    def __post_init__(self):
        if self.nsubs is None:
            self.nsubs = np.array([])
        # Normalize upstream aliases (e.g., "fnpwise" → "fpie")
        from cphmd.core.entropy import normalize_constraint_type

        self.constraint_type = normalize_constraint_type(self.constraint_type)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "engine": self.engine,
            "nblocks": int(self.nblocks),
            "nsubs": self.nsubs.tolist() if isinstance(self.nsubs, np.ndarray) else self.nsubs,
            "nreps": self.nreps,
            "ncentral": self.ncentral,
            "temp": self.temp,
            "nnodes": self.nnodes,
            "ntersite": list(self.ntersite),
            "constraint_type": self.constraint_type,
            "fnex": self.fnex,
            "fpie_width": self.fpie_width,
            "fpie_force": self.fpie_force,
            "g_imp_bins": self.g_imp_bins,
            "cutlsum": self.cutlsum,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ALFInfo":
        """Create ALFInfo from dictionary."""
        return cls(
            name=d["name"],
            engine=d.get("engine", "pycharmm"),
            nblocks=d["nblocks"],
            nsubs=np.array(d["nsubs"]),
            nreps=d["nreps"],
            ncentral=d["ncentral"],
            temp=d["temp"],
            nnodes=d.get("nnodes", 1),
            ntersite=tuple(d.get("ntersite", [1, 1])),
            constraint_type=d.get("constraint_type", "fnex"),
            fnex=d.get("fnex", 5.5),
            fpie_width=d.get("fpie_width", 1.0),
            fpie_force=d.get("fpie_force", 100.0),
            g_imp_bins=d.get("g_imp_bins", 32),
            cutlsum=d.get("cutlsum", 0.8),
        )


def ensure_alf_info(alf_info: ALFInfo | dict) -> ALFInfo:
    """Ensure alf_info is an ALFInfo dataclass.

    Args:
        alf_info: Either ALFInfo dataclass or dictionary.

    Returns:
        ALFInfo dataclass.
    """
    if isinstance(alf_info, dict):
        return ALFInfo.from_dict(alf_info)
    return alf_info


def _ensure_directories(work_dir: Path) -> dict[str, Path]:
    """Create standard ALF directory structure.

    Args:
        work_dir: Working directory for ALF simulation.

    Returns:
        Dictionary with paths to standard directories.
    """
    dirs = {
        "nbshift": work_dir / "nbshift",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def _load_preset_biases(
    alf_info: ALFInfo,
    preset_residue: str | None = None,
    preset_config: str | None = None,
    site_residue_types: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preset bias parameters and convert to arrays.

    Supports both single-site (one residue type) and multi-site (protein)
    systems. For multi-site, provide site_residue_types mapping each site
    to its residue type.

    Args:
        alf_info: ALF simulation information.
        preset_residue: Residue name for single-site presets. Ignored when
            site_residue_types is provided.
        preset_config: Preset configuration name (e.g., "pme_ex_vswitch_sca_nh").
            If None, uses default.
        site_residue_types: List of residue type names, one per site.
            E.g., ["ASP", "ASP", "GLU", "HSP", "LYS", "TYR"].
            Sites with unknown types (no preset available) get zero biases.

    Returns:
        Tuple of (b, c, x, s) arrays for bias parameters.
    """
    from cphmd.presets import get_preset_biases, list_presets

    nblocks = alf_info.nblocks
    nsubs = alf_info.nsubs
    nsites = len(nsubs)

    # Initialize arrays
    b = np.zeros([1, nblocks])
    c = np.zeros([nblocks, nblocks])
    x = np.zeros([nblocks, nblocks])
    s = np.zeros([nblocks, nblocks])

    # Compute subsite offsets: sub0[i] = starting block index for site i
    sub0 = np.cumsum(nsubs) - nsubs

    # Build list of residue types for each site
    if site_residue_types is not None:
        res_types = site_residue_types
    elif preset_residue:
        # Single residue type applied to all sites
        res_types = [preset_residue.upper()] * nsites
    elif alf_info.name:
        res_types = [alf_info.name.upper()] * nsites
    else:
        raise ValueError("preset_residue or site_residue_types must be specified")

    # Get available presets for this config
    available = set(list_presets(preset_config))

    # Fill intra-site biases from presets
    for si in range(nsites):
        restype = res_types[si].upper()
        if restype not in available:
            continue  # No preset for this residue type — stays at zero

        preset = get_preset_biases(restype, config=preset_config)
        ns = nsubs[si]
        off = sub0[si]

        # Linear biases (b/lam)
        lam = preset["lam"]
        for i in range(min(ns, len(lam))):
            b[0, off + i] = lam[i]

        # Intra-site coupling (c) — upper triangle, stored as list
        c_vals = preset["c"]
        idx = 0
        for i in range(ns):
            for j in range(i + 1, ns):
                if idx < len(c_vals):
                    c[off + i, off + j] = -c_vals[idx]
                    c[off + j, off + i] = -c_vals[idx]  # Symmetric
                    idx += 1

        # Intra-site omega (x) — off-diagonal elements row by row
        x_vals = preset["x"]
        idx = 0
        for i in range(ns):
            for j in range(ns):
                if i != j:
                    if idx < len(x_vals):
                        x[off + i, off + j] = -x_vals[idx]
                        idx += 1

        # Intra-site chi (s) — same layout as x
        s_vals = preset["s"]
        idx = 0
        for i in range(ns):
            for j in range(ns):
                if i != j:
                    if idx < len(s_vals):
                        s[off + i, off + j] = -s_vals[idx]
                        idx += 1

    return b, c, x, s


def init_vars(
    work_dir: str | Path,
    alf_info: ALFInfo | dict,
    minimize: bool = True,
    use_presets: bool = False,
    preset_residue: str | None = None,
    preset_config: str | None = None,
    site_residue_types: list[str] | None = None,
    b_init: np.ndarray | None = None,
    c_init: np.ndarray | None = None,
) -> Path:
    """Initialize ALF analysis directory with biases.

    Creates analysis0/ directory (flat, at work_dir level) with initial bias
    parameter files and generates variables1.inp for the first ALF cycle.

    Args:
        work_dir: Working directory for ALF simulation.
        alf_info: ALF simulation information.
        minimize: Whether to run minimization on first cycle.
        use_presets: If True, use preset biases instead of zeros.
        preset_residue: Residue name for presets (ASP, GLU, etc.).
            If None and use_presets=True, uses alf_info.name.
        preset_config: Preset configuration name (e.g., "pme_ex_vswitch_sca_nh").
            If None, uses default.
        site_residue_types: For multi-site systems, list of residue type
            names (one per site). Overrides preset_residue.
        b_init: Initial linear biases with shape (1, nblocks). If provided
            (and use_presets=False), written to b_prev.dat instead of zeros.
        c_init: Initial quadratic barriers with shape (nblocks, nblocks).
            If provided (and use_presets=False), written to c_prev.dat.

    Returns:
        Path to analysis0/ directory.
    """
    work_dir = Path(work_dir)
    alf_info = ensure_alf_info(alf_info)
    nblocks = alf_info.nblocks

    # Ensure nbshift directory exists
    dirs = _ensure_directories(work_dir)

    # Create initial bias arrays
    if use_presets:
        b, c, x, s = _load_preset_biases(
            alf_info, preset_residue, preset_config, site_residue_types
        )
    elif b_init is not None or c_init is not None:
        b = b_init if b_init is not None else np.zeros([1, nblocks])
        c = c_init if c_init is not None else np.zeros([nblocks, nblocks])
        x = np.zeros([nblocks, nblocks])
        s = np.zeros([nblocks, nblocks])
    else:
        # All zeros (legacy behavior)
        b = np.zeros([1, nblocks])
        c = np.zeros([nblocks, nblocks])
        x = np.zeros([nblocks, nblocks])
        s = np.zeros([nblocks, nblocks])

    # Create analysis0 directory (flat, matching external ALF convention)
    analysis0 = work_dir / "analysis0"
    analysis0.mkdir(exist_ok=True)

    # Write initial bias files
    np.savetxt(analysis0 / "b_prev.dat", b)
    np.savetxt(analysis0 / "b.dat", np.zeros_like(b))  # Current cycle starts at 0
    np.savetxt(analysis0 / "c_prev.dat", c)
    np.savetxt(analysis0 / "c.dat", np.zeros_like(c))
    np.savetxt(analysis0 / "x_prev.dat", x)
    np.savetxt(analysis0 / "x.dat", np.zeros_like(x))
    np.savetxt(analysis0 / "s_prev.dat", s)
    np.savetxt(analysis0 / "s.dat", np.zeros_like(s))

    # t/u terms (bcxstu) — always create zero files for compatibility
    t = np.zeros([nblocks, nblocks])
    u = np.zeros([nblocks, nblocks])
    np.savetxt(analysis0 / "t_prev.dat", t)
    np.savetxt(analysis0 / "t.dat", np.zeros_like(t))
    np.savetxt(analysis0 / "u_prev.dat", u)
    np.savetxt(analysis0 / "u.dat", np.zeros_like(u))

    # Create nbshift directory for replica exchange (if not exists)
    nbshift = dirs["nbshift"]
    if not (nbshift / "b_shift.dat").exists():
        np.savetxt(nbshift / "b_shift.dat", b)
        np.savetxt(nbshift / "c_shift.dat", c)
        np.savetxt(nbshift / "x_shift.dat", x)
        np.savetxt(nbshift / "s_shift.dat", s)

    # Generate variables1.inp from inside analysis0 (matching external ALF convention)
    set_vars_from_analysis_dir(analysis0, alf_info, step=1, minimize=minimize)

    return analysis0


def set_vars_from_analysis_dir(
    analysis_dir: str | Path,
    alf_info: ALFInfo | dict,
    step: int,
    minimize: bool = False,
) -> Path:
    """Generate variables file from inside an analysis directory.

    This matches the external ALF SetVars convention: called from inside
    an analysis directory (e.g., analysis0/ or analysis5/), reads bias
    files from the current directory, and writes variables{step}.inp
    one level up (i.e., at the work_dir level).

    The engine type in alf_info determines the output format:
    - "charmm"/"bladelib": CHARMM .inp format with 'set' commands
    - "pycharmm": Python .py format with yaml bias dictionary

    Args:
        analysis_dir: Path to analysis directory (e.g., work_dir/analysis5/).
        alf_info: ALF simulation information.
        step: ALF cycle number for which biases are being written.
        minimize: Whether to run minimization this cycle.

    Returns:
        Path to generated variables file.
    """
    analysis_dir = Path(analysis_dir)
    alf_info = ensure_alf_info(alf_info)
    work_dir = analysis_dir.parent

    nblocks = alf_info.nblocks
    nsubs = alf_info.nsubs
    nreps = alf_info.nreps
    ncentral = alf_info.ncentral
    name = alf_info.name
    nnodes = alf_info.nnodes
    temp = alf_info.temp
    engine = alf_info.engine

    # Load and accumulate bias parameters from analysis dir
    b_prev = np.loadtxt(analysis_dir / "b_prev.dat")
    b = np.loadtxt(analysis_dir / "b.dat")
    b_sum = b_prev + b
    b_sum = np.reshape(b_sum, (1, -1))
    np.savetxt(analysis_dir / "b_sum.dat", _clean_negzero(b_sum), fmt=" %10.5f")

    c_prev = np.loadtxt(analysis_dir / "c_prev.dat")
    c = np.loadtxt(analysis_dir / "c.dat")
    c_sum = c_prev + c
    np.savetxt(analysis_dir / "c_sum.dat", _clean_negzero(c_sum), fmt=" %10.5f")

    x_prev = np.loadtxt(analysis_dir / "x_prev.dat")
    x = np.loadtxt(analysis_dir / "x.dat")
    x_sum = x_prev + x
    np.savetxt(analysis_dir / "x_sum.dat", _clean_negzero(x_sum), fmt=" %10.5f")

    s_prev = np.loadtxt(analysis_dir / "s_prev.dat")
    s = np.loadtxt(analysis_dir / "s.dat")
    s_sum = s_prev + s
    np.savetxt(analysis_dir / "s_sum.dat", _clean_negzero(s_sum), fmt=" %10.5f")

    # t/u terms (bcxstu) — accumulate if files exist
    t_prev_path = analysis_dir / "t_prev.dat"
    t_path = analysis_dir / "t.dat"
    if t_prev_path.exists() and t_path.exists():
        t_prev = np.loadtxt(t_prev_path)
        t_data = np.loadtxt(t_path)
        t_sum = t_prev + t_data
        np.savetxt(analysis_dir / "t_sum.dat", _clean_negzero(t_sum), fmt=" %10.5f")
    else:
        t_sum = np.zeros((nblocks, nblocks))

    u_prev_path = analysis_dir / "u_prev.dat"
    u_path = analysis_dir / "u.dat"
    if u_prev_path.exists() and u_path.exists():
        u_prev = np.loadtxt(u_prev_path)
        u_data = np.loadtxt(u_path)
        u_sum = u_prev + u_data
        np.savetxt(analysis_dir / "u_sum.dat", _clean_negzero(u_sum), fmt=" %10.5f")
    else:
        u_sum = np.zeros((nblocks, nblocks))

    sub0 = np.cumsum(nsubs) - nsubs

    if engine == "pycharmm":
        # Write Python format: variables{step}.py
        output_file = work_dir / f"variables{step}.py"
        bias = {}

        # Linear biases
        for i in range(len(nsubs)):
            for j in range(nsubs[i]):
                key = f"lams{i+1}s{j+1}"
                bias[key] = float(b_sum[0, sub0[i] + j])

        # Quadratic psi coupling
        for si in range(len(nsubs)):
            for sj in range(si, len(nsubs)):
                for i in range(nsubs[si]):
                    j0 = (i + 1 if si == sj else 0)
                    for j in range(j0, nsubs[sj]):
                        key = f"cs{si+1}s{i+1}s{sj+1}s{j+1}"
                        bias[key] = -float(c_sum[sub0[si] + i, sub0[sj] + j])

        # Omega (x) coupling
        for si in range(len(nsubs)):
            for sj in range(len(nsubs)):
                for i in range(nsubs[si]):
                    for j in range(nsubs[sj]):
                        if sub0[si] + i != sub0[sj] + j:
                            key = f"xs{si+1}s{i+1}s{sj+1}s{j+1}"
                            bias[key] = -float(x_sum[sub0[si] + i, sub0[sj] + j])

        # Chi (s) coupling
        for si in range(len(nsubs)):
            for sj in range(len(nsubs)):
                for i in range(nsubs[si]):
                    for j in range(nsubs[sj]):
                        if sub0[si] + i != sub0[sj] + j:
                            key = f"ss{si+1}s{i+1}s{sj+1}s{j+1}"
                            bias[key] = -float(s_sum[sub0[si] + i, sub0[sj] + j])

        # t-term coupling (bcxstu)
        for si in range(len(nsubs)):
            for sj in range(len(nsubs)):
                for i in range(nsubs[si]):
                    for j in range(nsubs[sj]):
                        if sub0[si] + i != sub0[sj] + j:
                            key = f"ts{si+1}s{i+1}s{sj+1}s{j+1}"
                            bias[key] = float(t_sum[sub0[si] + i, sub0[sj] + j])

        # u-term coupling (bcxstu)
        for si in range(len(nsubs)):
            for sj in range(len(nsubs)):
                for i in range(nsubs[si]):
                    for j in range(nsubs[sj]):
                        if sub0[si] + i != sub0[sj] + j:
                            key = f"us{si+1}s{i+1}s{sj+1}s{j+1}"
                            bias[key] = -float(u_sum[sub0[si] + i, sub0[sj] + j])

        # Store full arrays
        bias["b"] = b_sum.tolist()
        bias["c"] = c_sum.tolist()
        bias["x"] = x_sum.tolist()
        bias["s"] = s_sum.tolist()
        bias["t"] = t_sum.tolist()
        bias["u"] = u_sum.tolist()

        with open(output_file, "w") as fp:
            fp.write("import yaml\n")
            fp.write("import numpy as np\n\n")

            fp.write('bias_string="""\n')
            yaml.dump(bias, fp)
            fp.write('"""\n')
            fp.write("bias=yaml.load(bias_string,Loader=yaml.Loader)\n")
            fp.write("bias['b']=np.array(bias['b'])\n")
            fp.write("bias['c']=np.array(bias['c'])\n")
            fp.write("bias['x']=np.array(bias['x'])\n")
            fp.write("bias['s']=np.array(bias['s'])\n")
            fp.write("bias['t']=np.array(bias['t'])\n")
            fp.write("bias['u']=np.array(bias['u'])\n\n")

            # Write alf_info
            alf_info_dict = alf_info.to_dict()
            fp.write('alf_info_string="""\n')
            yaml.dump(alf_info_dict, fp)
            fp.write('"""\n')
            fp.write("alf_info=yaml.load(alf_info_string,Loader=yaml.Loader)\n")
            fp.write("alf_info['nsubs']=np.array(alf_info['nsubs'])\n\n")

            fp.write(f"sysname='{name}'\n")
            fp.write(f"nnodes={nnodes}\n")
            fp.write(f"nreps={nreps}\n")
            fp.write(f"ncentral={ncentral}\n")
            fp.write(f"nblocks={nblocks}\n")
            fp.write(f"nsites={len(nsubs)}\n")
            fp.write(f"nsubs={nsubs.tolist()}\n")
            fp.write("nsubs=np.array(nsubs)\n")
            for i in range(len(nsubs)):
                fp.write(f"nsubs{i+1}={nsubs[i]}\n")
            fp.write(f"temp={temp}\n\n")
            fp.write(f"minimizeflag={minimize}\n")

    else:
        # Write CHARMM .inp format: variables{step}.inp
        output_file = work_dir / f"variables{step}.inp"

        with open(output_file, "w") as fp:
            fp.write(f"* Variables from step {step} of ALF\n")
            fp.write("*\n\n")

            ibuff = 0
            for i in range(len(nsubs)):
                for j in range(nsubs[i]):
                    fp.write(f"set lams{i+1}s{j+1} = {_fmtval(b_sum[0, ibuff+j]):12.5f}\n")
                ibuff += nsubs[i]

            ibuff = 0
            for si in range(len(nsubs)):
                jbuff = ibuff
                for sj in range(si, len(nsubs)):
                    for i in range(nsubs[si]):
                        j0 = i + 1 if si == sj else 0
                        for j in range(j0, nsubs[sj]):
                            fp.write(f"set cs{si+1}s{i+1}s{sj+1}s{j+1} = "
                                     f"{_fmtval(-c_sum[ibuff+i, jbuff+j]):12.5f}\n")
                    jbuff += nsubs[sj]
                ibuff += nsubs[si]

            ibuff = 0
            for si in range(len(nsubs)):
                jbuff = 0
                for sj in range(len(nsubs)):
                    for i in range(nsubs[si]):
                        for j in range(nsubs[sj]):
                            if ibuff + i != jbuff + j:
                                fp.write(f"set xs{si+1}s{i+1}s{sj+1}s{j+1} = "
                                         f"{_fmtval(-x_sum[ibuff+i, jbuff+j]):12.5f}\n")
                    jbuff += nsubs[sj]
                ibuff += nsubs[si]

            ibuff = 0
            for si in range(len(nsubs)):
                jbuff = 0
                for sj in range(len(nsubs)):
                    for i in range(nsubs[si]):
                        for j in range(nsubs[sj]):
                            if ibuff + i != jbuff + j:
                                fp.write(f"set ss{si+1}s{i+1}s{sj+1}s{j+1} = "
                                         f"{_fmtval(-s_sum[ibuff+i, jbuff+j]):12.5f}\n")
                    jbuff += nsubs[sj]
                ibuff += nsubs[si]

            fp.write(f'set sysname = "{name}\n')
            fp.write("trim sysname from 2\n")
            fp.write(f"set nnodes = {nnodes}\n")
            fp.write(f"set nreps = {nreps}\n")
            fp.write(f"set ncentral = {ncentral}\n")
            fp.write(f"set nblocks = {nblocks}\n")
            fp.write(f"set nsites = {len(nsubs)}\n")
            for i in range(len(nsubs)):
                fp.write(f"set nsubs{i+1} = {nsubs[i]}\n")
            fp.write(f"set temp = {temp}\n")
            fp.write(f"set minimizeflag = {int(minimize)}\n\n")

    return output_file


def _load_shift_file(
    analysis_dir: str | Path, filename: str, default_value: float = 0.0,
) -> np.ndarray | float:
    """Load shift file from analysis_dir/nbshift or ../nbshift fallback."""
    local_path = Path(analysis_dir) / "nbshift" / filename
    if local_path.exists():
        return np.loadtxt(local_path)
    fallback_path = Path("../nbshift") / filename
    if fallback_path.exists():
        return np.loadtxt(fallback_path)
    return default_value


def _load_simulation_data(
    alf_info: ALFInfo,
    start_cycle: int,
    end_cycle: int,
    skipE: int = 1,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[tuple[int, int, int]],
]:
    """Load lambda trajectories and bias parameters from analysis directories.

    Must be called from inside an analysis directory (e.g., analysis5/).

    Returns:
        Tuple of (Lambda, b_list, c_list, x_list, s_list, jk_map).
    """
    from cphmd.utils.lambda_io import find_lambda_files, read_lambda_values

    ncentral = alf_info.ncentral
    NF = end_cycle - start_cycle + 1

    Lambda: list[np.ndarray] = []
    b_list: list[np.ndarray] = []
    c_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    s_list: list[np.ndarray] = []
    jk_map: list[tuple[int, int, int]] = []

    for i in range(NF):
        analysis_dir = Path(f"../analysis{start_cycle + i}")
        data_dir = analysis_dir / "data"

        if not data_dir.is_dir():
            print(f"Warning: Directory {data_dir} not found")
            continue

        b_shift = _load_shift_file(analysis_dir, "b_shift.dat")
        c_shift = _load_shift_file(analysis_dir, "c_shift.dat")
        x_shift = _load_shift_file(analysis_dir, "x_shift.dat")
        s_shift = _load_shift_file(analysis_dir, "s_shift.dat")
        b_fix_shift = _load_shift_file(analysis_dir, "b_fix_shift.dat", 0.0)
        c_fix_shift = _load_shift_file(analysis_dir, "c_fix_shift.dat", 0.0)
        x_fix_shift = _load_shift_file(analysis_dir, "x_fix_shift.dat", 0.0)
        s_fix_shift = _load_shift_file(analysis_dir, "s_fix_shift.dat", 0.0)

        lambda_fpaths = find_lambda_files(data_dir)

        for fpath in lambda_fpaths:
            try:
                Lambda.append(read_lambda_values(fpath)[(skipE - 1)::skipE, :])
                j, k = map(int, fpath.name.split(".")[1:3])
                jk_map.append((j, k, i))
                b_old = np.loadtxt(analysis_dir / "b_prev.dat")
                b_list.append(b_old + b_shift * (k - ncentral) + b_fix_shift)
                c_old = np.loadtxt(analysis_dir / "c_prev.dat")
                c_list.append(c_old + c_shift * (k - ncentral) + c_fix_shift)
                x_old = np.loadtxt(analysis_dir / "x_prev.dat")
                x_list.append(x_old + x_shift * (k - ncentral) + x_fix_shift)
                s_old = np.loadtxt(analysis_dir / "s_prev.dat")
                s_list.append(s_old + s_shift * (k - ncentral) + s_fix_shift)
            except (OSError, ValueError) as e:
                logger.warning("Error loading file %s: %s", fpath, e)

    return Lambda, b_list, c_list, x_list, s_list, jk_map


def _load_bias_params(
    alf_info: ALFInfo,
    start_cycle: int,
    end_cycle: int,
    skipE: int = 1,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[tuple[int, int, int]],
    list[Path],
    int,
]:
    """Load bias parameters and record lambda file paths (no lambda data loaded).

    Same directory traversal as _load_simulation_data, but instead of loading
    lambda trajectories into memory, records their file paths and returns them.
    This enables streaming: load one lambda file at a time during packing.

    Must be called from inside an analysis directory (e.g., analysis5/).

    Returns:
        Tuple of (b_list, c_list, x_list, s_list, jk_map, lambda_files, skipE):
        - b_list: linear bias vectors, one per simulation
        - c_list: quadratic coupling matrices, one per simulation
        - x_list: omega coupling matrices, one per simulation
        - s_list: chi coupling matrices, one per simulation
        - jk_map: (j, k, analysis_idx) tuples for each simulation
        - lambda_files: Path objects for each lambda file (same order as bias lists)
        - skipE: passthrough of the subsample interval
    """
    from cphmd.utils.lambda_io import find_lambda_files

    ncentral = alf_info.ncentral
    NF = end_cycle - start_cycle + 1

    b_list: list[np.ndarray] = []
    c_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    s_list: list[np.ndarray] = []
    jk_map: list[tuple[int, int, int]] = []
    lambda_files: list[Path] = []

    for i in range(NF):
        analysis_dir = Path(f"../analysis{start_cycle + i}")
        data_dir = analysis_dir / "data"

        if not data_dir.is_dir():
            print(f"Warning: Directory {data_dir} not found")
            continue

        b_shift = _load_shift_file(analysis_dir, "b_shift.dat")
        c_shift = _load_shift_file(analysis_dir, "c_shift.dat")
        x_shift = _load_shift_file(analysis_dir, "x_shift.dat")
        s_shift = _load_shift_file(analysis_dir, "s_shift.dat")
        b_fix_shift = _load_shift_file(analysis_dir, "b_fix_shift.dat", 0.0)
        c_fix_shift = _load_shift_file(analysis_dir, "c_fix_shift.dat", 0.0)
        x_fix_shift = _load_shift_file(analysis_dir, "x_fix_shift.dat", 0.0)
        s_fix_shift = _load_shift_file(analysis_dir, "s_fix_shift.dat", 0.0)

        lambda_fpaths = find_lambda_files(data_dir)

        for fpath in lambda_fpaths:
            try:
                j, k = map(int, fpath.name.split(".")[1:3])
                jk_map.append((j, k, i))
                lambda_files.append(fpath)
                b_old = np.loadtxt(analysis_dir / "b_prev.dat")
                b_list.append(b_old + b_shift * (k - ncentral) + b_fix_shift)
                c_old = np.loadtxt(analysis_dir / "c_prev.dat")
                c_list.append(c_old + c_shift * (k - ncentral) + c_fix_shift)
                x_old = np.loadtxt(analysis_dir / "x_prev.dat")
                x_list.append(x_old + x_shift * (k - ncentral) + x_fix_shift)
                s_old = np.loadtxt(analysis_dir / "s_prev.dat")
                s_list.append(s_old + s_shift * (k - ncentral) + s_fix_shift)
            except (OSError, ValueError) as e:
                logger.warning("Error loading file %s: %s", fpath, e)

    return b_list, c_list, x_list, s_list, jk_map, lambda_files, skipE


def _compute_cross_energy_matrix(
    Lambda: list[np.ndarray],
    b_list: list[np.ndarray],
    c_list: list[np.ndarray],
    x_list: list[np.ndarray],
    s_list: list[np.ndarray],
) -> list[list[np.ndarray]]:
    """Compute cross-simulation bias energies.

    E[i][j] = bias energy of simulation j's lambda under simulation i's
    parameters. Each entry has shape (n_frames_j, 1).
    """
    nf = len(Lambda)
    energy_matrix: list[list[np.ndarray]] = [[] for _ in range(nf)]
    for i in range(nf):
        bi, ci, xi, si = b_list[i], c_list[i], x_list[i], s_list[i]
        for j in range(nf):
            Lj = Lambda[j]
            Eij = np.reshape(np.dot(Lj, -bi), (-1, 1))
            Eij += np.sum(np.dot(Lj, -ci) * Lj, axis=1, keepdims=True)
            Eij += np.sum(
                np.dot(1 - np.exp(OMEGA_DECAY * Lj), -xi) * Lj,
                axis=1, keepdims=True,
            )
            Eij += np.sum(
                np.dot(Lj / (Lj + CHI_OFFSET), -si) * Lj,
                axis=1, keepdims=True,
            )
            energy_matrix[i].append(Eij)
    return energy_matrix


def _cross_energy_vec(
    Lj: np.ndarray,
    bi: np.ndarray,
    ci: np.ndarray,
    xi: np.ndarray,
    si: np.ndarray,
) -> np.ndarray:
    """Compute cross-energy of simulation j's lambda under simulation i's biases.

    Same formula as _compute_cross_energy_matrix but returns a flat 1D array
    of shape (nframes,) instead of (nframes, 1). Used by the fused packing path.

    Args:
        Lj: Lambda trajectory, shape (nframes, nblocks).
        bi: Linear bias vector for simulation i.
        ci: Quadratic coupling matrix for simulation i.
        xi: Omega coupling matrix for simulation i.
        si: Chi coupling matrix for simulation i.

    Returns:
        Energy array of shape (nframes,).
    """
    E = np.dot(Lj, -bi).ravel()
    E += np.sum(np.dot(Lj, -ci) * Lj, axis=1)
    E += np.sum(
        np.dot(1 - np.exp(OMEGA_DECAY * Lj), -xi) * Lj,
        axis=1,
    )
    E += np.sum(
        np.dot(Lj / (Lj + CHI_OFFSET), -si) * Lj,
        axis=1,
    )
    return E


def _compute_gshift_data(
    jk_map: list[tuple[int, int, int]],
    start_cycle: int,
    nblocks: int,
    ncentral: int,
) -> np.ndarray:
    """Compute G_imp shift data for all simulations.

    Returns:
        Array of shape (n_simulations, nblocks).
    """
    nf = len(jk_map)
    gshift_data = np.zeros((nf, nblocks), dtype=np.float64)
    for sim_idx, (j, k, analysis_idx) in enumerate(jk_map):
        analysis_dir = f"../analysis{start_cycle + analysis_idx}"
        try:
            b_shift_arr = np.atleast_1d(
                _load_shift_file(analysis_dir, "b_shift.dat", 0.0)
            )
            b_fix_shift_arr = np.atleast_1d(
                _load_shift_file(analysis_dir, "b_fix_shift.dat", 0.0)
            )
            if b_shift_arr.size == 1:
                b_shift_arr = np.full(nblocks, float(b_shift_arr.flat[0]))
            if b_fix_shift_arr.size == 1:
                b_fix_shift_arr = np.full(nblocks, float(b_fix_shift_arr.flat[0]))
            for block_idx in range(nblocks):
                gshift_data[sim_idx, block_idx] = (
                    float(b_fix_shift_arr[block_idx])
                    + float(b_shift_arr[block_idx]) * (k - ncentral)
                )
        except Exception as e:
            print(
                f"Warning: Could not compute G_imp shifts for sim {sim_idx + 1}: {e}"
            )
    return gshift_data


def get_energy_from_analysis_dir(
    alf_info: ALFInfo | dict,
    start_cycle: int,
    end_cycle: int,
    skipE: int = 1,
) -> int:
    """Compute bias energies for WHAM from inside an analysis directory.

    This matches the external ALF GetEnergy convention: called from inside
    an analysis directory (e.g., analysis5/), reads lambda data from
    ../analysis{i}/data/ and bias parameters from ../analysis{i}/,
    and writes Lambda/ and Energy/ subdirectories in the current directory.

    Supports shift files for replica exchange (nbshift):
    - b_shift.dat, c_shift.dat, x_shift.dat, s_shift.dat
    - b_fix_shift.dat, c_fix_shift.dat, x_fix_shift.dat, s_fix_shift.dat

    Args:
        alf_info: ALF simulation information.
        start_cycle: First ALF cycle to include (inclusive).
        end_cycle: Final ALF cycle to include (inclusive).
        skipE: Subsample interval (only every Nth frame). Default 1 = all.

    Returns:
        Number of simulations (ESim files) created.
    """
    import os

    alf_info = ensure_alf_info(alf_info)

    Lambda, b_list, c_list, x_list, s_list, jk_map = _load_simulation_data(
        alf_info, start_cycle, end_cycle, skipE,
    )

    nf = len(Lambda)
    if nf == 0:
        print("Error: No Lambda files found.")
        return 0

    energy_matrix = _compute_cross_energy_matrix(
        Lambda, b_list, c_list, x_list, s_list,
    )

    os.makedirs("Lambda", exist_ok=True)
    os.makedirs("Energy", exist_ok=True)

    for i in range(nf):
        Ei = energy_matrix[i][i]
        for j in range(nf):
            Ei = np.concatenate((Ei, energy_matrix[j][i]), axis=1)
        np.savetxt(f"Energy/ESim{i + 1}.dat", Ei, fmt="%12.5f")

    for i in range(nf):
        np.savetxt(f"Lambda/Lambda{i + 1}.dat", Lambda[i], fmt="%10.6f")

    # Write G_imp shift files
    os.makedirs("G_imp_shifts", exist_ok=True)
    gshift_data = _compute_gshift_data(
        jk_map, start_cycle, alf_info.nblocks, alf_info.ncentral,
    )
    for sim_idx, (j, k, _) in enumerate(jk_map):
        with open(f"G_imp_shifts/shifts_sim{sim_idx + 1}.dat", "w") as f:
            f.write(f"# G_imp shift information for simulation {sim_idx + 1}\n")
            f.write(f"# j: {j}, k: {k}, ncentral: {alf_info.ncentral}\n")
            for block_idx in range(alf_info.nblocks):
                f.write(f"{gshift_data[sim_idx, block_idx]:.6f}\n")

    return nf


def compute_wham_inputs(
    alf_info: ALFInfo | dict,
    start_cycle: int,
    end_cycle: int,
    skipE: int = 1,
) -> tuple[list[np.ndarray], list[list[np.ndarray]], np.ndarray | None, int]:
    """Compute WHAM inputs as in-memory arrays (no intermediate text files).

    Same data-loading and energy-computation logic as get_energy_from_analysis_dir(),
    but returns numpy arrays instead of writing Lambda/, Energy/, and G_imp_shifts/.

    Must be called from inside an analysis directory (e.g., analysis5/).

    Args:
        alf_info: ALF simulation information.
        start_cycle: First ALF cycle to include (inclusive).
        end_cycle: Final ALF cycle to include (inclusive).
        skipE: Subsample interval (only every Nth frame). Default 1 = all.

    Returns:
        Tuple of (lambda_arrays, energy_matrix, gshift_data, nf):
        - lambda_arrays: list of nf arrays, each [n_frames_i, nblocks]
        - energy_matrix: E[i][j] = (n_frames_j, 1) array — bias energy of
          simulation j's lambda under simulation i's parameters
        - gshift_data: [nf, nblocks] array of G_imp shifts, or None if no shifts
        - nf: number of simulations (total lambda files found)
    """
    alf_info = ensure_alf_info(alf_info)

    Lambda, b_list, c_list, x_list, s_list, jk_map = _load_simulation_data(
        alf_info, start_cycle, end_cycle, skipE,
    )

    nf = len(Lambda)
    if nf == 0:
        print("Error: No Lambda files found.")
        return [], [], None, 0

    energy_matrix = _compute_cross_energy_matrix(
        Lambda, b_list, c_list, x_list, s_list,
    )
    gshift_data = _compute_gshift_data(
        jk_map, start_cycle, alf_info.nblocks, alf_info.ncentral,
    )

    return Lambda, energy_matrix, gshift_data, nf


def compute_wham_inputs_distributed(
    alf_info: ALFInfo | dict,
    start_cycle: int,
    end_cycle: int,
    skipE: int = 1,
    comm=None,
    rank: int = 0,
    nranks: int = 1,
) -> tuple[list[np.ndarray], list[list[np.ndarray]], np.ndarray | None, int]:
    """MPI-parallel WHAM input computation.

    Distributes cross-energy matrix computation across MPI ranks:
    - All ranks load simulation data (I/O is fast for parquet)
    - Each rank computes its assigned rows of the energy matrix
    - Rank 0 gathers all rows into the complete energy matrix

    Falls back to serial compute_wham_inputs() when nranks=1.

    Args:
        alf_info: ALF simulation information.
        start_cycle: First ALF cycle to include (inclusive).
        end_cycle: Final ALF cycle to include (inclusive).
        skipE: Subsample interval. Default 1 = all frames.
        comm: MPI communicator (None for serial).
        rank: This rank's index.
        nranks: Total number of ranks.

    Returns:
        On rank 0: same as compute_wham_inputs().
        On other ranks: ([], [], None, 0) — only rank 0 has the full data.
    """
    # Single-rank fallback
    if nranks <= 1 or comm is None:
        return compute_wham_inputs(alf_info, start_cycle, end_cycle, skipE)

    alf_info = ensure_alf_info(alf_info)

    # All ranks load the same simulation data (parquet I/O is fast)
    Lambda, b_list, c_list, x_list, s_list, jk_map = _load_simulation_data(
        alf_info, start_cycle, end_cycle, skipE,
    )

    nf = len(Lambda)
    if nf == 0:
        if rank == 0:
            print("Error: No Lambda files found.")
        return [], [], None, 0

    # Distribute cross-energy matrix rows across ranks
    my_rows = list(range(rank, nf, nranks))  # interleaved distribution

    # Each rank computes its rows of the energy matrix
    local_energy: dict[int, list[np.ndarray]] = {}
    for i in my_rows:
        bi, ci, xi, si = b_list[i], c_list[i], x_list[i], s_list[i]
        row = []
        for j in range(nf):
            Lj = Lambda[j]
            Eij = np.reshape(np.dot(Lj, -bi), (-1, 1))
            Eij += np.sum(np.dot(Lj, -ci) * Lj, axis=1, keepdims=True)
            Eij += np.sum(
                np.dot(1 - np.exp(OMEGA_DECAY * Lj), -xi) * Lj,
                axis=1, keepdims=True,
            )
            Eij += np.sum(
                np.dot(Lj / (Lj + CHI_OFFSET), -si) * Lj,
                axis=1, keepdims=True,
            )
            row.append(Eij)
        local_energy[i] = row

    # Gather all rows on rank 0
    gathered = comm.gather(local_energy, root=0)

    if rank == 0:
        # Reconstruct full energy matrix from gathered dicts
        energy_matrix: list[list[np.ndarray]] = [[] for _ in range(nf)]
        for rank_dict in gathered:
            for i, row in rank_dict.items():
                energy_matrix[i] = row

        gshift_data = _compute_gshift_data(
            jk_map, start_cycle, alf_info.nblocks, alf_info.ncentral,
        )
        return Lambda, energy_matrix, gshift_data, nf
    else:
        return [], [], None, 0


# --------------------------------------------------------------------------
# Fused packed computation — eliminates intermediate energy_matrix
# --------------------------------------------------------------------------


def _purge_stale_mmap_files() -> None:
    """Remove stale WHAM mmap files from /tmp owned by this user.

    Builds a set of open file paths via readlink on /proc/*/fd (single
    scan, NFS-safe — never stats target files), then removes mmap files
    not in that set.  Safe for concurrent jobs sharing the same node.
    """
    import glob
    import os

    my_uid = os.getuid()
    candidates = []
    for path in glob.glob("/tmp/tmp*.mmap"):
        try:
            if os.stat(path).st_uid == my_uid:
                candidates.append(os.path.abspath(path))
        except OSError:
            pass

    if not candidates:
        return

    # Build set of open file paths — single /proc scan via readlink (no NFS stat)
    open_paths: set[str] = set()
    for pid_dir in os.listdir("/proc"):
        if not pid_dir.isdigit():
            continue
        fd_dir = f"/proc/{pid_dir}/fd"
        try:
            for fd in os.listdir(fd_dir):
                try:
                    open_paths.add(os.readlink(f"{fd_dir}/{fd}"))
                except OSError:
                    continue
        except OSError:
            continue

    for path in candidates:
        if path not in open_paths:
            try:
                os.unlink(path)
            except OSError:
                pass


def compute_packed_wham_data(
    alf_info: ALFInfo | dict,
    start_cycle: int,
    end_cycle: int,
    skipE: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int, int]:
    """Compute WHAM inputs as a pre-packed flat D_h array for CUDA.

    Streams lambda files one at a time (j-outer loop) so peak lambda memory
    is ONE file (~640 KB) instead of ALL files (~96 MB for nf=150).
    Cross-energies are written directly into the flat D_h layout.

    Must be called from inside an analysis directory (e.g., analysis5/).

    Args:
        alf_info: ALF simulation information.
        start_cycle: First ALF cycle to include (inclusive).
        end_cycle: Final ALF cycle to include (inclusive).
        skipE: Subsample interval. Default 1 = all frames.

    Returns:
        Tuple of (D_flat, sim_indices, frame_counts, gshift_data, nf, total_frames):
        - D_flat: float64 array of shape (total_frames * ndim,) in CUDA D_h layout
        - sim_indices: int32 array of shape (total_frames,)
        - frame_counts: int32 array of shape (nf,)
        - gshift_data: [nf, nblocks] array, or None if no shifts
        - nf: number of simulations
        - total_frames: total frames across all simulations
    """
    from cphmd.utils.lambda_io import get_lambda_frame_count, read_lambda_values

    alf_info = ensure_alf_info(alf_info)

    b_list, c_list, x_list, s_list, jk_map, lambda_files, skipE = _load_bias_params(
        alf_info, start_cycle, end_cycle, skipE,
    )

    nf = len(b_list)
    if nf == 0:
        print("Error: No Lambda files found.")
        empty = np.zeros(0, dtype=np.float64)
        return empty, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), None, 0, 0

    NL = alf_info.nblocks
    ndim = NL + nf + 3

    frame_counts = np.array(
        [get_lambda_frame_count(f, skipE) for f in lambda_files], dtype=np.int32
    )
    total_frames = int(frame_counts.sum())

    # Use disk-backed memmap for D when it would exceed 512 MB in RAM.
    # Sequential write + sequential CUDA read is the ideal memmap access pattern.
    # Below threshold, plain numpy avoids filesystem overhead.
    d_bytes = total_frames * ndim * 8
    _MEMMAP_THRESHOLD = 512 * 1024 * 1024  # 512 MB
    if d_bytes > _MEMMAP_THRESHOLD:
        import tempfile
        _purge_stale_mmap_files()
        _d_tmpfile = tempfile.NamedTemporaryFile(suffix=".mmap", delete=True)
        D = np.memmap(_d_tmpfile, dtype=np.float64, mode="w+",
                      shape=(total_frames, ndim))
        logger.info(
            f"D matrix using memmap ({d_bytes / 1e9:.1f} GB): {_d_tmpfile.name}"
        )
    else:
        _d_tmpfile = None
        D = np.zeros((total_frames, ndim), dtype=np.float64)

    sim_indices = np.empty(total_frames, dtype=np.int32)

    # Stream one lambda file at a time (j-outer loop)
    offset = 0
    for j in range(nf):
        Lj = read_lambda_values(lambda_files[j])[(skipE - 1) :: skipE, :]
        n_j = frame_counts[j]

        # Fill lambda columns and sim index for this simulation
        D[offset : offset + n_j, 1 : 1 + NL] = Lj[:n_j]
        sim_indices[offset : offset + n_j] = j

        # Compute cross-energies against ALL bias sets
        for i in range(nf):
            D[offset : offset + n_j, NL + 1 + i] = _cross_energy_vec(
                Lj[:n_j], b_list[i], c_list[i], x_list[i], s_list[i],
            )

        offset += n_j
        # Lj goes out of scope — memory freed

    # E_self column = self-energy (each sim's frames under its OWN biases).
    # Combined with per-replica gshift, this gives a uniform target potential
    # dot(λ, -b_old) for all frames, avoiding a systematic offset.
    offset = 0
    for j in range(nf):
        n_j = frame_counts[j]
        D[offset : offset + n_j, 0] = D[offset : offset + n_j, NL + 1 + j]
        offset += n_j

    gshift_data = _compute_gshift_data(
        jk_map, start_cycle, alf_info.nblocks, alf_info.ncentral,
    )

    # For memmap: flush so CUDA reads see all written data.
    # ravel() of a memmap stays disk-backed, so peak RSS remains low
    # while cudaMemcpy pages data in sequentially.
    if _d_tmpfile is not None:
        D.flush()
        D_flat = D.ravel()
        # Keep tmpfile alive — NamedTemporaryFile deletes on close.
        # memmap (ndarray subclass) accepts arbitrary attributes.
        D_flat._memmap_tmpfile = _d_tmpfile
    else:
        D_flat = D.ravel()

    return D_flat, sim_indices, frame_counts, gshift_data, nf, total_frames


def compute_packed_wham_data_distributed(
    alf_info: ALFInfo | dict,
    start_cycle: int,
    end_cycle: int,
    skipE: int = 1,
    comm=None,
    rank: int = 0,
    nranks: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int, int]:
    """MPI-parallel fused packed WHAM data computation with streaming lambda I/O.

    Streams lambda files one at a time (j-outer loop) so peak lambda memory
    is ONE file instead of ALL files. Each rank computes its assigned
    cross-energy rows and sends them to rank 0 via buffer-based Send/Recv.

    Falls back to serial compute_packed_wham_data() when nranks <= 1.

    Args:
        alf_info: ALF simulation information.
        start_cycle: First ALF cycle to include (inclusive).
        end_cycle: Final ALF cycle to include (inclusive).
        skipE: Subsample interval. Default 1 = all frames.
        comm: MPI communicator (None for serial).
        rank: This rank's index.
        nranks: Total number of ranks.

    Returns:
        On rank 0: same as compute_packed_wham_data().
        On other ranks: empty arrays with nf=0, total_frames=0.
    """
    if nranks <= 1 or comm is None:
        return compute_packed_wham_data(alf_info, start_cycle, end_cycle, skipE)

    from cphmd.utils.lambda_io import get_lambda_frame_count, read_lambda_values

    alf_info = ensure_alf_info(alf_info)

    # All ranks load bias params (fast — no lambda data)
    b_list, c_list, x_list, s_list, jk_map, lambda_files, skipE = _load_bias_params(
        alf_info, start_cycle, end_cycle, skipE,
    )

    nf = len(b_list)
    if nf == 0:
        if rank == 0:
            print("Error: No Lambda files found.")
        empty = np.zeros(0, dtype=np.float64)
        return empty, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), None, 0, 0

    NL = alf_info.nblocks
    ndim = NL + nf + 3
    frame_counts = np.array(
        [get_lambda_frame_count(f, skipE) for f in lambda_files], dtype=np.int32
    )
    total_frames = int(frame_counts.sum())

    # Distribute cross-energy rows across ranks (interleaved by bias index)
    my_rows = list(range(rank, nf, nranks))

    # Each rank computes its assigned cross-energy rows by streaming lambda
    local_rows: dict[int, np.ndarray] = {i: np.empty(total_frames, dtype=np.float64) for i in my_rows}

    # Rank 0: pre-allocate D and fill lambda columns during the SAME pass
    # that computes cross-energies — avoids reading lambda files twice.
    if rank == 0:
        d_bytes = total_frames * ndim * 8
        _MEMMAP_THRESHOLD = 512 * 1024 * 1024  # 512 MB
        if d_bytes > _MEMMAP_THRESHOLD:
            import tempfile
            _purge_stale_mmap_files()
            _d_tmpfile = tempfile.NamedTemporaryFile(suffix=".mmap", delete=True)
            D = np.memmap(_d_tmpfile, dtype=np.float64, mode="w+",
                          shape=(total_frames, ndim))
            logger.info(
                f"D matrix using memmap ({d_bytes / 1e9:.1f} GB): {_d_tmpfile.name}"
            )
        else:
            _d_tmpfile = None
            D = np.zeros((total_frames, ndim), dtype=np.float64)
        sim_indices = np.empty(total_frames, dtype=np.int32)
    else:
        D = None
        sim_indices = None
        _d_tmpfile = None

    offset = 0
    for j in range(nf):
        Lj = read_lambda_values(lambda_files[j])[(skipE - 1) :: skipE, :]
        n_j = frame_counts[j]
        for i in my_rows:
            local_rows[i][offset : offset + n_j] = _cross_energy_vec(
                Lj[:n_j], b_list[i], c_list[i], x_list[i], s_list[i],
            )
        # Rank 0: fill lambda columns and sim indices in same pass (no second read)
        if rank == 0:
            D[offset : offset + n_j, 1 : 1 + NL] = Lj[:n_j]
            sim_indices[offset : offset + n_j] = j
        offset += n_j
        # Lj goes out of scope — memory freed

    if rank == 0:
        # Copy own rows into D
        for i, row in local_rows.items():
            D[:, NL + 1 + i] = row

        # Receive rows from other ranks via buffer-based Recv
        recv_buf = np.empty(total_frames, dtype=np.float64)
        for src_rank in range(1, nranks):
            src_rows = list(range(src_rank, nf, nranks))
            for i in src_rows:
                comm.Recv(recv_buf, source=src_rank, tag=i)
                D[:, NL + 1 + i] = recv_buf

        # E_self column = self-energy (each sim's frames under its OWN biases).
        offset = 0
        for j in range(nf):
            n_j = frame_counts[j]
            D[offset : offset + n_j, 0] = D[offset : offset + n_j, NL + 1 + j]
            offset += n_j

        gshift_data = _compute_gshift_data(
            jk_map, start_cycle, alf_info.nblocks, alf_info.ncentral,
        )

        if _d_tmpfile is not None:
            D.flush()
            D_flat = D.ravel()
            D_flat._memmap_tmpfile = _d_tmpfile
        else:
            D_flat = D.ravel()

        return D_flat, sim_indices, frame_counts, gshift_data, nf, total_frames
    else:
        # Send own rows to rank 0
        for i, row in local_rows.items():
            comm.Send(np.ascontiguousarray(row, dtype=np.float64), dest=0, tag=i)

        empty = np.zeros(0, dtype=np.float64)
        return empty, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), None, 0, 0


def compute_bias_energy(
    lambda_vec: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    s: np.ndarray
) -> float:
    """Compute total bias energy for a lambda configuration.

    The bias energy has four components:
    - Linear (phi): -b·λ
    - Quadratic psi: -λᵀ·c·λ
    - Omega (x-term): -λᵀ·(1-exp(OMEGA_DECAY·λ))·x·λ  (OMEGA_DECAY < 0)
    - Chi (s-term): -λᵀ·(λ/(λ+CHI_OFFSET))·s·λ

    Constants are derived from FNEX (see bias_constants module).

    Args:
        lambda_vec: Lambda coordinates (nblocks,).
        b: Linear bias (1, nblocks).
        c: Quadratic psi coupling (nblocks, nblocks).
        x: Omega coupling (nblocks, nblocks).
        s: Chi coupling (nblocks, nblocks).

    Returns:
        Total bias energy in kcal/mol.
    """
    lam = lambda_vec.flatten()
    b = b.flatten()

    # Linear term
    E_b = -np.dot(b, lam)

    # Quadratic psi term
    E_c = -np.dot(lam, np.dot(c, lam))

    # Omega term (exponential)
    omega_factor = 1.0 - np.exp(OMEGA_DECAY * lam)
    E_x = -np.dot(lam * omega_factor, np.dot(x, lam))

    # Chi term (sigmoid)
    chi_factor = lam / (lam + CHI_OFFSET + 1e-10)  # Avoid division by zero
    E_s = -np.dot(lam * chi_factor, np.dot(s, lam))

    return E_b + E_c + E_x + E_s


def convert_lambda_binary_to_parquet(
    alf_info: ALFInfo | dict,
    output_path: str | Path,
    input_files: list[str | Path],
) -> None:
    """Convert binary lambda file(s) to Parquet format.

    Args:
        alf_info: ALF simulation information (used for nsubs column naming).
        output_path: Path to output .parquet file.
        input_files: List of input binary lambda file paths.
    """
    from cphmd.utils.lambda_io import read_lambda_binary, write_lambda_parquet

    nsubs = alf_info.get("nsubs") if isinstance(alf_info, dict) else None

    all_data = []
    for input_file in input_files:
        input_path = Path(input_file)
        if input_path.exists():
            data, _ = read_lambda_binary(str(input_path))
            all_data.append(data)

    if all_data:
        combined = np.vstack(all_data)
        write_lambda_parquet(output_path, combined, nsubs=nsubs)
    else:
        print(f"Warning: No valid input files found for {output_path}")


def get_free_energy_lm(
    alf_info: ALFInfo | dict,
    ms: int = 0,
    msprof: int = 0,
    cutb: float = 2.0,
    cutc: float = 8.0,
    cutx: float = 2.0,
    cuts: float = 1.0,
    cutt: float = 0.0,
    cutu: float = 0.0,
    cutc2: float = 2.0,
    cutx2: float = 0.5,
    cuts2: float = 0.5,
    ntriangle: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert LMALF OUT.dat to b/c/x/s bias matrices.

    This function post-processes the output of LMALF optimization (OUT.dat)
    to generate the standard bias parameter files (b.dat, c.dat, x.dat, s.dat).

    The conversion involves:
    1. Reading the flat parameter vector from OUT.dat
    2. Applying adaptive scaling to prevent large parameter changes
    3. Unpacking into b/c/x/s matrices based on system topology
    4. Writing output files

    Args:
        alf_info: ALF simulation information with nsubs topology.
        ms: Multisite coupling flag (0=none, 1=full coupling, 2=c-only).
        msprof: Multisite profiles flag (unused, for API compatibility).
        cutb: Cutoff for b (linear) parameters.
        cutc: Cutoff for c (coupling) parameters (same-site).
        cutx: Cutoff for x (omega) parameters (same-site).
        cuts: Cutoff for s (chi) parameters (same-site).
        cutc2: Cutoff for c parameters (cross-site).
        cutx2: Cutoff for x parameters (cross-site).
        cuts2: Cutoff for s parameters (cross-site).

    Returns:
        Tuple of (b, c, x, s) numpy arrays.

    Raises:
        FileNotFoundError: If OUT.dat or *_prev.dat files not found.
        ValueError: If parameter count doesn't match topology.

    Note:
        Must be called from analysis directory containing:
        - OUT.dat: LMALF output (flat parameter vector)
        - b_prev.dat, c_prev.dat, x_prev.dat, s_prev.dat: Previous cycle biases

        Produces in current directory:
        - b.dat, c.dat, x.dat, s.dat: New bias parameter files
    """
    alf_info = ensure_alf_info(alf_info)
    kT = 0.001987 * alf_info.temp

    nsubs = alf_info.nsubs
    nblocks = alf_info.nblocks

    # Load previous parameters
    b_prev = np.loadtxt("b_prev.dat")
    c_prev = np.loadtxt("c_prev.dat")
    x_prev = np.loadtxt("x_prev.dat")
    s_prev = np.loadtxt("s_prev.dat")

    # Initialize output arrays
    b = np.zeros((1, nblocks))
    c = np.zeros((nblocks, nblocks))
    x = np.zeros((nblocks, nblocks))
    s = np.zeros((nblocks, nblocks))
    t = np.zeros((nblocks, nblocks))
    u = np.zeros((nblocks, nblocks))

    # Count expected parameters
    nparm = 0
    for isite in range(len(nsubs)):
        n1 = nsubs[isite]
        n2 = nsubs[isite] * (nsubs[isite] - 1) // 2
        for jsite in range(isite, len(nsubs)):
            n3 = nsubs[isite] * nsubs[jsite]
            if isite == jsite:
                nparm += n1 + ntriangle * n2
            elif ms == 1:
                nparm += ntriangle * n3
            elif ms == 2:
                nparm += n3

    # Build cutoff list for adaptive scaling
    cutlist = np.zeros(nparm)
    n0 = 0
    for isite in range(len(nsubs)):
        for jsite in range(isite, len(nsubs)):
            if isite == jsite:
                for i in range(nsubs[isite]):
                    cutlist[n0:n0 + 1] = cutb
                    n0 += 1
                    for j in range(i + 1, nsubs[isite]):
                        cutlist[n0:n0 + 1] = cutc
                        n0 += 1
                        if ntriangle >= 3:
                            cutlist[n0:n0 + 2] = cutx
                            n0 += 2
                        if ntriangle >= 5:
                            cutlist[n0:n0 + 2] = cuts
                            n0 += 2
                        if ntriangle >= 7:
                            cutlist[n0:n0 + 2] = cutt
                            n0 += 2
                        if ntriangle >= 9:
                            cutlist[n0:n0 + 2] = cutu
                            n0 += 2
            elif ms > 0:
                for i in range(nsubs[isite]):
                    for j in range(nsubs[jsite]):
                        cutlist[n0:n0 + 1] = cutc2
                        n0 += 1
                        if ms == 1:
                            if ntriangle >= 3:
                                cutlist[n0:n0 + 2] = cutx2
                                n0 += 2
                            if ntriangle >= 5:
                                cutlist[n0:n0 + 2] = cuts2
                                n0 += 2
                            if ntriangle >= 7:
                                cutlist[n0:n0 + 2] = cutt
                                n0 += 2
                            if ntriangle >= 9:
                                cutlist[n0:n0 + 2] = cutu
                                n0 += 2

    # Load LMALF output
    coeff = np.loadtxt("OUT.dat")

    # Global scaling: if the worst-case ratio |coeff[i]|/cutlist[i] exceeds
    # 1.5, scale the ENTIRE correction vector down proportionally.  This
    # preserves the optimizer's chosen proportions across all parameter
    # types — unlike per-parameter clipping, which distorts the coupled
    # LMALF solution and can trigger sign oscillation between iterations.
    active_mask = cutlist[:n0] > 0
    if np.any(active_mask):
        active_ratios = np.abs(coeff[:n0][active_mask] / cutlist[:n0][active_mask])
        max_ratio = float(np.max(active_ratios))
    else:
        max_ratio = 0.0

    if max_ratio > 1.5:
        scaling = 1.5 / max_ratio
        coeff[:n0] *= scaling
        print(f"LMALF: max_ratio={max_ratio:.2f}, global scaling={scaling:.4f}")
    else:
        print(f"LMALF: max_ratio={max_ratio:.2f}, no scaling needed")

    # Unpack coefficients into b/c/x/s matrices
    ind = 0
    iblock = 0
    for isite in range(len(nsubs)):
        jblock = iblock
        for jsite in range(isite, len(nsubs)):
            if isite == jsite:
                for i in range(nsubs[isite]):
                    b[0, iblock + i] = coeff[ind]
                    ind += 1
                    for j in range(i + 1, nsubs[isite]):
                        c[iblock + i, jblock + j] = coeff[ind]
                        ind += 1
                        if ntriangle >= 3:
                            x[iblock + i, jblock + j] = coeff[ind]
                            ind += 1
                            x[jblock + j, iblock + i] = coeff[ind]
                            ind += 1
                        if ntriangle >= 5:
                            s[iblock + i, jblock + j] = coeff[ind]
                            ind += 1
                            s[jblock + j, iblock + i] = coeff[ind]
                            ind += 1
                        if ntriangle >= 7:
                            t[iblock + i, jblock + j] = coeff[ind]
                            ind += 1
                            t[jblock + j, iblock + i] = coeff[ind]
                            ind += 1
                        if ntriangle >= 9:
                            u[iblock + i, jblock + j] = coeff[ind]
                            ind += 1
                            u[jblock + j, iblock + i] = coeff[ind]
                            ind += 1
            elif ms > 0:
                for i in range(nsubs[isite]):
                    for j in range(nsubs[jsite]):
                        c[iblock + i, jblock + j] = coeff[ind]
                        ind += 1
                        if ms == 1:
                            if ntriangle >= 3:
                                x[iblock + i, jblock + j] = coeff[ind]
                                ind += 1
                                x[jblock + j, iblock + i] = coeff[ind]
                                ind += 1
                            if ntriangle >= 5:
                                s[iblock + i, jblock + j] = coeff[ind]
                                ind += 1
                                s[jblock + j, iblock + i] = coeff[ind]
                                ind += 1
                            if ntriangle >= 7:
                                t[iblock + i, jblock + j] = coeff[ind]
                                ind += 1
                                t[jblock + j, iblock + i] = coeff[ind]
                                ind += 1
                            if ntriangle >= 9:
                                u[iblock + i, jblock + j] = coeff[ind]
                                ind += 1
                                u[jblock + j, iblock + i] = coeff[ind]
                                ind += 1
            jblock += nsubs[jsite]
        iblock += nsubs[isite]

    # Apply cross-site corrections to b
    # (absorb constant parts of cross-site coupling into b)
    iblock = 0
    for isite in range(len(nsubs)):
        jblock = iblock
        for jsite in range(isite, len(nsubs)):
            if isite != jsite:
                for i in range(nsubs[isite]):
                    b[0, iblock + i] += c[iblock + i, jblock]
                    c[iblock + i, jblock:jblock + nsubs[jsite]] -= c[iblock + i, jblock]
                for j in range(nsubs[jsite]):
                    b[0, jblock + j] += c[iblock, jblock + j]
                    c[iblock:iblock + nsubs[isite], jblock + j] -= c[iblock, jblock + j]
            jblock += nsubs[jsite]
        iblock += nsubs[isite]

    # Normalize b within each site (first substate becomes reference)
    iblock = 0
    for isite in range(len(nsubs)):
        b[0, iblock:iblock + nsubs[isite]] -= b[0, iblock]
        iblock += nsubs[isite]

    # Write output files
    np.savetxt("b.dat", _clean_negzero(b), fmt=" %10.5f")
    np.savetxt("c.dat", _clean_negzero(c), fmt=" %10.5f")
    np.savetxt("x.dat", _clean_negzero(x), fmt=" %10.5f")
    np.savetxt("s.dat", _clean_negzero(s), fmt=" %10.5f")
    np.savetxt("t.dat", _clean_negzero(t), fmt=" %10.5f")
    np.savetxt("u.dat", _clean_negzero(u), fmt=" %10.5f")

    print("LMALF: Wrote b.dat, c.dat, x.dat, s.dat, t.dat, u.dat")

    return b, c, x, s, t, u
