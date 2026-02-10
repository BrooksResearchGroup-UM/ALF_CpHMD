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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preset bias parameters and convert to arrays.

    Args:
        alf_info: ALF simulation information.
        preset_residue: Residue name for presets. If None, uses alf_info.name.
        preset_config: Preset configuration name ("scat_noh", "noe_h").
            If None, uses default ("scat_noh").

    Returns:
        Tuple of (b, c, x, s) arrays for bias parameters.

    Raises:
        KeyError: If preset not found for the residue/config.
    """
    from cphmd.presets import get_preset_biases

    # Determine residue name
    resname = preset_residue or alf_info.name
    if resname is None:
        raise ValueError("preset_residue must be specified or alf_info.name must be set")

    # Load preset biases
    preset = get_preset_biases(resname.upper(), config=preset_config)
    nblocks = alf_info.nblocks
    nsubs = alf_info.nsubs

    # Initialize arrays
    b = np.zeros([1, nblocks])
    c = np.zeros([nblocks, nblocks])
    x = np.zeros([nblocks, nblocks])
    s = np.zeros([nblocks, nblocks])

    # Compute subsite offsets (for multi-site support)
    sub0 = np.cumsum(nsubs) - nsubs

    # Fill linear biases (b/lams)
    for si in range(len(nsubs)):
        for i in range(nsubs[si]):
            key = f"lams{si+1}s{i+1}"
            if key in preset:
                b[0, sub0[si] + i] = preset[key]

    # Fill psi coupling (c/cs) - stored as negative in preset
    for si in range(len(nsubs)):
        for sj in range(si, len(nsubs)):
            for i in range(nsubs[si]):
                j0 = (i + 1 if si == sj else 0)
                for j in range(j0, nsubs[sj]):
                    key = f"cs{si+1}s{i+1}s{sj+1}s{j+1}"
                    if key in preset:
                        # Preset stores as negative, arrays store positive
                        c[sub0[si] + i, sub0[sj] + j] = -preset[key]
                        c[sub0[sj] + j, sub0[si] + i] = -preset[key]  # Symmetric

    # Fill omega coupling (x/xs)
    for si in range(len(nsubs)):
        for sj in range(len(nsubs)):
            for i in range(nsubs[si]):
                for j in range(nsubs[sj]):
                    if sub0[si] + i != sub0[sj] + j:
                        key = f"xs{si+1}s{i+1}s{sj+1}s{j+1}"
                        if key in preset:
                            x[sub0[si] + i, sub0[sj] + j] = -preset[key]

    # Fill chi coupling (s/ss)
    for si in range(len(nsubs)):
        for sj in range(len(nsubs)):
            for i in range(nsubs[si]):
                for j in range(nsubs[sj]):
                    if sub0[si] + i != sub0[sj] + j:
                        key = f"ss{si+1}s{i+1}s{sj+1}s{j+1}"
                        if key in preset:
                            s[sub0[si] + i, sub0[sj] + j] = -preset[key]

    return b, c, x, s


def init_vars(
    work_dir: str | Path,
    alf_info: ALFInfo | dict,
    minimize: bool = True,
    use_presets: bool = False,
    preset_residue: str | None = None,
    preset_config: str | None = None,
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
        preset_config: Preset configuration ("scat_noh", "noe_h").
            If None, uses default ("scat_noh").

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
        b, c, x, s = _load_preset_biases(alf_info, preset_residue, preset_config)
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

    # Create nbshift directory for replica exchange (if not exists)
    nbshift = dirs["nbshift"]
    if not (nbshift / "b_shift.dat").exists():
        np.savetxt(nbshift / "b_shift.dat", b)
        np.savetxt(nbshift / "c_shift.dat", c)
        np.savetxt(nbshift / "x_shift.dat", c)
        np.savetxt(nbshift / "s_shift.dat", c)

    # Generate variables1.inp from inside analysis0 (matching external ALF convention)
    set_vars_from_analysis_dir(analysis0, alf_info, step=1, minimize=minimize)

    return analysis0


def set_vars(
    work_dir: str | Path,
    alf_info: ALFInfo | dict,
    step: int,
    minimize: bool = False
) -> Path:
    """Generate variables file for pyCHARMM.

    Creates variables/{step}.py containing bias parameters formatted
    for pyCHARMM to read.

    Args:
        work_dir: Working directory for ALF simulation.
        alf_info: ALF simulation information.
        step: ALF cycle number (for which biases are being written).
        minimize: Whether to run minimization this cycle.

    Returns:
        Path to generated variables file.
    """
    work_dir = Path(work_dir)
    alf_info = ensure_alf_info(alf_info)

    nblocks = alf_info.nblocks
    nsubs = alf_info.nsubs
    nreps = alf_info.nreps
    ncentral = alf_info.ncentral
    name = alf_info.name
    nnodes = alf_info.nnodes
    temp = alf_info.temp

    # Ensure directories exist
    dirs = _ensure_directories(work_dir)

    # Find the analysis directory to read from (step-1, or 0 for step=1)
    analysis_idx = max(0, step - 1)
    analysis_dir = dirs["analysis"] / str(analysis_idx)

    # Output file
    output_file = dirs["variables"] / f"{step}.py"

    with open(output_file, "w") as fp:
        fp.write("import yaml\n")
        fp.write("import numpy as np\n\n")

        bias = {}
        sub0 = np.cumsum(nsubs) - nsubs

        # Load and accumulate bias parameters
        b_prev = np.loadtxt(analysis_dir / "b_prev.dat")
        b = np.loadtxt(analysis_dir / "b.dat")
        b_sum = b_prev + b
        # Validate for NaN/Inf - use previous values if invalid
        if not np.all(np.isfinite(b_sum)):
            logger.warning(f"NaN/Inf in b_sum at step {step}, using previous values")
            b_sum = b_prev.copy()
        b_sum = np.reshape(b_sum, (1, -1))
        np.savetxt(analysis_dir / "b_sum.dat", _clean_negzero(b_sum), fmt=" %10.5f")

        # Linear biases per site/subsite
        for i in range(len(nsubs)):
            for j in range(nsubs[i]):
                key = f"lams{i+1}s{j+1}"
                bias[key] = float(b_sum[0, sub0[i] + j])

        # Quadratic psi coupling
        c_prev = np.loadtxt(analysis_dir / "c_prev.dat")
        c = np.loadtxt(analysis_dir / "c.dat")
        c_sum = c_prev + c
        # Validate for NaN/Inf - use previous values if invalid
        if not np.all(np.isfinite(c_sum)):
            logger.warning(f"NaN/Inf in c_sum at step {step}, using previous values")
            c_sum = c_prev.copy()
        np.savetxt(analysis_dir / "c_sum.dat", _clean_negzero(c_sum), fmt=" %10.5f")

        for si in range(len(nsubs)):
            for sj in range(si, len(nsubs)):
                for i in range(nsubs[si]):
                    j0 = (i + 1 if si == sj else 0)
                    for j in range(j0, nsubs[sj]):
                        key = f"cs{si+1}s{i+1}s{sj+1}s{j+1}"
                        bias[key] = -float(c_sum[sub0[si] + i, sub0[sj] + j])

        # Omega (x) coupling
        x_prev = np.loadtxt(analysis_dir / "x_prev.dat")
        x = np.loadtxt(analysis_dir / "x.dat")
        x_sum = x_prev + x
        # Validate for NaN/Inf - use previous values if invalid
        if not np.all(np.isfinite(x_sum)):
            logger.warning(f"NaN/Inf in x_sum at step {step}, using previous values")
            x_sum = x_prev.copy()
        np.savetxt(analysis_dir / "x_sum.dat", _clean_negzero(x_sum), fmt=" %10.5f")

        for si in range(len(nsubs)):
            for sj in range(len(nsubs)):
                for i in range(nsubs[si]):
                    for j in range(nsubs[sj]):
                        if sub0[si] + i != sub0[sj] + j:
                            key = f"xs{si+1}s{i+1}s{sj+1}s{j+1}"
                            bias[key] = -float(x_sum[sub0[si] + i, sub0[sj] + j])

        # Chi (s) coupling
        s_prev = np.loadtxt(analysis_dir / "s_prev.dat")
        s = np.loadtxt(analysis_dir / "s.dat")
        s_sum = s_prev + s
        # Validate for NaN/Inf - use previous values if invalid
        if not np.all(np.isfinite(s_sum)):
            logger.warning(f"NaN/Inf in s_sum at step {step}, using previous values")
            s_sum = s_prev.copy()
        np.savetxt(analysis_dir / "s_sum.dat", _clean_negzero(s_sum), fmt=" %10.5f")

        for si in range(len(nsubs)):
            for sj in range(len(nsubs)):
                for i in range(nsubs[si]):
                    for j in range(nsubs[sj]):
                        if sub0[si] + i != sub0[sj] + j:
                            key = f"ss{si+1}s{i+1}s{sj+1}s{j+1}"
                            bias[key] = -float(s_sum[sub0[si] + i, sub0[sj] + j])

        # Store full arrays
        bias["b"] = b_sum.tolist()
        bias["c"] = c_sum.tolist()
        bias["x"] = x_sum.tolist()
        bias["s"] = s_sum.tolist()

        # Write bias dictionary as YAML
        fp.write('bias_string="""\n')
        yaml.dump(bias, fp)
        fp.write('"""\n')
        fp.write("bias=yaml.load(bias_string,Loader=yaml.Loader)\n")
        fp.write("bias['b']=np.array(bias['b'])\n")
        fp.write("bias['c']=np.array(bias['c'])\n")
        fp.write("bias['x']=np.array(bias['x'])\n")
        fp.write("bias['s']=np.array(bias['s'])\n\n")

        # Write alf_info dictionary
        alf_info_dict = alf_info.to_dict()
        fp.write('alf_info_string="""\n')
        yaml.dump(alf_info_dict, fp)
        fp.write('"""\n')
        fp.write("alf_info=yaml.load(alf_info_string,Loader=yaml.Loader)\n")
        fp.write("alf_info['nsubs']=np.array(alf_info['nsubs'])\n\n")

        # Write system parameters
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

        # Minimize flag
        fp.write(f"minimizeflag={minimize}\n")

    # Also write .inp format for CHARMM compatibility
    inp_file = dirs["variables"] / f"variables_{step:03d}.inp"
    with open(inp_file, "w") as fp:
        fp.write(f"* Variables for step {step}\n")
        fp.write("*\n\n")

        # Write bias parameters (lams, cs, xs, ss)
        for key, value in sorted(bias.items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # Final validation - sanitize any NaN/Inf before writing to CHARMM
                if not np.isfinite(value):
                    logger.warning(f"Invalid value for {key}: {value}, replacing with 0.0")
                    value = 0.0
                fp.write(f"set {key} = {_fmtval(value):12.5f}\n")

        fp.write("\n")
        fp.write(f'set sysname = "{name}"\n')
        fp.write(f"set nnodes = {nnodes}\n")
        fp.write(f"set nreps = {nreps}\n")
        fp.write(f"set ncentral = {ncentral}\n")
        fp.write(f"set nblocks = {nblocks}\n")
        fp.write(f"set nsites = {len(nsubs)}\n")
        for i in range(len(nsubs)):
            fp.write(f"set nsubs{i+1} = {nsubs[i]}\n")
        fp.write(f"set temp = {temp}\n")
        fp.write(f"set minimize = {1 if minimize else 0}\n")

    return output_file


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

        # Store full arrays
        bias["b"] = b_sum.tolist()
        bias["c"] = c_sum.tolist()
        bias["x"] = x_sum.tolist()
        bias["s"] = s_sum.tolist()

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
            fp.write("bias['s']=np.array(bias['s'])\n\n")

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
    nblocks = alf_info.nblocks
    nsubs = alf_info.nsubs
    nreps = alf_info.nreps
    ncentral = alf_info.ncentral

    NF = end_cycle - start_cycle + 1

    def load_shift_file(analysis_dir: str, filename: str, default_value=0.0):
        """Load shift file from analysis_dir/nbshift or ../nbshift."""
        local_path = os.path.join(analysis_dir, "nbshift", filename)
        if os.path.exists(local_path):
            return np.loadtxt(local_path)
        fallback_path = os.path.join("../nbshift", filename)
        if os.path.exists(fallback_path):
            return np.loadtxt(fallback_path)
        return default_value

    Lambda = []
    b = []
    c = []
    x = []
    s = []

    for i in range(NF):
        analysis_dir = f"../analysis{start_cycle + i}"
        data_dir = os.path.join(analysis_dir, "data")

        if not os.path.isdir(data_dir):
            print(f"Warning: Directory {data_dir} not found")
            continue

        # Load shift files
        b_shift = load_shift_file(analysis_dir, "b_shift.dat")
        c_shift = load_shift_file(analysis_dir, "c_shift.dat")
        x_shift = load_shift_file(analysis_dir, "x_shift.dat")
        s_shift = load_shift_file(analysis_dir, "s_shift.dat")

        b_fix_shift = load_shift_file(analysis_dir, "b_fix_shift.dat", 0.0)
        c_fix_shift = load_shift_file(analysis_dir, "c_fix_shift.dat", 0.0)
        x_fix_shift = load_shift_file(analysis_dir, "x_fix_shift.dat", 0.0)
        s_fix_shift = load_shift_file(analysis_dir, "s_fix_shift.dat", 0.0)

        from cphmd.utils.lambda_io import find_lambda_files, read_lambda_values
        lambda_fpaths = find_lambda_files(Path(data_dir))

        for fpath in lambda_fpaths:
            lambda_file = fpath.name
            try:
                # read_lambda_values returns lambda-only (no time column)
                Lambda.append(read_lambda_values(fpath)[(skipE - 1)::skipE, :])

                # Extract j and k from filename (Lambda.j.k.{parquet,dat})
                j, k = map(int, lambda_file.split(".")[1:3])
                b_old = np.loadtxt(os.path.join(analysis_dir, "b_prev.dat"))
                b.append(b_old + b_shift * (k - ncentral) + b_fix_shift)
                c_old = np.loadtxt(os.path.join(analysis_dir, "c_prev.dat"))
                c.append(c_old + c_shift * (k - ncentral) + c_fix_shift)
                x_old = np.loadtxt(os.path.join(analysis_dir, "x_prev.dat"))
                x.append(x_old + x_shift * (k - ncentral) + x_fix_shift)
                s_old = np.loadtxt(os.path.join(analysis_dir, "s_prev.dat"))
                s.append(s_old + s_shift * (k - ncentral) + s_fix_shift)
            except Exception as e:
                print(f"Error loading file {fpath}: {e}")

    os.makedirs("Lambda", exist_ok=True)
    os.makedirs("Energy", exist_ok=True)

    total_simulations = len(Lambda)
    if total_simulations == 0:
        print("Error: No Lambda files found.")
        return 0

    # Compute cross-simulation bias energies
    E = [[] for _ in range(total_simulations)]
    for i in range(total_simulations):
        for j_idx in range(total_simulations):
            bi, ci, xi, si = b[i], c[i], x[i], s[i]
            Lj = Lambda[j_idx]
            Eij = np.reshape(np.dot(Lj, -bi), (-1, 1))
            Eij += np.sum(np.dot(Lj, -ci) * Lj, axis=1, keepdims=True)
            Eij += np.sum(np.dot(1 - np.exp(-OMEGA_DECAY * Lj), -xi) * Lj, axis=1, keepdims=True)
            Eij += np.sum(np.dot(Lj / (Lj + CHI_OFFSET), -si) * Lj, axis=1, keepdims=True)
            E[i].append(Eij)

    for i in range(total_simulations):
        Ei = E[total_simulations - 1][i]
        for j_idx in range(total_simulations):
            Ei = np.concatenate((Ei, E[j_idx][i]), axis=1)
        np.savetxt(f"Energy/ESim{i + 1}.dat", Ei, fmt="%12.5f")

    for i in range(total_simulations):
        np.savetxt(f"Lambda/Lambda{i + 1}.dat", Lambda[i], fmt="%10.6f")

    # Create G_imp shift information
    os.makedirs("G_imp_shifts", exist_ok=True)

    simulation_jk_map = {}
    sim_idx = 0
    for i in range(NF):
        analysis_dir = f"../analysis{start_cycle + i}"
        data_dir = os.path.join(analysis_dir, "data")
        if not os.path.isdir(data_dir):
            continue
        lambda_fpaths2 = find_lambda_files(Path(data_dir))
        for fpath2 in lambda_fpaths2:
            try:
                j, k = map(int, fpath2.name.split(".")[1:3])
                simulation_jk_map[sim_idx] = (j, k, i)
                sim_idx += 1
            except (ValueError, IndexError):
                simulation_jk_map[sim_idx] = (1, sim_idx, i)
                sim_idx += 1

    for sim_idx in range(total_simulations):
        if sim_idx not in simulation_jk_map:
            continue
        j, k, analysis_idx = simulation_jk_map[sim_idx]
        analysis_dir = f"../analysis{start_cycle + analysis_idx}"
        try:
            b_shift_arr = np.atleast_1d(load_shift_file(analysis_dir, "b_shift.dat", 0.0))
            b_fix_shift_arr = np.atleast_1d(load_shift_file(analysis_dir, "b_fix_shift.dat", 0.0))

            if b_shift_arr.size > 1:
                num_blocks = len(b_shift_arr)
            else:
                b_shift_arr = np.full(nblocks, float(b_shift_arr.flat[0]))
                b_fix_shift_arr = np.full(nblocks, float(b_fix_shift_arr.flat[0]))
                num_blocks = nblocks

            with open(f"G_imp_shifts/shifts_sim{sim_idx + 1}.dat", "w") as f:
                f.write(f"# G_imp shift information for simulation {sim_idx + 1}\n")
                f.write(f"# j: {j}, k: {k}, ncentral: {ncentral}\n")
                for block_idx in range(num_blocks):
                    total_shift = float(b_fix_shift_arr[block_idx]) + float(b_shift_arr[block_idx]) * (k - ncentral)
                    f.write(f"{total_shift:.6f}\n")
        except Exception as e:
            print(f"Warning: Could not create G_imp shifts for simulation {sim_idx + 1}: {e}")

    return total_simulations


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
    import os

    alf_info = ensure_alf_info(alf_info)
    nblocks = alf_info.nblocks
    ncentral = alf_info.ncentral

    NF = end_cycle - start_cycle + 1

    def load_shift_file(analysis_dir: str, filename: str, default_value=0.0):
        local_path = os.path.join(analysis_dir, "nbshift", filename)
        if os.path.exists(local_path):
            return np.loadtxt(local_path)
        fallback_path = os.path.join("../nbshift", filename)
        if os.path.exists(fallback_path):
            return np.loadtxt(fallback_path)
        return default_value

    Lambda: list[np.ndarray] = []
    b_list: list[np.ndarray] = []
    c_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    s_list: list[np.ndarray] = []
    jk_map: list[tuple[int, int, int]] = []  # (j, k, analysis_idx) per sim

    from cphmd.utils.lambda_io import find_lambda_files, read_lambda_values

    for i in range(NF):
        analysis_dir = f"../analysis{start_cycle + i}"
        data_dir = os.path.join(analysis_dir, "data")

        if not os.path.isdir(data_dir):
            print(f"Warning: Directory {data_dir} not found")
            continue

        b_shift = load_shift_file(analysis_dir, "b_shift.dat")
        c_shift = load_shift_file(analysis_dir, "c_shift.dat")
        x_shift = load_shift_file(analysis_dir, "x_shift.dat")
        s_shift = load_shift_file(analysis_dir, "s_shift.dat")
        b_fix_shift = load_shift_file(analysis_dir, "b_fix_shift.dat", 0.0)
        c_fix_shift = load_shift_file(analysis_dir, "c_fix_shift.dat", 0.0)
        x_fix_shift = load_shift_file(analysis_dir, "x_fix_shift.dat", 0.0)
        s_fix_shift = load_shift_file(analysis_dir, "s_fix_shift.dat", 0.0)

        lambda_fpaths = find_lambda_files(Path(data_dir))

        for fpath in lambda_fpaths:
            try:
                Lambda.append(read_lambda_values(fpath)[(skipE - 1)::skipE, :])

                j, k = map(int, fpath.name.split(".")[1:3])
                jk_map.append((j, k, i))
                b_old = np.loadtxt(os.path.join(analysis_dir, "b_prev.dat"))
                b_list.append(b_old + b_shift * (k - ncentral) + b_fix_shift)
                c_old = np.loadtxt(os.path.join(analysis_dir, "c_prev.dat"))
                c_list.append(c_old + c_shift * (k - ncentral) + c_fix_shift)
                x_old = np.loadtxt(os.path.join(analysis_dir, "x_prev.dat"))
                x_list.append(x_old + x_shift * (k - ncentral) + x_fix_shift)
                s_old = np.loadtxt(os.path.join(analysis_dir, "s_prev.dat"))
                s_list.append(s_old + s_shift * (k - ncentral) + s_fix_shift)
            except Exception as e:
                print(f"Error loading file {fpath}: {e}")

    total_simulations = len(Lambda)
    if total_simulations == 0:
        print("Error: No Lambda files found.")
        return [], [], None, 0

    # Compute cross-simulation bias energies: E[i][j] = energy of sim j's
    # lambda under sim i's parameters
    energy_matrix: list[list[np.ndarray]] = [[] for _ in range(total_simulations)]
    for i in range(total_simulations):
        for j_idx in range(total_simulations):
            bi, ci, xi, si = b_list[i], c_list[i], x_list[i], s_list[i]
            Lj = Lambda[j_idx]
            Eij = np.reshape(np.dot(Lj, -bi), (-1, 1))
            Eij += np.sum(np.dot(Lj, -ci) * Lj, axis=1, keepdims=True)
            Eij += np.sum(
                np.dot(1 - np.exp(-OMEGA_DECAY * Lj), -xi) * Lj,
                axis=1, keepdims=True,
            )
            Eij += np.sum(
                np.dot(Lj / (Lj + CHI_OFFSET), -si) * Lj,
                axis=1, keepdims=True,
            )
            energy_matrix[i].append(Eij)

    # Compute G_imp shift data as array [total_simulations, nblocks]
    gshift_data = np.zeros((total_simulations, nblocks), dtype=np.float64)
    for sim_idx, (j, k, analysis_idx) in enumerate(jk_map):
        analysis_dir = f"../analysis{start_cycle + analysis_idx}"
        try:
            b_shift_arr = np.atleast_1d(
                load_shift_file(analysis_dir, "b_shift.dat", 0.0)
            )
            b_fix_shift_arr = np.atleast_1d(
                load_shift_file(analysis_dir, "b_fix_shift.dat", 0.0)
            )
            if b_shift_arr.size == 1:
                b_shift_arr = np.full(nblocks, float(b_shift_arr.flat[0]))
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

    return Lambda, energy_matrix, gshift_data, total_simulations


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
    - Omega (x-term): -λᵀ·(1-exp(-OMEGA_DECAY·λ))·x·λ
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
    omega_factor = 1.0 - np.exp(-OMEGA_DECAY * lam)
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
        alf_info: ALF simulation information (unused, for API compatibility).
        output_path: Path to output .parquet file.
        input_files: List of input binary lambda file paths.
    """
    from cphmd.utils.lambda_io import read_lambda_binary, write_lambda_parquet

    all_data = []
    for input_file in input_files:
        input_path = Path(input_file)
        if input_path.exists():
            data, _ = read_lambda_binary(str(input_path))
            all_data.append(data)

    if all_data:
        combined = np.vstack(all_data)
        write_lambda_parquet(output_path, combined)
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
    cutc2: float = 2.0,
    cutx2: float = 0.5,
    cuts2: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    # Count expected parameters
    nparm = 0
    for isite in range(len(nsubs)):
        n1 = nsubs[isite]
        n2 = nsubs[isite] * (nsubs[isite] - 1) // 2
        for jsite in range(isite, len(nsubs)):
            n3 = nsubs[isite] * nsubs[jsite]
            if isite == jsite:
                nparm += n1 + 5 * n2
            elif ms == 1:
                nparm += 5 * n3
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
                        cutlist[n0:n0 + 2] = cutx
                        n0 += 2
                        cutlist[n0:n0 + 2] = cuts
                        n0 += 2
            elif ms > 0:
                for i in range(nsubs[isite]):
                    for j in range(nsubs[jsite]):
                        cutlist[n0:n0 + 1] = cutc2
                        n0 += 1
                        if ms == 1:
                            cutlist[n0:n0 + 2] = cutx2
                            n0 += 2
                            cutlist[n0:n0 + 2] = cuts2
                            n0 += 2

    # Load LMALF output
    coeff = np.loadtxt("OUT.dat")

    # Apply per-parameter clipping to prevent large changes
    # This matches WHAM's behavior better than global scaling:
    # - Global scaling: if ONE parameter is too large, ALL get reduced
    # - Per-parameter clipping: each parameter is independently limited
    #
    # The cutoff acts as a "speed limit" for each parameter type,
    # preventing oscillation from overcorrection.
    n_clipped = 0
    max_ratio = 0.0
    for i in range(n0):
        limit = 1.5 * cutlist[i]
        ratio = abs(coeff[i] / cutlist[i]) if cutlist[i] > 0 else 0
        max_ratio = max(max_ratio, ratio)
        if abs(coeff[i]) > limit:
            coeff[i] = np.clip(coeff[i], -limit, limit)
            n_clipped += 1

    print(f"LMALF: max_ratio={max_ratio:.2f}, clipped {n_clipped}/{n0} parameters")

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
                        x[iblock + i, jblock + j] = coeff[ind]
                        ind += 1
                        x[jblock + j, iblock + i] = coeff[ind]
                        ind += 1
                        s[iblock + i, jblock + j] = coeff[ind]
                        ind += 1
                        s[jblock + j, iblock + i] = coeff[ind]
                        ind += 1
            elif ms > 0:
                for i in range(nsubs[isite]):
                    for j in range(nsubs[jsite]):
                        c[iblock + i, jblock + j] = coeff[ind]
                        ind += 1
                        if ms == 1:
                            x[iblock + i, jblock + j] = coeff[ind]
                            ind += 1
                            x[jblock + j, iblock + i] = coeff[ind]
                            ind += 1
                            s[iblock + i, jblock + j] = coeff[ind]
                            ind += 1
                            s[jblock + j, iblock + i] = coeff[ind]
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

    print("LMALF: Wrote b.dat, c.dat, x.dat, s.dat")

    return b, c, x, s
