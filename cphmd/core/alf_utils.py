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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ALF bias potential constants
OMEGA_DECAY = 5.56       # Exponential decay for omega bias term
CHI_OFFSET = 0.017       # Offset for chi sigmoid bias term


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
        "runs": work_dir / "runs",
        "analysis": work_dir / "analysis",
        "variables": work_dir / "variables",
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

    Creates analysis/0/ directory with initial bias parameter files
    and generates variables/1.py for the first ALF cycle.

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
        Path to analysis/0/ directory.
    """
    work_dir = Path(work_dir)
    alf_info = ensure_alf_info(alf_info)
    nblocks = alf_info.nblocks

    # Ensure directory structure exists
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

    # Create analysis/0 directory (initial state)
    analysis0 = dirs["analysis"] / "0"
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

    # Generate variables/1.py
    set_vars(work_dir, alf_info, step=1, minimize=minimize)

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
        np.savetxt(analysis_dir / "b_sum.dat", b_sum, fmt=" %7.3f")

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
        np.savetxt(analysis_dir / "c_sum.dat", c_sum, fmt=" %7.3f")

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
        np.savetxt(analysis_dir / "x_sum.dat", x_sum, fmt=" %7.3f")

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
        np.savetxt(analysis_dir / "s_sum.dat", s_sum, fmt=" %7.3f")

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
                fp.write(f"set {key} = {value:10.3f}\n")

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


def get_energy(
    work_dir: str | Path,
    alf_info: ALFInfo | dict,
    start_iter: int,
    end_iter: int,
    skip_frames: int = 1
) -> dict[str, np.ndarray]:
    """Compute bias energies from lambda trajectories for WHAM.

    Reads lambda trajectories from multiple ALF iterations and computes
    the bias energies needed for WHAM reweighting.

    Args:
        work_dir: Working directory for ALF simulation.
        alf_info: ALF simulation information.
        start_iter: First iteration to include (inclusive).
        end_iter: Last iteration to include (inclusive).
        skip_frames: Only analyze every Nth frame.

    Returns:
        Dictionary with:
            - 'Lambda': Concatenated lambda trajectories
            - 'Energy': Bias energy matrix for WHAM
            - 'b', 'c', 'x', 's': Bias parameters per iteration
    """
    work_dir = Path(work_dir)
    alf_info = ensure_alf_info(alf_info)

    nblocks = alf_info.nblocks
    nsubs = alf_info.nsubs
    nreps = alf_info.nreps

    n_iters = end_iter - start_iter + 1
    analysis_base = work_dir / "analysis"

    Lambda_all = []
    b_all = []
    c_all = []
    x_all = []
    s_all = []

    for i in range(n_iters):
        iter_idx = start_iter + i
        analysis_dir = analysis_base / str(iter_idx)
        data_dir = analysis_dir / "data"

        if not data_dir.is_dir():
            print(f"Warning: {data_dir} not found, skipping")
            continue

        # Load bias parameters
        b_prev = np.loadtxt(analysis_dir / "b_prev.dat").reshape(1, -1)
        c_prev = np.loadtxt(analysis_dir / "c_prev.dat")
        x_prev = np.loadtxt(analysis_dir / "x_prev.dat")
        s_prev = np.loadtxt(analysis_dir / "s_prev.dat")

        b_all.append(b_prev)
        c_all.append(c_prev)
        x_all.append(x_prev)
        s_all.append(s_prev)

        # Load lambda trajectories (files are Lambda.{kk}.{rep}.dat where kk=0 for phases 1-2)
        for rep in range(nreps):
            # Try phase 1/2 naming (Lambda.0.{rep}.dat) first
            lambda_file = data_dir / f"Lambda.0.{rep}.dat"
            if lambda_file.exists():
                Lambda = np.loadtxt(lambda_file)
                Lambda_all.append(Lambda[::skip_frames])
            else:
                # Fallback to phase 3 naming or other patterns
                alt_file = data_dir / f"Lambda.{rep}.dat"
                if alt_file.exists():
                    Lambda = np.loadtxt(alt_file)
                    Lambda_all.append(Lambda[::skip_frames])

    if not Lambda_all:
        raise ValueError("No lambda data found")

    # Concatenate all lambda data
    Lambda_concat = np.vstack(Lambda_all)

    # Create output directories in current analysis dir
    current_analysis = analysis_base / str(end_iter)
    (current_analysis / "Lambda").mkdir(exist_ok=True)
    (current_analysis / "Energy").mkdir(exist_ok=True)

    # Save concatenated lambda (one file per iteration, 1-indexed for WHAM)
    for i in range(n_iters):
        if i < len(Lambda_all):
            np.savetxt(
                current_analysis / "Lambda" / f"Lambda{i + 1}.dat",
                Lambda_all[i],
                fmt="%10.6f"
            )

    # Compute and save cross-simulation bias energies for WHAM
    # ESim{i+1}.dat contains the bias energy of ALL frames using biases from sim i
    for sim_idx in range(len(b_all)):
        b = b_all[sim_idx]
        c = c_all[sim_idx]
        x = x_all[sim_idx]
        s = s_all[sim_idx]

        # Compute energy for all frames using this simulation's biases
        energies = []
        for frame_lam in Lambda_concat:
            # Skip first column (time) if lambda has more columns than nblocks
            lam = frame_lam[1:] if len(frame_lam) > nblocks else frame_lam
            E = compute_bias_energy(lam, b, c, x, s)
            energies.append(E)

        # WHAM expects ESim files numbered 1-N (not 0-indexed)
        np.savetxt(
            current_analysis / "Energy" / f"ESim{sim_idx + 1}.dat",
            np.array(energies).reshape(-1, 1),
            fmt="%12.6f"
        )

    return {
        "Lambda": Lambda_concat,
        "b": b_all,
        "c": c_all,
        "x": x_all,
        "s": s_all,
    }


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
    - Omega: -λᵀ·(1-exp(-5.56λ))·x·λ
    - Chi: -λᵀ·(λ/(λ+0.017))·s·λ

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


def write_lambda_text(output_path: str | Path, data: np.ndarray) -> None:
    """Write lambda data in human-readable text format.

    Args:
        output_path: Path to output file.
        data: Lambda trajectory data (nframes x nblocks).
    """
    np.savetxt(output_path, data, fmt="%10.6f")


def convert_lambda_binary_to_text(
    alf_info: ALFInfo | dict,
    output_path: str | Path,
    input_files: list[str | Path],
) -> None:
    """Convert binary lambda file(s) to human-readable text format.

    This function replaces ALF's GetLambda.GetLambda function.

    Args:
        alf_info: ALF simulation information (unused, for compatibility).
        output_path: Path to output text file.
        input_files: List of input binary lambda file paths.

    Note:
        Uses cphmd.utils.lambda_io.read_lambda_binary for file reading.
    """
    from cphmd.utils.lambda_io import read_lambda_binary

    # Concatenate all input files
    all_data = []
    for input_file in input_files:
        input_path = Path(input_file)
        if input_path.exists():
            data, _ = read_lambda_binary(str(input_path))
            all_data.append(data)

    if all_data:
        combined = np.vstack(all_data)
        write_lambda_text(output_path, combined)
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
    np.savetxt("b.dat", b, fmt=" %7.2f")
    np.savetxt("c.dat", c, fmt=" %7.2f")
    np.savetxt("x.dat", x, fmt=" %7.2f")
    np.savetxt("s.dat", s, fmt=" %7.2f")

    print(f"LMALF: Wrote b.dat, c.dat, x.dat, s.dat")

    return b, c, x, s