"""
Energy profile analysis for ALF/CpHMD simulations.

This module provides tools for visualizing and analyzing the evolution of
bias energy landscapes during ALF optimization. It computes energy profiles
on a simplex grid and tracks convergence via RMSD between iterations.

Key Features:
- Compute bias energy from b, c, s, x parameters
- Generate simplex grid for multi-state systems
- Track RMSD convergence between iterations
- Create 2D/3D visualizations with matplotlib/plotly
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from cphmd.analysis.plot_style import apply_pub_style, clean_axes, get_state_colors, savefig
from cphmd.core.bias_constants import CHI_OFFSET, OMEGA_DECAY

# Optional plotly import for interactive plots
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


@dataclass
class EnergyProfileConfig:
    """Configuration for energy profile analysis.

    Attributes:
        input_folder: Path to folder containing analysis directories
        num_points: Number of grid points per dimension for simplex
        output_dir: Directory for output plots
        create_animation: Whether to create video animation (3D only)
        plot_format: Output format for static plots
    """
    input_folder: str | Path
    num_points: int = 100
    output_dir: str | Path = "plots"
    create_animation: bool = False
    plot_format: Literal["png", "pdf", "svg"] = "png"

    def __post_init__(self):
        self.input_folder = Path(self.input_folder)
        self.output_dir = Path(self.output_dir)


@dataclass
class EnergyProfileResult:
    """Results from energy profile analysis.

    Attributes:
        iterations: List of iteration numbers analyzed
        rmsd_values: RMSD between consecutive iterations
        energy_profiles: Dict mapping iteration -> energy values on simplex
        simplex_grid: Grid points on the simplex
        n_states: Number of states (dimension of lambda space)
    """
    iterations: list[int]
    rmsd_values: list[float]
    energy_profiles: dict[int, np.ndarray]
    simplex_grid: np.ndarray
    n_states: int
    rmsd_plot: Path | None = None
    profile_plot: Path | None = None


def generate_simplex_grid(n_states: int, num_points: int) -> np.ndarray:
    """Generate grid points on the (n-1)-simplex.

    The simplex is defined by lambda_i >= 0 and sum(lambda) = 1.

    Args:
        n_states: Number of states (dimensions)
        num_points: Number of points per dimension

    Returns:
        Array of shape (M, n_states) with M points on the simplex
    """
    grids = [np.linspace(0, 1, num_points) for _ in range(n_states)]
    mesh = np.array(list(product(*grids)))
    # Keep only points on the simplex (sum = 1)
    simplex_points = mesh[np.isclose(np.sum(mesh, axis=1), 1, atol=1e-6)]
    return simplex_points


def energy_E_b(lambda_vec: np.ndarray, b: np.ndarray) -> float:
    """Compute linear bias energy E_b = -sum(lambda_i * b_i)."""
    return -np.dot(lambda_vec, b)


def energy_E_c(lambda_vec: np.ndarray, c: np.ndarray) -> float:
    """Compute quadratic coupling energy E_c."""
    n = len(lambda_vec)
    E_c = 0.0
    for i in range(n):
        for j in range(n):
            E_c += (c[i, j] + c[j, i]) * lambda_vec[i] * lambda_vec[j]
    return E_c


def energy_E_s(lambda_vec: np.ndarray, s: np.ndarray, epsilon: float = CHI_OFFSET) -> float:
    """Compute endpoint bias energy E_s with regularization."""
    n = len(lambda_vec)
    E_s = 0.0
    for i in range(n):
        for j in range(n):
            term_i = s[i, j] / (lambda_vec[i] + epsilon)
            term_j = s[j, i] / (lambda_vec[j] + epsilon)
            E_s += (term_i + term_j) * lambda_vec[i] * lambda_vec[j]
    return E_s


def energy_E_x(lambda_vec: np.ndarray, x: np.ndarray, alpha: float = OMEGA_DECAY) -> float:
    """Compute skew bias energy E_x with exponential form."""
    n = len(lambda_vec)
    E_x = 0.0
    for i in range(n):
        for j in range(n):
            E_x += x[i, j] * lambda_vec[j] * (1 - np.exp(-alpha * lambda_vec[i]))
            E_x += x[j, i] * lambda_vec[i] * (1 - np.exp(-alpha * lambda_vec[j]))
    return E_x


def total_energy(
    lambda_vec: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    s: np.ndarray,
    x: np.ndarray
) -> float:
    """Compute total bias energy E = E_b + E_c + E_s + E_x."""
    return (
        energy_E_b(lambda_vec, b) +
        energy_E_c(lambda_vec, c) +
        energy_E_s(lambda_vec, s) +
        energy_E_x(lambda_vec, x)
    )


def _load_bias_parameters(analysis_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load b, c, s, x bias parameters from analysis directory."""
    b = np.loadtxt(analysis_dir / "b_prev.dat")
    c = np.loadtxt(analysis_dir / "c_prev.dat")
    s = np.loadtxt(analysis_dir / "s_prev.dat")
    x = np.loadtxt(analysis_dir / "x_prev.dat")

    # Ensure b is 1D
    if b.ndim == 0:
        b = np.array([float(b)])

    return b, c, s, x


def _calculate_energy_profile(
    analysis_dir: Path,
    simplex_grid: np.ndarray
) -> tuple[int, np.ndarray]:
    """Calculate energy profile for a single iteration."""
    # Extract iteration number from directory name
    itt = int(analysis_dir.name.replace("analysis", ""))

    # Load parameters
    b, c, s, x = _load_bias_parameters(analysis_dir)

    # Calculate energy at each grid point
    energies = np.array([
        total_energy(lv, b, c, s, x) for lv in simplex_grid
    ])

    return itt, energies


def analyze_energy_profiles(config: EnergyProfileConfig) -> EnergyProfileResult:
    """Analyze energy profiles across ALF iterations.

    Args:
        config: Configuration for analysis

    Returns:
        EnergyProfileResult with convergence data and plots
    """
    input_folder = config.input_folder

    # Find analysis directories (support both analysis/0 and analysis0 formats)
    analysis_dirs = []

    # Check for analysis/N format (e.g., analysis/0, analysis/1, ...)
    analysis_parent = input_folder / "analysis"
    if analysis_parent.exists() and analysis_parent.is_dir():
        analysis_dirs = sorted([
            d for d in analysis_parent.iterdir()
            if d.is_dir() and d.name.isdigit()
        ], key=lambda d: int(d.name))

    # Fallback: check for analysisN format (e.g., analysis0, analysis1, ...)
    if not analysis_dirs:
        analysis_dirs = sorted([
            d for d in input_folder.iterdir()
            if d.is_dir() and d.name.startswith("analysis") and d.name[8:].isdigit()
        ], key=lambda d: int(d.name[8:]))

    if not analysis_dirs:
        raise ValueError(f"No analysis directories found in {input_folder}")

    # Determine number of states from first directory
    b_sample, _, _, _ = _load_bias_parameters(analysis_dirs[0])
    n_states = len(b_sample)

    # Generate simplex grid
    simplex_grid = generate_simplex_grid(n_states, config.num_points)
    print(f"Generated {len(simplex_grid)} simplex points for {n_states} states")

    # Calculate energy profiles (parallel)
    energy_profiles = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(_calculate_energy_profile, d, simplex_grid): d
            for d in analysis_dirs
        }
        for future in as_completed(futures):
            itt, energies = future.result()
            energy_profiles[itt] = energies
            print(f"  Processed analysis{itt}")

    # Sort iterations
    iterations = sorted(energy_profiles.keys())

    # Calculate RMSD between consecutive iterations
    rmsd_values = []
    for i in range(1, len(iterations)):
        prev = energy_profiles[iterations[i - 1]]
        curr = energy_profiles[iterations[i]]
        rmsd = np.sqrt(np.mean((curr - prev) ** 2))
        rmsd_values.append(rmsd)

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Plot RMSD convergence
    rmsd_plot = _plot_rmsd_convergence(
        iterations[1:], rmsd_values, config.output_dir, config.plot_format
    )

    # Create energy profile visualization
    profile_plot = None
    if n_states == 2:
        profile_plot = _plot_2d_profiles(
            simplex_grid, energy_profiles, iterations,
            config.output_dir, config.plot_format
        )
    elif n_states == 3:
        profile_plot = _plot_3d_profiles(
            simplex_grid, energy_profiles, iterations,
            config.output_dir, config.plot_format, config.create_animation
        )

    return EnergyProfileResult(
        iterations=iterations,
        rmsd_values=rmsd_values,
        energy_profiles=energy_profiles,
        simplex_grid=simplex_grid,
        n_states=n_states,
        rmsd_plot=rmsd_plot,
        profile_plot=profile_plot,
    )


def _plot_rmsd_convergence(
    iterations: list[int],
    rmsd_values: list[float],
    output_dir: Path,
    fmt: str
) -> Path:
    """Plot RMSD convergence over iterations with semilog scale.

    Energy decay is shown on log scale to visualize exponential convergence.
    """
    apply_pub_style()

    colors = plt.cm.Set1.colors

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(
        iterations, rmsd_values,
        marker='o', color=colors[0], linewidth=2, markersize=7,
        markeredgecolor='black', markeredgewidth=0.5,
    )
    from matplotlib.ticker import MaxNLocator

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("RMSD (kcal/mol)", fontsize=12)
    ax.set_title("Energy Profile RMSD Convergence", fontsize=14, fontweight='bold')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which='both', linestyle='--', alpha=0.3)

    # Annotate final value
    if rmsd_values:
        final_rmsd = rmsd_values[-1]
        ax.annotate(
            f"Final = {final_rmsd:.3f}",
            xy=(0.97, 0.95), xycoords="axes fraction",
            fontsize=11, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=colors[0], alpha=0.9),
        )

    clean_axes(ax)

    output_path = output_dir / f"rmsd_convergence.{fmt}"
    return savefig(fig, output_path)


def _plot_2d_profiles(
    simplex_grid: np.ndarray,
    energy_profiles: dict[int, np.ndarray],
    iterations: list[int],
    output_dir: Path,
    fmt: str
) -> Path:
    """Plot 2D energy profiles (for 2-state systems)."""
    # Create static plot with final iteration
    final_itt = iterations[-1]
    energies = energy_profiles[final_itt]

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        simplex_grid[:, 0], energies,
        c=energies, cmap="viridis", s=20
    )
    ax.set_xlabel("λ₁", fontsize=12)
    ax.set_ylabel("Energy (kcal/mol)", fontsize=12)
    ax.set_title(f"Energy Profile (Iteration {final_itt})", fontsize=14)
    plt.colorbar(scatter, label="Energy")

    output_path = output_dir / f"energy_profile_2d.{fmt}"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Create interactive HTML if plotly available
    if HAS_PLOTLY:
        _create_2d_interactive(simplex_grid, energy_profiles, iterations, output_dir)

    return output_path


def _plot_3d_profiles(
    simplex_grid: np.ndarray,
    energy_profiles: dict[int, np.ndarray],
    iterations: list[int],
    output_dir: Path,
    fmt: str,
    create_animation: bool
) -> Path:
    """Plot 3D energy profiles (for 3-state systems)."""
    final_itt = iterations[-1]
    energies = energy_profiles[final_itt]

    # Transform to 2D projection for visualization
    x_proj = simplex_grid[:, 0] - simplex_grid[:, 1]  # λ1 - λ2
    y_proj = simplex_grid[:, 0] - simplex_grid[:, 2]  # λ1 - λ3

    # Static matplotlib plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x_proj, y_proj, energies, c=energies, cmap="viridis", s=10)
    ax.set_xlabel("λ₁ - λ₂")
    ax.set_ylabel("λ₁ - λ₃")
    ax.set_zlabel("Energy (kcal/mol)")
    ax.set_title(f"Energy Profile (Iteration {final_itt})")
    plt.colorbar(scatter, shrink=0.6, label="Energy")

    output_path = output_dir / f"energy_profile_3d.{fmt}"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Create interactive HTML if plotly available
    if HAS_PLOTLY:
        _create_3d_interactive(simplex_grid, energy_profiles, iterations, output_dir)

    return output_path


def _create_2d_interactive(
    simplex_grid: np.ndarray,
    energy_profiles: dict[int, np.ndarray],
    iterations: list[int],
    output_dir: Path
):
    """Create interactive 2D plot with slider (requires plotly)."""
    fig = go.Figure()

    for itt in iterations:
        energies = energy_profiles[itt]
        fig.add_trace(go.Scatter(
            x=simplex_grid[:, 0],
            y=energies,
            mode='markers',
            marker=dict(color=energies, colorscale='Viridis', showscale=True),
            name=f'Iteration {itt}',
            visible=False
        ))

    fig.data[0].visible = True

    steps = []
    for i, itt in enumerate(iterations):
        step = dict(
            method='update',
            args=[{'visible': [j == i for j in range(len(iterations))]}],
            label=str(itt)
        )
        steps.append(step)

    fig.update_layout(
        sliders=[dict(active=0, steps=steps, currentvalue={"prefix": "Iteration: "})],
        xaxis_title="λ₁",
        yaxis_title="Energy (kcal/mol)",
        title="Energy Profile Evolution"
    )

    fig.write_html(output_dir / "energy_profile_2d.html")


def _create_3d_interactive(
    simplex_grid: np.ndarray,
    energy_profiles: dict[int, np.ndarray],
    iterations: list[int],
    output_dir: Path
):
    """Create interactive 3D plot with slider (requires plotly)."""
    x_proj = simplex_grid[:, 0] - simplex_grid[:, 1]
    y_proj = simplex_grid[:, 0] - simplex_grid[:, 2]

    # Get energy range for consistent coloring
    all_energies = np.concatenate(list(energy_profiles.values()))
    q1, q3 = np.percentile(all_energies, [5, 95])

    fig = go.Figure()

    for itt in iterations:
        energies = energy_profiles[itt]
        fig.add_trace(go.Scatter3d(
            x=x_proj, y=y_proj, z=energies,
            mode='markers',
            marker=dict(size=3, color=energies, colorscale='Viridis',
                       cmin=q1, cmax=q3, showscale=True),
            name=f'Iteration {itt}',
            visible=False
        ))

    fig.data[0].visible = True

    steps = []
    for i, itt in enumerate(iterations):
        step = dict(
            method='update',
            args=[{'visible': [j == i for j in range(len(iterations))]}],
            label=str(itt)
        )
        steps.append(step)

    fig.update_layout(
        sliders=[dict(active=0, steps=steps, currentvalue={"prefix": "Iteration: "})],
        scene=dict(
            xaxis_title="λ₁ - λ₂",
            yaxis_title="λ₁ - λ₃",
            zaxis_title="Energy (kcal/mol)"
        ),
        title="3D Energy Profile Evolution"
    )

    fig.write_html(output_dir / "energy_profile_3d.html")


def _extract_site_params(
    b_full: np.ndarray,
    c_full: np.ndarray,
    s_full: np.ndarray,
    x_full: np.ndarray,
    nsubs: list[int],
    site_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract bias parameters for a single site from the flat/full arrays.

    Args:
        b_full: Full b vector (nblocks,)
        c_full: Full c matrix (nblocks, nblocks)
        s_full: Full s matrix (nblocks, nblocks)
        x_full: Full x matrix (nblocks, nblocks)
        nsubs: Number of substates per site.
        site_idx: Which site to extract.

    Returns:
        (b, c, s, x) sliced to the site's substate block.
    """
    offset = sum(nsubs[:site_idx])
    ns = nsubs[site_idx]
    sl = slice(offset, offset + ns)
    return b_full[sl], c_full[sl, :][:, sl], s_full[sl, :][:, sl], x_full[sl, :][:, sl]


def _site_energy_1d(
    b: np.ndarray,
    c: np.ndarray,
    s: np.ndarray,
    x: np.ndarray,
    ns: int,
    sub_from: int,
    sub_to: int,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute total bias energy along a 1D transition coordinate.

    Scans from pure state `sub_from` (t=0) to pure state `sub_to` (t=1):
        lambda[sub_from] = 1 - t,  lambda[sub_to] = t,  others = 0.

    Returns:
        (t_values, energies)  both shape (n_points,)
    """
    t = np.linspace(0, 1, n_points)
    energies = np.empty(n_points)
    for k, tk in enumerate(t):
        lam = np.zeros(ns)
        lam[sub_from] = 1.0 - tk
        lam[sub_to] = tk
        energies[k] = total_energy(lam, b, c, s, x)
    return t, energies


def plot_1d_profiles(
    analysis_dir: Path,
    nsubs: list[int],
    output_dir: Path | None = None,
    main_plots_dir: Path | None = None,
    n_points: int = 200,
    fmt: str = "png",
) -> list[Path]:
    """Plot 1D bias energy profiles for each site in an analysis directory.

    For each site, generates an overlay of all pairwise transition profiles
    (Energy vs reaction coordinate).

    Args:
        analysis_dir: Path to a single analysisN/ directory.
        nsubs: Number of substates per site.
        output_dir: Where to save plots. Defaults to analysis_dir / "plots".
        main_plots_dir: When set, outputs to profiles/ subdir under this path.
        n_points: Lambda grid resolution for 1D scans.
        fmt: Image format (png, pdf, svg).

    Returns:
        List of output file paths.
    """
    if main_plots_dir is not None:
        output_dir = main_plots_dir / "profiles"
    elif output_dir is None:
        output_dir = analysis_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    b_full, c_full, s_full, x_full = _load_bias_parameters(analysis_dir)

    # Reshape c, s, x to square if they came in as 1D
    nblocks = sum(nsubs)
    if c_full.ndim == 1:
        c_full = c_full.reshape(nblocks, nblocks)
    if s_full.ndim == 1:
        s_full = s_full.reshape(nblocks, nblocks)
    if x_full.ndim == 1:
        x_full = x_full.reshape(nblocks, nblocks)

    apply_pub_style()

    itt = analysis_dir.name.replace("analysis", "")
    saved = []

    for site_idx, ns in enumerate(nsubs):
        b, c, s, x = _extract_site_params(b_full, c_full, s_full, x_full, nsubs, site_idx)

        # --- 1D profile overlay ---
        fig, ax = plt.subplots(figsize=(10, 6))
        n_pairs = ns * (ns - 1) // 2
        colors = get_state_colors(n_pairs)

        pair_idx = 0
        for i in range(ns):
            for j in range(i + 1, ns):
                t, E = _site_energy_1d(b, c, s, x, ns, i, j, n_points)
                # Shift so E(t=0) = 0 for readability
                E_shifted = E - E[0]

                ax.plot(
                    t, E_shifted,
                    color=colors[pair_idx % len(colors)],
                    linewidth=1.0 if n_pairs > 10 else 1.5,
                    alpha=0.6 if n_pairs > 10 else 0.8,
                    label=f"{i}\u2192{j}" if n_pairs <= 15 else None,
                )
                pair_idx += 1

        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Reaction coordinate (\u03bb)")
        ax.set_ylabel("\u0394E (kcal/mol)")
        ax.set_title(
            f"Site {site_idx + 1} \u2014 1D bias profiles (iter {itt})",
            fontweight="bold",
        )
        clean_axes(ax)
        if n_pairs <= 15:
            ax.legend(fontsize=8, ncol=max(1, n_pairs // 7), loc="best")

        out_1d = output_dir / f"site{site_idx + 1}_run{itt}.{fmt}"
        saved.append(savefig(fig, out_1d))

    return saved


# CLI entry point
def main():
    """Command-line interface for energy profile analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze ALF energy profiles")
    parser.add_argument("-i", "--input", required=True, help="Input folder with analysis directories")
    parser.add_argument("-o", "--output", default="plots", help="Output directory for plots")
    parser.add_argument("-n", "--num-points", type=int, default=100, help="Grid points per dimension")
    parser.add_argument("--animation", action="store_true", help="Create animation (3D only)")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png")

    args = parser.parse_args()

    config = EnergyProfileConfig(
        input_folder=args.input,
        num_points=args.num_points,
        output_dir=args.output,
        create_animation=args.animation,
        plot_format=args.format,
    )

    print(f"Analyzing energy profiles in {config.input_folder}")
    result = analyze_energy_profiles(config)

    print("\nResults:")
    print(f"  Iterations analyzed: {len(result.iterations)}")
    print(f"  Number of states: {result.n_states}")
    print(f"  Final RMSD: {result.rmsd_values[-1]:.4f} kcal/mol")
    print(f"  RMSD plot: {result.rmsd_plot}")
    if result.profile_plot:
        print(f"  Profile plot: {result.profile_plot}")


if __name__ == "__main__":
    main()
