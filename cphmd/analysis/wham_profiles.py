"""WHAM free energy profile visualization.

Plots the free energy profiles computed by CUDA WHAM from the
multisite/G{n}.dat files. These represent the actual thermodynamic
landscape estimated from simulation data, as opposed to the applied
bias potential (plotted by plot_1d_profiles in energy_profiles.py).

Profile types (matching CUDA ptype):
  0 — 1D substituent profiles (G1): free energy vs lambda for each substate
  1 — 1D transition profiles (G12): pairwise transition free energies
  2 — 2D joint profiles (G2): surface plots for >2-state systems
  3 — Cross-site profiles (G11): 2D surfaces for inter-site coupling

Originally implemented as PlotFreeEnergy5 in the ALF utility library.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cphmd.analysis.plot_style import apply_pub_style, clean_axes, get_state_colors, savefig


def plot_wham_profiles(
    analysis_dir: Path,
    nsubs: list[int],
    msprof: int = 0,
    output_dir: Path | None = None,
    main_plots_dir: Path | None = None,
    fmt: str = "png",
    include_1d: bool = True,
    include_2d: bool = True,
    include_cross: bool = True,
) -> list[Path]:
    """Plot WHAM free energy profiles from multisite/G{n}.dat files.

    Generates per-site visualizations of the WHAM-estimated free energy
    surface. For each site:
      1. 1D overlay of per-substituent profiles (G1)
      2. 1D overlay of pairwise transition profiles (G12)
      3. 2D surface plots for >2-state systems (G2)
    If msprof is enabled, also plots cross-site coupling surfaces (G11).

    Args:
        analysis_dir: Path to a single analysisN/ directory.
        nsubs: Number of substates per site.
        msprof: Multisite profile flag (0=no cross-site, 1=yes).
        output_dir: Where to save plots. Defaults to analysis_dir / "plots".
        main_plots_dir: When set, outputs to organized subdirs under this path
            (wham_1d/, wham_transitions/, wham_2d/, wham_cross/).
        fmt: Image format (png, pdf, svg).
        include_1d: Plot ptype 0/1 1D WHAM profiles.
        include_2d: Plot ptype 2 joint WHAM profiles.
        include_cross: Plot ptype 3 cross-site WHAM profiles when msprof is enabled.

    Returns:
        List of saved plot file paths.
    """
    ms_dir = analysis_dir / "multisite"
    if not ms_dir.is_dir():
        return []

    # Resolve output directories
    if main_plots_dir is not None:
        wham_1d_dir = main_plots_dir / "wham_1d"
        wham_trans_dir = main_plots_dir / "wham_transitions"
        wham_2d_dir = main_plots_dir / "wham_2d"
        wham_cross_dir = main_plots_dir / "wham_cross"
    else:
        if output_dir is None:
            output_dir = analysis_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        wham_1d_dir = wham_trans_dir = wham_2d_dir = wham_cross_dir = output_dir

    itt = analysis_dir.name.replace("analysis", "")
    saved: list[Path] = []

    # Walk the G file index exactly as CUDA enumerates profiles
    i_g = 1
    iblock = 0

    for isite in range(len(nsubs)):
        ns = nsubs[isite]
        jblock = iblock

        for jsite in range(isite, len(nsubs)):
            if isite == jsite:
                # --- ptype 0: 1D substituent profiles ---
                g1_data: dict[int, np.ndarray] = {}
                for i in range(ns):
                    g_path = ms_dir / f"G{i_g}.dat"
                    if include_1d and g_path.exists():
                        g1_data[i] = np.loadtxt(g_path)
                    i_g += 1

                if include_1d and g1_data:
                    p = _plot_1d_substituent(
                        g1_data, ns, isite, itt, wham_1d_dir, fmt,
                    )
                    if p:
                        saved.append(p)

                # --- ptype 1: transition profiles (G12) ---
                g12_data: dict[tuple[int, int], np.ndarray] = {}
                for i in range(ns):
                    for j in range(i + 1, ns):
                        g_path = ms_dir / f"G{i_g}.dat"
                        if include_1d and g_path.exists():
                            g12_data[(i, j)] = np.loadtxt(g_path)
                        i_g += 1

                if include_1d and g12_data:
                    p = _plot_1d_transition(
                        g12_data, ns, isite, itt, wham_trans_dir, fmt,
                    )
                    if p:
                        saved.append(p)

                # --- ptype 2: 2D joint profiles (G2, only nsubs > 2) ---
                if ns > 2:
                    g2_data: dict[tuple[int, int], np.ndarray] = {}
                    for i in range(ns):
                        for j in range(i + 1, ns):
                            g_path = ms_dir / f"G{i_g}.dat"
                            if include_2d and g_path.exists():
                                g2_data[(i, j)] = np.loadtxt(g_path)
                            i_g += 1

                    if include_2d and g2_data:
                        p = _plot_2d_joint(
                            g2_data, ns, isite, itt, wham_2d_dir, fmt,
                        )
                        if p:
                            saved.append(p)

            elif msprof:
                # --- ptype 3: cross-site coupling profiles (G11) ---
                g11_data: dict[tuple[int, int], np.ndarray] = {}
                for i in range(nsubs[isite]):
                    for j in range(nsubs[jsite]):
                        g_path = ms_dir / f"G{i_g}.dat"
                        if include_cross and g_path.exists():
                            g11_data[(i, j)] = np.loadtxt(g_path)
                        i_g += 1

                if include_cross and g11_data:
                    p = _plot_cross_site(
                        g11_data, nsubs[isite], nsubs[jsite],
                        isite, jsite, itt, wham_cross_dir, fmt,
                    )
                    if p:
                        saved.append(p)

            jblock += nsubs[jsite]
        iblock += nsubs[isite]

    return saved


# ---------------------------------------------------------------------------
# Internal plotting helpers
# ---------------------------------------------------------------------------

def _make_1d_grid(n_values: int) -> np.ndarray:
    """Build a 1D lambda grid matching WHAM bin centers.

    WHAM uses bins^2 values for 1D profiles, with bin centers at
    (0.5/bins^2, 1.5/bins^2, ..., (bins^2-0.5)/bins^2).
    """
    return np.arange(0.5 / n_values, 1.0, 1.0 / n_values)


def _make_2d_grid(n_values: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a 2D lambda grid matching WHAM bin centers.

    For 2D profiles, n_values = bins * bins. Each axis has `bins` points.
    """
    bins = int(math.isqrt(n_values))
    ax = np.arange(0.5 / bins, 1.0, 1.0 / bins)
    return np.meshgrid(ax, ax)


def _plot_1d_substituent(
    g1_data: dict[int, np.ndarray],
    ns: int,
    site_idx: int,
    itt: str,
    output_dir: Path,
    fmt: str,
) -> Path | None:
    """Plot 1D per-substituent free energy profiles."""
    if not g1_data:
        return None

    apply_pub_style()

    # Infer grid from first available profile
    first = next(iter(g1_data.values()))
    emid = _make_1d_grid(len(first))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = get_state_colors(ns)

    for i in sorted(g1_data):
        g = g1_data[i]
        # Shift so minimum = 0 for readability
        g_shifted = g - np.nanmin(g)
        ax.plot(
            emid, g_shifted,
            color=colors[i % len(colors)],
            linewidth=1.5,
            label=f"State {i + 1}",
        )

    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("Free energy (kcal/mol)")
    ax.set_title(
        f"Site {site_idx + 1} \u2014 WHAM 1D profiles (iter {itt})",
        fontweight="bold",
    )
    clean_axes(ax)
    ax.legend(loc="best")

    out = output_dir / f"site{site_idx + 1}_run{itt}.{fmt}"
    return savefig(fig, out)


def _plot_1d_transition(
    g12_data: dict[tuple[int, int], np.ndarray],
    ns: int,
    site_idx: int,
    itt: str,
    output_dir: Path,
    fmt: str,
) -> Path | None:
    """Plot 1D pairwise transition free energy profiles."""
    if not g12_data:
        return None

    apply_pub_style()

    first = next(iter(g12_data.values()))
    emid = _make_1d_grid(len(first))

    n_pairs = len(g12_data)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = get_state_colors(n_pairs)

    for idx, ((i, j), g) in enumerate(sorted(g12_data.items())):
        ax.plot(
            emid, g,
            color=colors[idx],
            linewidth=1.5 if n_pairs <= 10 else 1.0,
            alpha=0.8 if n_pairs <= 10 else 0.6,
            label=f"{i}\u2194{j}" if n_pairs <= 15 else None,
        )

    ax.set_xlabel("Transition coordinate $\\lambda_i / (\\lambda_i + \\lambda_j)$")
    ax.set_ylabel("Free energy (kcal/mol)")
    ax.set_title(
        f"Site {site_idx + 1} \u2014 WHAM transition profiles (iter {itt})",
        fontweight="bold",
    )
    clean_axes(ax)
    if n_pairs <= 15:
        ax.legend(loc="best")

    out = output_dir / f"site{site_idx + 1}_run{itt}.{fmt}"
    return savefig(fig, out)


def _plot_2d_joint(
    g2_data: dict[tuple[int, int], np.ndarray],
    ns: int,
    site_idx: int,
    itt: str,
    output_dir: Path,
    fmt: str,
) -> Path | None:
    """Plot 2D joint free energy surfaces for >2-state systems."""
    if not g2_data:
        return None

    apply_pub_style()

    # Grid layout: (ns-1) x (ns-1) subplots
    ncols = ns - 1
    nrows = ns - 1
    fig = plt.figure(figsize=(5 * ncols, 4 * nrows))

    first = next(iter(g2_data.values()))
    grid_x, grid_y = _make_2d_grid(len(first))
    bins = grid_x.shape[0]

    for (i, j), g in sorted(g2_data.items()):
        g2d = g.reshape(bins, bins)
        subplot_idx = i * ncols + j
        ax = fig.add_subplot(nrows, ncols, subplot_idx, projection="3d")
        ax.plot_surface(grid_x, grid_y, g2d, cmap="viridis", alpha=0.8)
        ax.set_xlabel(f"$\\lambda_{{{i}}}$", fontsize=9)
        ax.set_ylabel(f"$\\lambda_{{{j}}}$", fontsize=9)
        ax.set_zlabel("G", fontsize=9)
        ax.set_title(f"{i}\u2194{j}", fontsize=10)

    fig.suptitle(
        f"Site {site_idx + 1} \u2014 WHAM 2D profiles (iter {itt})",
        fontweight="bold",
    )
    plt.tight_layout()

    out = output_dir / f"site{site_idx + 1}_run{itt}.{fmt}"
    return savefig(fig, out, dpi=200)


def _plot_cross_site(
    g11_data: dict[tuple[int, int], np.ndarray],
    ns_i: int,
    ns_j: int,
    site_i: int,
    site_j: int,
    itt: str,
    output_dir: Path,
    fmt: str,
) -> Path | None:
    """Plot cross-site coupling 2D surfaces."""
    if not g11_data:
        return None

    apply_pub_style()

    fig = plt.figure(figsize=(5 * ns_j, 4 * ns_i))

    first = next(iter(g11_data.values()))
    grid_x, grid_y = _make_2d_grid(len(first))
    bins = grid_x.shape[0]

    for (i, j), g in sorted(g11_data.items()):
        g2d = g.reshape(bins, bins)
        subplot_idx = i * ns_j + j + 1
        ax = fig.add_subplot(ns_i, ns_j, subplot_idx, projection="3d")
        ax.plot_surface(grid_x, grid_y, g2d, cmap="viridis", alpha=0.8)
        ax.set_xlabel(f"$\\lambda_{{{i}}}^{{s{site_i + 1}}}$", fontsize=9)
        ax.set_ylabel(f"$\\lambda_{{{j}}}^{{s{site_j + 1}}}$", fontsize=9)
        ax.set_zlabel("G", fontsize=9)
        ax.set_title(f"({i},{j})", fontsize=10)

    fig.suptitle(
        f"Sites {site_i + 1}\u2194{site_j + 1} \u2014 WHAM coupling profiles (iter {itt})",
        fontweight="bold",
    )
    plt.tight_layout()

    out = output_dir / f"sites{site_i + 1}_{site_j + 1}_run{itt}.{fmt}"
    return savefig(fig, out, dpi=200)
