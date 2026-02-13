"""Shared plot styling for CpHMD analysis plots.

Provides consistent rcParams, color palettes, and save helpers
so all diagnostic plots (WHAM profiles, energy profiles, HH curves,
convergence) share a unified publication-ready appearance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def apply_pub_style() -> None:
    """Apply publication-ready matplotlib rcParams.

    Call once at the start of any plot function to ensure consistent
    axes linewidth, tick direction, spine style, and font sizing.
    """
    plt.rcParams.update({
        "axes.linewidth": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
    })


def get_state_colors(n: int) -> list:
    """Return a deterministic color list for *n* items.

    Uses ``tab10`` for <= 10 items, ``tab20`` for <= 20, and ``turbo``
    beyond that.  The same *n* always produces the same palette, so a
    given state/pair index gets the same color across plot types.
    """
    if n <= 10:
        cmap = plt.get_cmap("tab10")
        return [cmap(i) for i in range(n)]
    elif n <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(i) for i in range(n)]
    else:
        cmap = plt.get_cmap("turbo")
        return [cmap(i / max(n - 1, 1)) for i in range(n)]


def clean_axes(ax) -> None:
    """Remove top/right spines and add light grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3)


def savefig(fig, path: Path, dpi: int = 300) -> Path:
    """Save *fig* to *path* with consistent settings, then close.

    Creates parent directories as needed.  Returns *path* for chaining.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
