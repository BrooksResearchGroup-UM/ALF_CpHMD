"""pKa plotting: static titration curves and time-resolved convergence.

Provides two main functions:

- :func:`plot_pka` -- scatter + fitted sigmoid with bootstrap CI bands.
- :func:`plot_pka_convergence` -- pKa vs time with error bands and optional
  experimental reference line.

Both functions use the shared ``plot_style`` helpers for publication-ready
output and operate headlessly via the ``Agg`` backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .ldin_parser import SiteInfo
    from .pka_fitting import FitResult, MultiStateFitResult

# ---------------------------------------------------------------------------
# Patch-name to human-readable label mapping
# ---------------------------------------------------------------------------

RESNAME_LABELS: dict[str, str] = {
    "ASPO": "Deprotonated",
    "ASH1": "O1 Protonation",
    "ASH2": "O2 Protonation",
    "GLUO": "Deprotonated",
    "GLH1": "O1 Protonation",
    "GLH2": "O2 Protonation",
    "HSPO": "Protonated",
    "HSPE": "\u03b5 Deprotonated",
    "HSPD": "\u03b4 Deprotonated",
    "LYSO": "Protonated",
    "LYSU": "Deprotonated",
    "TYRO": "Protonated",
    "TYRU": "Deprotonated",
    "ARGO": "Protonated",
    "ARU1": "N1 Deprotonated",
    "ARU2": "N2 Deprotonated",
    "SERO": "Protonated",
    "SERD": "Deprotonated",
    "CYSO": "Protonated",
    "CYSD": "Deprotonated",
}

# Markers cycled across replicate simulations
_MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "X", "h"]


# ---------------------------------------------------------------------------
# Static pKa plot
# ---------------------------------------------------------------------------


def plot_pka(
    site_label: str,
    pH_values: np.ndarray,
    populations: list[np.ndarray],
    errors: list[np.ndarray] | None,
    fit_result: FitResult | MultiStateFitResult,
    state_names: list[str],
    site_info: SiteInfo | None = None,
    exp_pka: float | None = None,
    output_path: str | Path = "pka_plot.png",
) -> Path:
    """Plot population vs pH with fitted sigmoid(s) and bootstrap CI bands.

    Parameters
    ----------
    site_label : str
        Human-readable site identifier (e.g. ``"GLU 45"``).
    pH_values : ndarray
        Sorted pH values used in the titration (length *N*).
    populations : list[ndarray]
        Per-state arrays of shape ``(N,)`` or ``(N, n_reps)`` giving the
        population at each pH.  For a 2-state system the list has one entry
        (the main state); for multi-state it has one entry per state.
    errors : list[ndarray] or None
        Same structure as *populations* but containing standard errors.  May
        be *None* to omit error bars.
    fit_result : FitResult or MultiStateFitResult
        Fit result from :mod:`cphmd.analysis.pka_fitting`.
    state_names : list[str]
        CHARMM patch names for each state (e.g. ``["GLUO", "GLH1", "GLH2"]``).
        Used to look up :data:`RESNAME_LABELS`.
    site_info : SiteInfo or None
        Parsed LDIN site information.  When provided, ``main_slope_sign`` is
        used for the experimental reference curve.
    exp_pka : float or None
        Experimental pKa to overlay as a dashed reference curve.
    output_path : str or Path
        Destination file path for the saved figure.

    Returns
    -------
    Path
        The *output_path* after writing.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .pka_fitting import MultiStateFitResult, sigmoid
    from .plot_style import apply_pub_style, clean_axes, get_state_colors
    from .plot_style import savefig as _savefig

    output_path = Path(output_path)
    apply_pub_style()

    fig, ax = plt.subplots(figsize=(10, 6))
    n_states = len(populations)
    colors = get_state_colors(max(n_states, 3))
    x_fit = np.linspace(float(pH_values.min()), float(pH_values.max()), 500)

    is_multi = isinstance(fit_result, MultiStateFitResult)
    legend_labels: set[str] = set()

    if is_multi:
        _plot_multistate(
            ax,
            pH_values,
            populations,
            errors,
            fit_result,
            state_names,
            site_info,
            x_fit,
            colors,
            legend_labels,
        )
    else:
        _plot_2state(
            ax,
            pH_values,
            populations,
            errors,
            fit_result,
            state_names,
            x_fit,
            colors,
            legend_labels,
        )

    # Experimental reference curve
    if exp_pka is not None:
        slope_sign = site_info.main_slope_sign if site_info is not None else -1
        ref_curve = sigmoid(x_fit, 1.0, exp_pka, float(slope_sign))
        ax.plot(
            x_fit,
            ref_curve,
            color="black",
            linestyle="--",
            linewidth=2.5,
            label=f"Experimental: pKa={exp_pka:.2f}",
        )

    ax.set_title(f"pKa plot for {site_label}", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("pH")
    ax.set_ylabel("Population")
    clean_axes(ax)

    # Legend placement: left for HSP-like multi-state, right otherwise
    restype = site_label.split()[0] if " " in site_label else site_label
    loc = "center left" if restype in ("HSP", "HIS") else "center right"
    ax.legend(title="Residue Type", loc=loc)

    fig.tight_layout()
    return _savefig(fig, output_path)


# ---------------------------------------------------------------------------
# Internal helpers for plot_pka
# ---------------------------------------------------------------------------


def _plot_2state(
    ax,
    pH_values: np.ndarray,
    populations: list[np.ndarray],
    errors: list[np.ndarray] | None,
    fit: FitResult,
    state_names: list[str],
    x_fit: np.ndarray,
    colors: list,
    legend_labels: set[str],
) -> None:
    """Plot a single 2-state sigmoid with bootstrap CI band."""
    from .pka_fitting import sigmoid

    color = colors[0]
    pop = populations[0]
    err = errors[0] if errors is not None else None
    resname_display = RESNAME_LABELS.get(state_names[0], "State 0")

    # Scatter data points (one marker per replicate column)
    _scatter_state(ax, pH_values, pop, err, color, resname_display, fit, legend_labels)

    # Fitted curve
    if fit.bootstrap_params is not None and len(fit.bootstrap_params) > 0:
        curves = np.array([sigmoid(x_fit, *p) for p in fit.bootstrap_params])
        mean_curve = np.mean(curves, axis=0)
        ax.plot(x_fit, mean_curve, color=color, linestyle="-", linewidth=2, zorder=1)
        if len(fit.bootstrap_params) > 1:
            lb = np.percentile(curves, 2.5, axis=0)
            ub = np.percentile(curves, 97.5, axis=0)
            ax.fill_between(x_fit, lb, ub, color=color, alpha=0.3, edgecolor=color, linewidth=0.5)
    else:
        # Single deterministic curve
        curve = sigmoid(x_fit, fit.amplitude, fit.pka, fit.slope)
        ax.plot(x_fit, curve, color=color, linestyle="-", linewidth=2, zorder=1)


def _plot_multistate(
    ax,
    pH_values: np.ndarray,
    populations: list[np.ndarray],
    errors: list[np.ndarray] | None,
    fit: MultiStateFitResult,
    state_names: list[str],
    site_info: SiteInfo | None,
    x_fit: np.ndarray,
    colors: list,
    legend_labels: set[str],
) -> None:
    """Plot all states of a multi-state sigmoid with CI bands."""
    from .pka_fitting import make_multi_sigmoid

    n_states = len(populations)
    slope_sign = site_info.main_slope_sign if site_info is not None else -1
    multi_func = make_multi_sigmoid(slope_sign)

    for state_idx in range(n_states):
        color = colors[state_idx]
        pop = populations[state_idx]
        err = errors[state_idx] if errors is not None else None
        resname_display = RESNAME_LABELS.get(
            state_names[state_idx] if state_idx < len(state_names) else "",
            f"State {state_idx}",
        )

        # Build per-state label
        if state_idx == 0:
            label = (
                f"{resname_display}: macro pKa={fit.pka_macro:.2f}"
                f"\u00b1{fit.pka_macro_err:.2f}, h={fit.hill:.2f}\u00b1{fit.hill_err:.2f}"
            )
        elif state_idx == 1:
            micro = fit.pka_micro[0] if fit.pka_micro else float("nan")
            label = (
                f"{resname_display}: micro pKa={micro:.2f},"
                f" f={fit.f_taut:.3f}\u00b1{fit.f_taut_err:.3f}"
            )
        else:
            micro = fit.pka_micro[1] if len(fit.pka_micro) > 1 else float("nan")
            label = f"{resname_display}: micro pKa={micro:.2f}," f" f={1.0 - fit.f_taut:.3f}"

        # Scatter data
        _scatter_state_raw(ax, pH_values, pop, err, color, label, legend_labels)

        # Fitted curve with CI band
        if fit.bootstrap_params is not None and len(fit.bootstrap_params) > 0:

            def _state_curve(pH: np.ndarray, params: np.ndarray, si: int) -> np.ndarray:
                full = multi_func(np.tile(pH, n_states), *params)
                n = len(pH)
                return full[si * n : (si + 1) * n]

            curves = np.array([_state_curve(x_fit, p, state_idx) for p in fit.bootstrap_params])
            mean_curve = np.mean(curves, axis=0)
            ax.plot(x_fit, mean_curve, color=color, linestyle="-", linewidth=2, zorder=1)
            if len(fit.bootstrap_params) > 1:
                lb = np.percentile(curves, 2.5, axis=0)
                ub = np.percentile(curves, 97.5, axis=0)
                ax.fill_between(
                    x_fit, lb, ub, color=color, alpha=0.3, edgecolor=color, linewidth=0.5
                )
        else:
            # Deterministic curve from fitted parameters
            full = multi_func(np.tile(x_fit, n_states), fit.pka_macro, fit.hill, fit.f_taut)
            n = len(x_fit)
            curve = full[state_idx * n : (state_idx + 1) * n]
            ax.plot(x_fit, curve, color=color, linestyle="-", linewidth=2, zorder=1)


def _scatter_state(
    ax,
    pH_values: np.ndarray,
    pop: np.ndarray,
    err: np.ndarray | None,
    color,
    resname_display: str,
    fit: FitResult,
    legend_labels: set[str],
) -> None:
    """Scatter data points for a 2-state fit, building a combined label."""
    label = (
        f"{resname_display}: pKa={fit.pka_corrected:.2f}\u00b1{fit.pka_err:.2f},"
        f" h={fit.slope:.2f}\u00b1{fit.slope_err:.2f}"
    )
    _scatter_state_raw(ax, pH_values, pop, err, color, label, legend_labels)


def _scatter_state_raw(
    ax,
    pH_values: np.ndarray,
    pop: np.ndarray,
    err: np.ndarray | None,
    color,
    label: str,
    legend_labels: set[str],
) -> None:
    """Scatter population data with one marker per replicate."""
    pop = np.asarray(pop)
    if pop.ndim == 1:
        # Single replicate per pH
        e = np.asarray(err) if err is not None else None
        use_label = label if label not in legend_labels else None
        ax.errorbar(
            pH_values,
            pop,
            yerr=e,
            fmt=_MARKERS[0],
            color=color,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.5,
            capsize=3,
            alpha=0.8,
            label=use_label,
        )
        legend_labels.add(label)
    else:
        # Multiple replicates: shape (N_pH, n_reps)
        n_reps = pop.shape[1]
        err_arr = np.asarray(err) if err is not None else None
        for j in range(n_reps):
            ms = _MARKERS[j % len(_MARKERS)]
            e_col = err_arr[:, j] if err_arr is not None and err_arr.ndim == 2 else None
            use_label = label if label not in legend_labels else None
            ax.errorbar(
                pH_values,
                pop[:, j],
                yerr=e_col,
                fmt=ms,
                color=color,
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=0.5,
                capsize=3,
                alpha=0.8,
                label=use_label,
            )
            legend_labels.add(label)


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------


def plot_pka_convergence(
    site_label: str,
    time_bins_ns: np.ndarray,
    pka_per_bin: np.ndarray,
    pka_err_per_bin: np.ndarray,
    exp_pka: float | None = None,
    accumulate: bool = False,
    output_path: str | Path = "pka_convergence.png",
    data_path: str | Path | None = None,
) -> Path:
    """Plot pKa vs simulation time with error bands.

    Parameters
    ----------
    site_label : str
        Human-readable site identifier (e.g. ``"GLU 45"``).
    time_bins_ns : ndarray
        Time bin centres in nanoseconds (length *M*).
    pka_per_bin : ndarray
        pKa value at each time bin (length *M*).  May contain NaN for
        bins where fitting failed.
    pka_err_per_bin : ndarray
        Error (95 % CI half-width) at each time bin (length *M*).
    exp_pka : float or None
        Experimental pKa to overlay as a horizontal dashed line.
    accumulate : bool
        If *True*, the title reads "Accumulated"; otherwise "Instantaneous".
    output_path : str or Path
        Destination file path for the saved figure.
    data_path : str or Path or None
        If provided, write a tab-separated ``.dat`` file with columns
        ``Time(ns)  pKa  pKa_err``.

    Returns
    -------
    Path
        The *output_path* after writing.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plot_style import apply_pub_style, clean_axes
    from .plot_style import savefig as _savefig

    output_path = Path(output_path)
    apply_pub_style()

    time_arr = np.asarray(time_bins_ns, dtype=float)
    pka_arr = np.asarray(pka_per_bin, dtype=float)
    err_arr = np.asarray(pka_err_per_bin, dtype=float)

    # Filter NaN entries for plotting
    valid = np.isfinite(pka_arr)
    t_valid = time_arr[valid]
    pka_valid = pka_arr[valid]
    err_valid = err_arr[valid]

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(t_valid) > 0:
        color = plt.cm.Set1.colors[0]

        ax.plot(t_valid, pka_valid, linestyle="-", color=color, linewidth=2)
        ax.fill_between(
            t_valid,
            pka_valid - err_valid,
            pka_valid + err_valid,
            color=color,
            alpha=0.25,
        )
        ax.errorbar(
            t_valid,
            pka_valid,
            yerr=err_valid,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
            alpha=0.8,
            label=site_label,
        )

        # Average pKa reference
        avg_pka = float(np.nanmean(pka_valid))
        ax.axhline(
            avg_pka,
            color=color,
            linestyle=":",
            linewidth=1.5,
            label=f"Average pKa={avg_pka:.2f}",
        )

    # Experimental reference
    if exp_pka is not None:
        ax.axhline(
            exp_pka,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Experimental: pKa={exp_pka:.2f}",
        )

    mode = "Accumulated" if accumulate else "Instantaneous"
    ax.set_title(f"{mode} pKa: {site_label}", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("pKa")
    ax.set_xlim(left=0)
    clean_axes(ax)
    ax.legend(title="Residue Type", loc="upper right")

    fig.tight_layout()

    # Optional data export
    if data_path is not None:
        data_path = Path(data_path)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, "w") as f:
            f.write("Time(ns)\tpKa\tpKa_err\n")
            for t, pk, pe in zip(time_arr, pka_arr, err_arr):
                pk_s = "NaN" if np.isnan(pk) else f"{pk:.2f}"
                pe_s = "NaN" if np.isnan(pe) else f"{pe:.2f}"
                f.write(f"{t:.2f}\t{pk_s}\t{pe_s}\n")

    return _savefig(fig, output_path)
