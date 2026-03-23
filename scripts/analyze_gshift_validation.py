#!/usr/bin/env python3
"""Analyze gshift validation runs — ensemble version.

Walks analysis directories for each run, extracts populations and pKa estimates
at each iteration, and produces publication-quality convergence plots comparing
gshift on vs off.  Supports ensemble aggregation: if seed directories like
``lys_gshift_on_s{1..5}/`` exist, they are grouped and plotted as mean ± SEM
shaded bands.

Output figures:
  - fig_md_convergence.png    — ΔpKa(t) for LYS/GLU/HSP, on vs off
  - fig_md_K_independence.png — final pKa error vs K for GLU
  - fig_md_W_dependence.png   — convergence traces vs W for LYS

Usage:
    python scripts/analyze_gshift_validation.py [--runs-dir runs/]
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Publication style (matching gshift paper exploration figures)
# ---------------------------------------------------------------------------

def apply_style():
    mpl.rcParams.update({
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "legend.handlelength": 1.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.5)


# Algorithm colors (consistent with paper exploration figures)
C_ON  = (0.12, 0.47, 0.71, 0.85)   # dodgerblue
C_OFF = (1.00, 0.27, 0.00, 0.85)   # orangered

# ---------------------------------------------------------------------------
# Reference pKa values
# ---------------------------------------------------------------------------
PKA_REF = {
    "lys": 10.4,
    "glu": 4.25,   # macroscopic pKa (micro=3.95, N_D=2, macro=micro+log10(2))
    "hsp": None,    # computed from patches.dat (~6.45)
}

KT = 0.001987204 * 298.15


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def read_populations(analysis_dir: Path) -> dict | None:
    """Read populations.dat and return normalized populations per site."""
    pop_file = analysis_dir / "populations.dat"
    if not pop_file.exists():
        return None

    nsubs = []
    norm_vals = []

    text = pop_file.read_text()
    m = re.search(r"nsubs=\[([^\]]+)\]", text)
    if m:
        nsubs = [int(x) for x in m.group(1).split(",")]
    else:
        return None

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        parts = line.split()
        if len(parts) >= 3:
            norm_vals.append(float(parts[2]))

    if not norm_vals:
        return None

    site_pops = []
    idx = 0
    for n in nsubs:
        site_pops.append(np.array(norm_vals[idx : idx + n]))
        idx += n

    return {"nsubs": nsubs, "norm": site_pops}


def read_phase(analysis_dir: Path) -> int | None:
    f = analysis_dir / "phase.dat"
    if f.exists():
        return int(f.read_text().strip())
    return None


def pka_from_populations_2state(pops: np.ndarray, pka_ref: float, tag: str) -> float:
    if len(pops) != 2 or np.sum(pops) < 0.01:
        return np.nan
    p0, p1 = np.clip(pops[0], 1e-6, 1 - 1e-6), np.clip(pops[1], 1e-6, 1 - 1e-6)
    if tag == "UPOS":
        return pka_ref - np.log10(p1 / p0)
    else:
        return pka_ref + np.log10(p1 / p0)


def pka_from_populations_3state(pops: np.ndarray, pka_ref: float, tag: str) -> float:
    if len(pops) != 3 or np.sum(pops) < 0.01:
        return np.nan
    pops = np.clip(pops, 1e-6, 1.0)
    if tag == "UNEG":
        p_deprot = pops[0]
        p_prot = pops[1] + pops[2]
        return pka_ref + np.log10(p_prot / p_deprot)
    elif tag == "UPOS":
        p_prot = pops[0]
        p_deprot = pops[1] + pops[2]
        return pka_ref - np.log10(p_deprot / p_prot)
    return np.nan


def detect_system(run_dir: Path) -> tuple[str, float, str]:
    patches = run_dir / "solvated" / "prep" / "patches.dat"
    if not patches.exists():
        return "unknown", 7.0, "NONE"

    text = patches.read_text()
    lines = [l for l in text.splitlines() if l and not l.startswith("SEGID")]

    if "LYSO" in text or "LYSU" in text:
        for line in lines:
            if "UPOS" in line:
                pka = float(line.split("UPOS")[1].strip())
                return "lys", pka, "UPOS"
        return "lys", 10.4, "UPOS"

    elif "GLUO" in text or "GLH1" in text:
        micro_pkas = []
        for line in lines:
            if "UNEG" in line:
                micro_pkas.append(float(line.split("UNEG")[1].strip()))
        if micro_pkas:
            # macro-pKa = micro + log10(N_D) for equal tautomers
            N_D = len(micro_pkas)
            macro = micro_pkas[0] + np.log10(N_D)
            return "glu", macro, "UNEG"
        return "glu", 4.25, "UNEG"

    elif "HSPO" in text or "HSPD" in text:
        pkas = []
        for line in lines:
            if "UPOS" in line:
                pkas.append(float(line.split("UPOS")[1].strip()))
        if pkas:
            macro = -np.log10(sum(10 ** (-p) for p in pkas))
            return "hsp", macro, "UPOS"
        return "hsp", 6.45, "UPOS"

    return "unknown", 7.0, "NONE"


def extract_convergence(run_dir: Path) -> dict:
    system, pka_ref, tag = detect_system(run_dir)

    iterations, pka_values, phases = [], [], []

    sol_dir = run_dir / "solvated"
    analysis_dirs = sorted(
        sol_dir.glob("analysis*"),
        key=lambda d: int(re.search(r"(\d+)$", d.name).group())
        if re.search(r"(\d+)$", d.name) else -1,
    )

    for adir in analysis_dirs:
        m = re.search(r"(\d+)$", adir.name)
        if not m:
            continue
        idx = int(m.group())
        if idx == 0:
            continue

        pop_data = read_populations(adir)
        if pop_data is None:
            continue

        nsubs = pop_data["nsubs"]
        pops = pop_data["norm"][0]

        if nsubs[0] == 2:
            pka = pka_from_populations_2state(pops, pka_ref, tag)
        elif nsubs[0] == 3:
            pka = pka_from_populations_3state(pops, pka_ref, tag)
        else:
            pka = np.nan

        phase = read_phase(adir) or 0

        iterations.append(idx)
        pka_values.append(pka)
        phases.append(phase)

    return {
        "iterations": np.array(iterations),
        "pka": np.array(pka_values),
        "phase": np.array(phases),
        "system": system,
        "pka_ref": pka_ref,
        "tag": tag,
    }


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def ewma(x, span=10):
    """Exponentially weighted moving average."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def ragged_mean_sem(traces: list[np.ndarray],
                    iters_list: list[np.ndarray] | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and SEM from ragged arrays (unequal lengths).

    Returns (iters, mean, sem) using all available data at each position.
    The iters array spans the longest trace.
    """
    max_len = max(len(t) for t in traces)
    longest_idx = max(range(len(traces)), key=lambda i: len(traces[i]))
    if iters_list is not None:
        iters = iters_list[longest_idx]
    else:
        iters = np.arange(max_len)

    mean = np.full(max_len, np.nan)
    sem = np.full(max_len, np.nan)
    for j in range(max_len):
        vals = [t[j] for t in traces if j < len(t)]
        if vals:
            mean[j] = np.mean(vals)
            if len(vals) > 1:
                sem[j] = np.std(vals, ddof=1) / np.sqrt(len(vals))
    return iters, mean, sem


# ---------------------------------------------------------------------------
# Ensemble helpers
# ---------------------------------------------------------------------------

def find_seed_dirs(runs_dir: Path, base_name: str) -> list[Path]:
    """Find seed directories matching ``{base_name}_s{1..N}/`` (case-insensitive)."""
    dirs = []
    for s in range(1, 20):
        # Try exact case first, then lowercase variant
        for variant in [base_name, base_name.lower()]:
            d = runs_dir / f"{variant}_s{s}"
            if d.is_dir() and (d / "solvated").exists() and d not in dirs:
                dirs.append(d)
                break
    return dirs


def find_all_runs(runs_dir: Path, base_name: str) -> list[Path]:
    """Find ALL runs for a condition: seeds + original (case-insensitive).

    Includes ``{base_name}_s{1..N}`` seed dirs AND the original run dir
    (which may be lowercase, e.g. ``glu_k3_off`` vs ``glu_K3_off``).
    The original run is the same design as the seeds, just not numbered.
    """
    dirs = find_seed_dirs(runs_dir, base_name)

    # Also try original run dir (exact match and lowercase variant)
    for candidate in [base_name, base_name.lower()]:
        orig = runs_dir / candidate
        if orig not in dirs and orig.is_dir() and (orig / "solvated").exists():
            dirs.insert(0, orig)

    return dirs


def extract_ensemble(runs_dir: Path, base_name: str, smooth_span: int = 12
                     ) -> dict | None:
    """Extract convergence from all seeds and compute mean ± SEM.

    Returns dict with keys: iters, mean, sem, individual, pka_ref, system,
    n_seeds, or None if no seed data found.
    """
    seed_dirs = find_seed_dirs(runs_dir, base_name)
    if not seed_dirs:
        return None

    all_data = []
    pka_ref = None
    system = None

    for sd in seed_dirs:
        data = extract_convergence(sd)
        if len(data["pka"]) < 3:
            continue
        delta = data["pka"] - data["pka_ref"]
        valid = ~np.isnan(delta)
        iters = data["iterations"][valid]
        dv = delta[valid]
        if len(dv) < 3:
            continue
        sm = ewma(dv, span=smooth_span)
        all_data.append({"iters": iters, "delta_raw": dv, "delta_smooth": sm})
        if pka_ref is None:
            pka_ref = data["pka_ref"]
            system = data["system"]

    if not all_data:
        return None

    # Align seeds onto common iteration grid (use shortest common range)
    min_len = min(len(d["iters"]) for d in all_data)
    # Use iteration indices from first seed, truncated to min_len
    common_iters = all_data[0]["iters"][:min_len]

    raw_matrix = np.array([d["delta_raw"][:min_len] for d in all_data])
    smooth_matrix = np.array([d["delta_smooth"][:min_len] for d in all_data])

    return {
        "iters": common_iters,
        "raw_mean": np.mean(raw_matrix, axis=0),
        "raw_sem": np.std(raw_matrix, axis=0, ddof=1) / np.sqrt(len(all_data))
                   if len(all_data) > 1 else np.zeros(min_len),
        "smooth_mean": np.mean(smooth_matrix, axis=0),
        "smooth_sem": np.std(smooth_matrix, axis=0, ddof=1) / np.sqrt(len(all_data))
                      if len(all_data) > 1 else np.zeros(min_len),
        "individual": all_data,
        "pka_ref": pka_ref,
        "system": system,
        "n_seeds": len(all_data),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_trace(ax, iters, delta, color, label, smooth_span=12):
    """Plot raw trace (faint) + EWMA smooth (solid)."""
    valid = ~np.isnan(delta)
    it, dv = iters[valid], delta[valid]
    if len(dv) < 3:
        return
    ax.plot(it, dv, color=(*color[:3], 0.15), lw=0.5, zorder=1)
    sm = ewma(dv, span=smooth_span)
    ax.plot(it, sm, color=color, lw=1.8, label=label, zorder=5)


def _plot_ensemble(ax, ens, color, label):
    """Plot ensemble mean ± SEM as solid line + shaded band."""
    it = ens["iters"]
    mean = ens["smooth_mean"]
    sem = ens["smooth_sem"]

    # Individual seeds (very faint)
    for seed_data in ens["individual"]:
        n = min(len(it), len(seed_data["delta_raw"]))
        ax.plot(seed_data["iters"][:n], seed_data["delta_raw"][:n],
                color=(*color[:3], 0.08), lw=0.4, zorder=1)

    # Mean line
    ax.plot(it, mean, color=color, lw=2.0,
            label=f"{label} ($n$={ens['n_seeds']})", zorder=5)

    # SEM band
    if ens["n_seeds"] > 1:
        ax.fill_between(it, mean - sem, mean + sem,
                        color=(*color[:3], 0.2), lw=0, zorder=3)


def plot_convergence(runs_data: dict, runs_dir: Path, output_dir: Path):
    """Fig 1: ΔpKa convergence for LYS/GLU/HSP, gshift on vs off.

    Uses ensemble data if available, falls back to single-run data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), constrained_layout=True)

    systems = ["lys", "glu", "hsp"]
    labels = [
        r"(a) LYS — 2-state",
        r"(b) GLU — 3-state ($N_D\!=\!2$)",
        r"(c) HSP — 3-state (unequal)",
    ]

    for ax, sys_name, panel_label in zip(axes, systems, labels):
        on_key = f"{sys_name}_gshift_on"
        off_key = f"{sys_name}_gshift_off"

        # Try ensemble first
        ens_on = extract_ensemble(runs_dir, on_key)
        ens_off = extract_ensemble(runs_dir, off_key)

        if ens_on is not None:
            _plot_ensemble(ax, ens_on, C_ON, "per-frame correction")
        elif on_key in runs_data and len(runs_data[on_key]["pka"]) > 0:
            d = runs_data[on_key]
            _plot_trace(ax, d["iterations"], d["pka"] - d["pka_ref"],
                        C_ON, "per-frame correction")

        if ens_off is not None:
            _plot_ensemble(ax, ens_off, C_OFF, "uncorrected")
        elif off_key in runs_data and len(runs_data[off_key]["pka"]) > 0:
            d = runs_data[off_key]
            _plot_trace(ax, d["iterations"], d["pka"] - d["pka_ref"],
                        C_OFF, "uncorrected")

        ax.axhline(0, color="k", ls="--", lw=0.6, alpha=0.4)

        # Theory prediction for the wrong fixed point (gshift off)
        if sys_name == "glu":
            ax.axhline(-np.log10(2), color=C_OFF[:3], ls=":", lw=1.0, alpha=0.5)
            ax.text(0.97, 0.05, r"$-\log_{10}2$", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6, color=C_OFF[:3],
                    alpha=0.7)

        ax.set_xlabel("ALF iteration")
        if ax == axes[0]:
            ax.set_ylabel(r"$\Delta$pKa (estimated $-$ reference)")
        ax.set_title(panel_label, fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
        ax.set_ylim(-2.5, 2.5)
        clean_axes(ax)

    fig.savefig(output_dir / "fig_md_convergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir / 'fig_md_convergence.png'}")


def plot_K_independence(runs_data: dict, runs_dir: Path, output_dir: Path):
    """Fig 2: Final ΔpKa vs K for GLU (gshift off).

    Uses ensemble seeds when available.
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

    K_values, pka_means, pka_sems = [], [], []

    for k in [3, 5, 7]:
        # Try ensemble
        if k == 5:
            base = "glu_gshift_off"
        else:
            base = f"glu_K{k}_off"

        ens = extract_ensemble(runs_dir, base)
        if ens is not None and ens["n_seeds"] >= 2:
            # Use final 20 iterations from each seed
            final_deltas = []
            for seed_data in ens["individual"]:
                tail = seed_data["delta_raw"][-20:]
                final_deltas.append(np.nanmean(tail))
            K_values.append(k)
            pka_means.append(np.mean(final_deltas))
            pka_sems.append(np.std(final_deltas, ddof=1) / np.sqrt(len(final_deltas)))
        else:
            # Fallback to single run
            key = f"glu_k{k}_off"
            if key in runs_data and len(runs_data[key]["pka"]) >= 10:
                d = runs_data[key]
                tail = d["pka"][-20:]
                tail = tail[~np.isnan(tail)]
                if len(tail) > 0:
                    K_values.append(k)
                    pka_means.append(np.mean(tail) - d["pka_ref"])
                    pka_sems.append(np.std(tail) / np.sqrt(len(tail)))

    if K_values:
        ax.bar(K_values, pka_means, yerr=pka_sems, capsize=4,
               color=(*C_OFF[:3], 0.7), edgecolor=C_OFF[:3],
               width=0.8, linewidth=0.8)

        # Theory line
        theory = np.log10(2)
        ax.axhline(theory, color=C_ON[:3], ls="--", lw=1.5, alpha=0.8,
                   label=rf"theory: $+\log_{{10}}(N_D) = +{theory:.3f}$")
        ax.axhline(-theory, color=C_ON[:3], ls=":", lw=1.0, alpha=0.5,
                   label=r"$-\log_{10}(N_D)$")
        ax.axhline(0, color="k", ls="-", lw=0.4, alpha=0.3)

        ax.set_xlabel("Number of replicas ($K$)")
        ax.set_ylabel(r"$\Delta$pKa (last 20 iters)")
        ax.set_title("GLU: pKa error vs $K$ (uncorrected)", fontsize=9,
                     fontweight="bold")
        ax.set_xticks(K_values)
        ax.legend(fontsize=6, loc="lower left")
    else:
        ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes, ha="center")

    clean_axes(ax)
    fig.savefig(output_dir / "fig_md_K_independence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir / 'fig_md_K_independence.png'}")


def plot_W_dependence(runs_data: dict, runs_dir: Path, output_dir: Path):
    """Fig 3: ΔpKa convergence traces for LYS — varying W, gshift off vs on.

    Uses ensemble seeds when available.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0), constrained_layout=True)

    W_list = [1, 3, 5, 10]
    off_cmap = plt.cm.Oranges(np.linspace(0.4, 0.95, len(W_list)))
    on_cmap = plt.cm.Blues(np.linspace(0.4, 0.95, len(W_list)))

    # Left: gshift OFF
    ax = axes[0]
    for w, color in zip(W_list, off_cmap):
        ens = extract_ensemble(runs_dir, f"lys_W{w}_off")
        if ens is not None:
            _plot_ensemble(ax, ens, color, f"$W\\!={w}$")
        else:
            key = f"lys_w{w}_off"
            if key in runs_data and len(runs_data[key]["pka"]) > 0:
                d = runs_data[key]
                _plot_trace(ax, d["iterations"], d["pka"] - d["pka_ref"],
                            color, f"$W\\!={w}$", smooth_span=15)

    ax.axhline(0, color="k", ls="--", lw=0.6, alpha=0.4)
    ax.set_xlabel("ALF iteration")
    ax.set_ylabel(r"$\Delta$pKa")
    ax.set_title("(a) LYS uncorrected — varying $W$", fontsize=9,
                 fontweight="bold")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_ylim(-2.5, 2.5)
    clean_axes(ax)

    # Right: gshift ON
    ax = axes[1]
    for w, color in zip(W_list, on_cmap):
        ens = extract_ensemble(runs_dir, f"lys_W{w}_on")
        if ens is not None:
            _plot_ensemble(ax, ens, color, f"$W\\!={w}$")
        else:
            key = f"lys_w{w}_on"
            if key in runs_data and len(runs_data[key]["pka"]) > 0:
                d = runs_data[key]
                _plot_trace(ax, d["iterations"], d["pka"] - d["pka_ref"],
                            color, f"$W\\!={w}$", smooth_span=15)

    ax.axhline(0, color="k", ls="--", lw=0.6, alpha=0.4)
    ax.set_xlabel("ALF iteration")
    ax.set_title("(b) LYS per-frame correction — varying $W$", fontsize=9,
                 fontweight="bold")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_ylim(-2.5, 2.5)
    clean_axes(ax)

    fig.savefig(output_dir / "fig_md_W_dependence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir / 'fig_md_W_dependence.png'}")


# ---------------------------------------------------------------------------
# Profile convergence (from bias parameters b, c, s, x)
# ---------------------------------------------------------------------------

# Bias energy constants (derived from FNEX=5.5 via cphmd/core/bias_constants.py)
from cphmd.core.bias_constants import CHI_OFFSET, OMEGA_DECAY


def _load_bias_params(analysis_dir: Path) -> tuple:
    """Load b_sum, c_sum, s_sum, x_sum from analysis directory."""
    b = np.loadtxt(analysis_dir / "b_sum.dat")
    c = np.loadtxt(analysis_dir / "c_sum.dat")
    s = np.loadtxt(analysis_dir / "s_sum.dat")
    x = np.loadtxt(analysis_dir / "x_sum.dat")
    if b.ndim == 0:
        b = np.array([float(b)])
    ns = len(b)
    if c.ndim == 1:
        c = c.reshape(ns, ns)
    if s.ndim == 1:
        s = s.reshape(ns, ns)
    if x.ndim == 1:
        x = x.reshape(ns, ns)
    return b, c, s, x


def _bias_energy_at_lambda(lam: np.ndarray, b, c, s, x) -> float:
    """Total bias energy E_b + E_c + E_s + E_x at a given λ vector.

    Matches CHARMM BLOCK convention and CUDA reactioncoord kernels:
      E_b = -Σ b_i λ_i              (phi — class 1 LDBV)
      E_c = ½ Σ c_{ij} λ_i λ_j     (psi — class 2 LDBV)
      E_s = Σ s_{ij} [λ_i/(λ_i+ε)] λ_j  (omega — class 8 LDBV, ε=CHI_OFFSET)
      E_x = Σ x_{ij} λ_j [1-exp(-α λ_i)]  (chi — class 10 LDBV, α=OMEGA_DECAY)
    """
    E_b = -np.dot(lam, b)
    E_c = 0.5 * float(np.einsum("i,ij,j->", lam, c, lam))
    ratio = lam / (lam + CHI_OFFSET)
    E_s = float(np.einsum("ij,i,j->", s, ratio, lam))
    exp_term = 1.0 - np.exp(OMEGA_DECAY * lam)
    E_x = float(np.einsum("ij,j,i->", x, lam, exp_term))
    return E_b + E_c + E_s + E_x


def _compute_profile_general(analysis_dir: Path, n_points: int = 200):
    """Compute bias energy profile along 0→j transition paths.

    For 2-state: returns (t, E) with n_points values.
    For N-state: returns (t, E) concatenating (N-1) transition paths,
    each 0→j with n_points values → (N-1)*n_points total.
    """
    b, c, s, x = _load_bias_params(analysis_dir)
    ns = len(b)
    t_grid = np.linspace(0, 1, n_points)
    E_parts = []
    for sub_to in range(1, ns):
        E_1d = np.empty(n_points)
        for k, tk in enumerate(t_grid):
            lam = np.zeros(ns)
            lam[0] = 1.0 - tk
            lam[sub_to] = tk
            E_1d[k] = _bias_energy_at_lambda(lam, b, c, s, x)
        E_parts.append(E_1d)
    E = np.concatenate(E_parts)
    t_all = np.concatenate([t_grid + i for i in range(ns - 1)])
    return t_all, E


def plot_profile_convergence(runs_dir: Path, output_dir: Path):
    """Fig: 1D bias energy profile at selected iterations, on vs off.

    Shows how the corrected and uncorrected runs flatten the landscape differently.
    Uses seed 1 for each condition (representative).
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), constrained_layout=True)

    systems = [
        ("lys", "LYS — 2-state"),
        ("glu", "GLU — 3-state"),
        ("hsp", "HSP — 3-state"),
    ]

    snapshot_iters = [10, 30, 60, 100, 150]
    cmap_on = plt.cm.Blues(np.linspace(0.3, 0.9, len(snapshot_iters)))
    cmap_off = plt.cm.Oranges(np.linspace(0.3, 0.9, len(snapshot_iters)))

    for ax, (sys_name, title) in zip(axes, systems):
        # Prefer whichever run has the most iterations
        def _pick_best_run(base_name):
            candidates = [runs_dir / base_name]
            for i in range(1, 10):
                candidates.append(runs_dir / f"{base_name}_s{i}")
            best, best_n = None, 0
            for c in candidates:
                sol = c / "solvated"
                if not sol.exists():
                    continue
                n = sum(1 for d in sol.iterdir()
                        if d.name.startswith("analysis") and (d / "b_sum.dat").exists())
                if n > best_n:
                    best, best_n = c, n
            return best if best else candidates[0]

        on_dir = _pick_best_run(f"{sys_name}_gshift_on")
        off_dir = _pick_best_run(f"{sys_name}_gshift_off")

        has_data = False
        for i_iter, (snap, c_on, c_off) in enumerate(
            zip(snapshot_iters, cmap_on, cmap_off)
        ):
            # ON
            adir_on = on_dir / "solvated" / f"analysis{snap}"
            if adir_on.exists() and (adir_on / "b_sum.dat").exists():
                try:
                    t, E = _compute_profile_general(adir_on)
                    E -= E[0]  # Shift so E(λ=0) = 0
                    ax.plot(t, E, color=c_on, lw=1.2,
                            label=f"on {snap}" if i_iter == 0 or i_iter == len(snapshot_iters) - 1 else None)
                    has_data = True
                except Exception:
                    pass

            # OFF
            adir_off = off_dir / "solvated" / f"analysis{snap}"
            if adir_off.exists() and (adir_off / "b_sum.dat").exists():
                try:
                    t, E = _compute_profile_general(adir_off)
                    E -= E[0]
                    ax.plot(t, E, color=c_off, lw=1.2, ls="--",
                            label=f"off {snap}" if i_iter == 0 or i_iter == len(snapshot_iters) - 1 else None)
                    has_data = True
                except Exception:
                    pass

        if not has_data:
            ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                    ha="center", fontsize=8, alpha=0.5)

        ax.axhline(0, color="k", ls="-", lw=0.4, alpha=0.3)
        ax.set_xlabel(r"$\lambda$")
        if ax == axes[0]:
            ax.set_ylabel(r"$\Delta E_{\mathrm{bias}}$ (kcal/mol)")
        ax.set_title(title, fontsize=9, fontweight="bold")
        if has_data:
            ax.legend(fontsize=5, loc="best", ncol=2)
        clean_axes(ax)

    fig.savefig(output_dir / "fig_md_profile_convergence.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir / 'fig_md_profile_convergence.png'}")


def plot_bsum_convergence(runs_dir: Path, output_dir: Path):
    """Fig: b_sum-derived ΔpKa convergence trajectory (ensemble).

    Complementary to population-derived pKa: b_sum is the direct WHAM output,
    smoother and antiphase (b_sum_error = -pKa_error × kT·ln10).
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), constrained_layout=True)

    kTln10 = KT * np.log(10)

    systems = [
        ("lys", "LYS — 2-state", 10.4),
        ("glu", "GLU — 3-state", 4.25),
        ("hsp", "HSP — 3-state", None),
    ]

    labels_panel = ["(a)", "(b)", "(c)"]

    for ax, (sys_name, title, pka_ref), lbl in zip(axes, systems, labels_panel):
        for condition, color, label in [
            ("on", C_ON, "per-frame correction"),
            ("off", C_OFF, "uncorrected"),
        ]:
            base = f"{sys_name}_gshift_{condition}"
            seed_dirs = find_seed_dirs(runs_dir, base)
            if not seed_dirs:
                # Try single run
                single = runs_dir / base
                if (single / "solvated").exists():
                    seed_dirs = [single]

            if not seed_dirs:
                continue

            all_iters, all_deltas = [], []
            for sd in seed_dirs:
                iters, deltas = [], []
                for i in range(1, 300):
                    f = sd / "solvated" / f"analysis{i}" / "b_sum.dat"
                    if not f.exists():
                        continue
                    vals = [float(v) for v in f.read_text().split()]
                    if len(vals) < 2:
                        continue
                    # For 2-state: ΔpKa from b_sum = -(b[1]-b[0]) / (kT*ln10)
                    # For 3-state: use (b[1]+b[2])/2 - b[0] as proxy
                    if len(vals) == 2:
                        dpka = -(vals[1] - vals[0]) / kTln10
                    else:
                        dpka = -(np.mean(vals[1:]) - vals[0]) / kTln10
                    iters.append(i)
                    deltas.append(dpka)

                if len(iters) < 3:
                    continue
                iters = np.array(iters)
                deltas = np.array(deltas)
                # Subtract pka_ref to get ΔpKa
                if pka_ref is not None:
                    deltas = deltas - pka_ref
                sm = ewma(deltas, span=12)
                all_iters.append(iters)
                all_deltas.append(sm)

                # Faint individual trace
                ax.plot(iters, deltas, color=(*color[:3], 0.08), lw=0.4)

            if not all_deltas:
                continue

            # Ensemble mean ± SEM (ragged)
            n_s = len(all_deltas)
            iters_r, mean, sem = ragged_mean_sem(all_deltas, all_iters)

            ax.plot(iters_r, mean, color=color, lw=2.5,
                    label=f"{label} ($n$={n_s})")
            valid = ~np.isnan(sem)
            if valid.any():
                ax.fill_between(iters_r[valid],
                                mean[valid] - sem[valid],
                                mean[valid] + sem[valid],
                                color=(*color[:3], 0.2), lw=0)

        ax.axhline(0, color="k", ls="--", lw=0.6, alpha=0.4)
        if sys_name == "glu":
            ax.axhline(-np.log10(2), color=C_OFF[:3], ls=":", lw=1.0, alpha=0.5)

        ax.set_xlabel("ALF iteration")
        if ax == axes[0]:
            ax.set_ylabel(r"$\Delta$pKa (from $b_{\mathrm{sum}}$)")
        ax.set_title(f"{lbl} {title}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
        ax.set_ylim(-2.5, 2.5)
        clean_axes(ax)

    fig.savefig(output_dir / "fig_md_bsum_convergence.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir / 'fig_md_bsum_convergence.png'}")


def _profile_rmsd_trajectory(run_dir: Path, n_points: int = 200) -> tuple:
    """Compute inter-iteration profile RMSD for a single run.

    Returns (iters, rmsds) arrays.
    """
    sol = run_dir / "solvated"
    prev_E = None
    iters, rmsds = [], []
    for i in range(1, 300):
        adir = sol / f"analysis{i}"
        if not (adir / "b_sum.dat").exists():
            continue
        try:
            _, E = _compute_profile_general(adir, n_points)
        except Exception:
            continue

        if prev_E is not None and len(E) == len(prev_E):
            rmsd = np.sqrt(np.mean((E - prev_E) ** 2))
            iters.append(i)
            rmsds.append(rmsd)
        prev_E = E

    return np.array(iters), np.array(rmsds)


def plot_profile_rmsd(runs_dir: Path, output_dir: Path):
    """Fig: Profile RMSD convergence — how fast the energy landscape stabilizes.

    Shows inter-iteration RMSD of the full bias profile E(λ; b,c,s,x) for
    corrected vs uncorrected, with ensemble bands.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), constrained_layout=True)

    systems = [
        ("lys", "(a) LYS — 2-state"),
        ("glu", "(b) GLU — 3-state"),
        ("hsp", "(c) HSP — 3-state"),
    ]

    for ax, (sys_name, title) in zip(axes, systems):
        for condition, color, label in [
            ("on", C_ON, "corrected"),
            ("off", C_OFF, "uncorrected"),
        ]:
            base = f"{sys_name}_gshift_{condition}"
            seed_dirs = find_seed_dirs(runs_dir, base)
            if not seed_dirs:
                single = runs_dir / base
                if (single / "solvated").exists():
                    seed_dirs = [single]

            all_data = []
            for sd in seed_dirs:
                it, rm = _profile_rmsd_trajectory(sd)
                if len(it) < 5:
                    continue
                sm = ewma(rm, span=8)
                all_data.append({"iters": it, "rmsd": sm})

            if not all_data:
                continue

            n_s = len(all_data)
            iters_r, mean, sem = ragged_mean_sem(
                [d["rmsd"] for d in all_data],
                [d["iters"] for d in all_data])

            # Faint individual traces
            for d in all_data:
                ax.semilogy(d["iters"], d["rmsd"],
                            color=(*color[:3], 0.1), lw=0.4)

            ax.semilogy(iters_r, mean, color=color, lw=2.5,
                        label=f"{label} ($n$={n_s})")
            valid = ~np.isnan(sem)
            if valid.any():
                ax.fill_between(iters_r[valid],
                                mean[valid] - sem[valid],
                                mean[valid] + sem[valid],
                                color=(*color[:3], 0.2), lw=0)

        ax.set_xlabel("ALF iteration")
        if ax == axes[0]:
            ax.set_ylabel("Profile RMSD (kcal/mol)")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
        clean_axes(ax)

    fig.savefig(output_dir / "fig_md_profile_rmsd.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir / 'fig_md_profile_rmsd.png'}")


def _build_reference_profile(runs_dir: Path, sys_name: str, n_points: int = 200,
                              min_iters: int = 50):
    """Build reference profile from ensemble mean of well-converged ON runs.

    Uses the final profile from each ON seed (and the original single run) that
    has at least ``min_iters`` completed iterations.  Returns (t, E_ref) where
    E_ref is zero-shifted so E(λ=0) = 0.
    """
    candidates = [runs_dir / f"{sys_name}_gshift_on"]
    for i in range(1, 20):
        candidates.append(runs_dir / f"{sys_name}_gshift_on_s{i}")

    profiles = []
    for run_dir in candidates:
        sol = run_dir / "solvated"
        if not sol.exists():
            continue
        # Find last complete iteration (has b_sum.dat)
        max_iter = 0
        for d in sol.iterdir():
            if d.name.startswith("analysis") and (d / "b_sum.dat").exists():
                try:
                    n = int(d.name[8:])
                    if n > max_iter:
                        max_iter = n
                except ValueError:
                    pass
        if max_iter < min_iters:
            continue
        try:
            _, E = _compute_profile_general(sol / f"analysis{max_iter}", n_points)
            E = E - E[0]  # shift so E(λ=0) = 0
            profiles.append(E)
        except Exception:
            continue

    if not profiles:
        return None, None
    ref = np.mean(profiles, axis=0)
    # t-axis length matches ref (n_points per transition path)
    n_transitions = len(ref) // n_points
    t = np.concatenate([np.linspace(0, 1, n_points) + i for i in range(n_transitions)])
    return t, ref


def _profile_rmsd_to_ref_trajectory(run_dir: Path, ref_profile: np.ndarray,
                                     n_points: int = 200):
    """Compute RMSD of each iteration's profile to a reference profile.

    Returns (iters, rmsds, endpoint_errors) arrays.
    """
    sol = run_dir / "solvated"
    iters, rmsds, ep_errors = [], [], []
    for i in range(1, 300):
        adir = sol / f"analysis{i}"
        if not (adir / "b_sum.dat").exists():
            continue
        try:
            _, E = _compute_profile_general(adir, n_points)
        except Exception:
            continue

        E = E - E[0]
        if len(E) != len(ref_profile):
            continue
        rmsd = np.sqrt(np.mean((E - ref_profile) ** 2))
        ep_err = abs(E[-1] - ref_profile[-1])
        iters.append(i)
        rmsds.append(rmsd)
        ep_errors.append(ep_err)

    return np.array(iters), np.array(rmsds), np.array(ep_errors)


def plot_profile_rmsd_to_ref(runs_dir: Path, output_dir: Path):
    """Fig: Profile RMSD to reference — corrected vs uncorrected.

    The reference is the final converged ensemble-average ON profile.
    ON runs should converge to RMSD ≈ 0 (noise floor).
    OFF runs should plateau at a nonzero floor (the structural gshift error).

    Two-row figure: top = full RMSD, bottom = endpoint error |E(1) - E_ref(1)|.
    """
    systems = [
        ("lys", "(a) LYS"),
        ("glu", "(b) GLU"),
        ("hsp", "(c) HSP"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.0), constrained_layout=True)

    for col, (sys_name, label) in enumerate(systems):
        # Build reference
        _, ref = _build_reference_profile(runs_dir, sys_name)
        if ref is None:
            for row in range(2):
                axes[row, col].text(0.5, 0.5, "No reference\navailable",
                                    transform=axes[row, col].transAxes,
                                    ha="center", fontsize=8, alpha=0.5)
            continue

        # n_points is the per-transition grid size (200), NOT len(ref)
        # For 3-state: ref has 2*200=400 points, but _profile_rmsd_to_ref_trajectory
        # needs the per-transition value (200) to build matching profiles.
        n_points = 200

        for condition, color, cond_label in [
            ("on", C_ON, "corrected"),
            ("off", C_OFF, "uncorrected"),
        ]:
            base = f"{sys_name}_gshift_{condition}"
            seed_dirs = find_seed_dirs(runs_dir, base)
            if not seed_dirs:
                single = runs_dir / base
                if (single / "solvated").exists():
                    seed_dirs = [single]

            all_rmsd, all_ep = [], []
            for sd in seed_dirs:
                it, rm, ep = _profile_rmsd_to_ref_trajectory(sd, ref, n_points)
                if len(it) < 5:
                    continue
                sm_rm = ewma(rm, span=8)
                sm_ep = ewma(ep, span=8)
                all_rmsd.append({"iters": it, "vals": sm_rm})
                all_ep.append({"iters": it, "vals": sm_ep})

            if not all_rmsd:
                continue

            # RMSD panel (top row)
            ax_rm = axes[0, col]
            n_s = len(all_rmsd)
            iters_r, mean_rm, sem_rm = ragged_mean_sem(
                [d["vals"] for d in all_rmsd],
                [d["iters"] for d in all_rmsd])

            for d in all_rmsd:
                ax_rm.plot(d["iters"], d["vals"],
                           color=(*color[:3], 0.12), lw=0.4)
            ax_rm.plot(iters_r, mean_rm, color=color, lw=2.5,
                       label=f"{cond_label} ($n$={n_s})")
            valid = ~np.isnan(sem_rm)
            if valid.any():
                ax_rm.fill_between(iters_r[valid],
                                   mean_rm[valid] - sem_rm[valid],
                                   mean_rm[valid] + sem_rm[valid],
                                   color=(*color[:3], 0.2), lw=0)

            # Endpoint error panel (bottom row)
            ax_ep = axes[1, col]
            iters_e, mean_ep, sem_ep = ragged_mean_sem(
                [d["vals"] for d in all_ep],
                [d["iters"] for d in all_ep])

            for d in all_ep:
                ax_ep.plot(d["iters"], d["vals"],
                           color=(*color[:3], 0.12), lw=0.4)
            ax_ep.plot(iters_e, mean_ep, color=color, lw=2.5,
                       label=f"{cond_label} ($n$={n_s})")
            valid_e = ~np.isnan(sem_ep)
            if valid_e.any():
                ax_ep.fill_between(iters_e[valid_e],
                                   mean_ep[valid_e] - sem_ep[valid_e],
                                   mean_ep[valid_e] + sem_ep[valid_e],
                                   color=(*color[:3], 0.2), lw=0)

        for row, ylabel, title_suffix in [
            (0, "Profile RMSD (kcal/mol)", "full profile"),
            (1, r"$|\Delta E(\lambda\!=\!1)|$ (kcal/mol)", "endpoint"),
        ]:
            ax = axes[row, col]
            ax.set_yscale("log")
            ax.set_xlabel("ALF iteration")
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(f"{label} — {title_suffix}", fontsize=8,
                         fontweight="bold")
            ax.legend(fontsize=5, loc="upper right", framealpha=0.8)
            ax.set_ylim(0.05, 100)
            clean_axes(ax)

    fig.savefig(output_dir / "fig_md_profile_rmsd_to_ref.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir / 'fig_md_profile_rmsd_to_ref.png'}")


def print_summary(runs_data: dict):
    print("\n" + "=" * 80)
    print(f"{'Run':<24s} {'System':<6s} {'pKa_ref':>8s} {'pKa_final':>10s} "
          f"{'ΔpKa':>8s} {'Phase':>6s} {'Iters':>6s}")
    print("-" * 80)

    for name in sorted(runs_data.keys()):
        d = runs_data[name]
        n = len(d["pka"])
        if n == 0:
            print(f"{name:<24s} {d['system']:<6s} {d['pka_ref']:>8.2f} {'N/A':>10s} "
                  f"{'N/A':>8s} {'N/A':>6s} {0:>6d}")
            continue

        final_pka = np.nanmean(d["pka"][-10:]) if n >= 10 else np.nanmean(d["pka"])
        delta = final_pka - d["pka_ref"]
        last_phase = d["phase"][-1] if len(d["phase"]) > 0 else 0

        print(f"{name:<24s} {d['system']:<6s} {d['pka_ref']:>8.2f} {final_pka:>10.3f} "
              f"{delta:>+8.3f} {last_phase:>6d} {n:>6d}")

    print("=" * 80)


import csv

from scipy.optimize import curve_fit


def _extract_ref_pka_from_csv(csv_path: Path) -> float | None:
    """Extract macroscopic reference pKa from HH CSV tag_pKa values.

    For 2-state: returns the single tag_pKa (= macro-pKa).
    For 3-state UPOS (protonation, e.g. HSP): macro = -log10(sum(10^(-pKa_i)))
    For 3-state UNEG (deprotonation, e.g. GLU): macro = micro + log10(N_D)
      (equivalent to -log10(sum(10^(-pKa_i))) only when tag is UPOS)
    """
    try:
        data = _read_hh_csv(csv_path)
    except Exception:
        return None
    tag_pkas = []
    tag_type = None
    for sid in sorted(data.keys()):
        if sid == "s1s1":
            continue
        tp = data[sid]["tag_pKa"]
        if tp:
            tag_pkas.append(float(tp))
        if tag_type is None:
            tag_type = data[sid].get("tag_type", "")
    if not tag_pkas:
        return None
    if len(tag_pkas) == 1:
        return float(tag_pkas[0])
    # Multiple tautomers: compute macroscopic pKa
    if tag_type == "UPOS":
        # Protonation events (HSP): macro = -log10(sum 10^(-pKa_i))
        return float(-np.log10(sum(10.0 ** (-p) for p in tag_pkas)))
    else:
        # Deprotonation events (GLU/UNEG): macro = micro + log10(N_D)
        # For equal tautomers; for unequal, use weighted formula
        if abs(tag_pkas[0] - tag_pkas[1]) < 0.01:
            return float(tag_pkas[0] + np.log10(len(tag_pkas)))
        else:
            # Unequal UNEG tautomers (rare): macro = -log10(1/sum(10^pKa_i))
            return float(np.log10(sum(10.0 ** p for p in tag_pkas)))
    return None


def _hh_prot(pH, pKa):
    """Henderson-Hasselbalch: fraction protonated."""
    return 1.0 / (1.0 + 10.0 ** (pH - pKa))


def _hh_deprot(pH, pKa):
    """Henderson-Hasselbalch: fraction deprotonated."""
    return 1.0 / (1.0 + 10.0 ** (pKa - pH))


def _read_hh_csv(csv_path: Path) -> dict:
    """Parse HH CSV into {state_id: [(pH, pop_obs, pop_theo), ...]}."""
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["state_id"]
            if sid not in data:
                data[sid] = {"pH": [], "pop": [], "theo": [],
                             "tag_type": row.get("tag_type", ""),
                             "tag_pKa": row.get("tag_pKa", "")}
            data[sid]["pH"].append(float(row["pH"]))
            data[sid]["pop"].append(float(row["population"]))
            data[sid]["theo"].append(float(row["theoretical_population"]))
    return data


def _fit_pka_from_hh(csv_path: Path, pka_ref: float) -> float | None:
    """Fit macroscopic pKa from HH CSV by least-squares on s1s1.

    The s1s1 state can be either protonated or deprotonated depending on
    the tag type:
      UPOS (LYS, HSP): s1s1 = protonated → P = 1/(1 + 10^(pH - pKa))
      UNEG (GLU):       s1s1 = deprotonated → P = 1/(1 + 10^(pKa - pH))

    For 2-state and 3-state equal tautomers, a simple 1-parameter HH fit
    gives macro-pKa directly. For unequal tautomers (HSP), we fit 2
    micro-pKas and derive macro-pKa.

    Returns the macroscopic pKa (LYS=10.4, GLU=4.25, HSP=6.45).
    """
    try:
        data = _read_hh_csv(csv_path)
    except Exception:
        return None

    states = sorted(data.keys())
    n_states = len(states)

    if "s1s1" not in data:
        return None

    s = data["s1s1"]
    pH = np.array(s["pH"])
    pop = np.array(s["pop"])

    # Determine tag type and collect micro-pKas from tautomers
    tag_pkas = []
    tag_type = None
    for sid in states:
        if sid == "s1s1":
            continue
        tp = data[sid]["tag_pKa"]
        if tp:
            tag_pkas.append(float(tp))
        if tag_type is None:
            tag_type = data[sid].get("tag_type", "")

    # Choose HH model based on tag: UPOS → s1s1 is protonated, UNEG → deprotonated
    if tag_type == "UNEG":
        hh_model = _hh_deprot  # P_deprot = 1/(1 + 10^(pKa - pH))
    else:
        hh_model = _hh_prot    # P_prot = 1/(1 + 10^(pH - pKa))

    # All systems: simple 1-parameter HH fit on s1s1 → macro-pKa directly.
    # This works because all states are bound in CpHMD and the s1s1
    # fraction follows a simple HH curve for the macroscopic pKa.
    try:
        popt, _ = curve_fit(hh_model, pH, pop, p0=[pka_ref],
                            bounds=(pka_ref - 5, pka_ref + 5))
        return float(popt[0])
    except Exception:
        return None


# Global cache for pre-computed trajectories
_HH_FIT_CACHE: dict[str, tuple] = {}
_POP_RMSD_CACHE: dict[str, tuple] = {}


def _hh_fit_pka_trajectory(run_dir: Path, pka_ref: float) -> tuple:
    """Extract fitted pKa trajectory from HH CSVs.

    Returns (iters, pkas, detected_ref) where detected_ref is the reference
    pKa extracted from the CSV tag_pKa values (or pka_ref if unavailable).
    Uses global cache if available.
    """
    key = str(run_dir)
    if key in _HH_FIT_CACHE:
        return _HH_FIT_CACHE[key]

    hh_dir = run_dir / "solvated" / "plots" / "hh_plots"
    if not hh_dir.exists():
        result = np.array([]), np.array([]), pka_ref
        _HH_FIT_CACHE[key] = result
        return result

    # Auto-detect reference from first available CSV
    detected_ref = pka_ref
    for i in range(1, 300):
        f = hh_dir / f"data_run{i}.csv"
        if f.exists():
            ref = _extract_ref_pka_from_csv(f)
            if ref is not None:
                detected_ref = ref
            break

    iters, pkas = [], []
    for i in range(1, 300):
        f = hh_dir / f"data_run{i}.csv"
        if not f.exists():
            continue
        pka = _fit_pka_from_hh(f, detected_ref)
        if pka is not None:
            iters.append(i)
            pkas.append(pka)
    result = np.array(iters), np.array(pkas), detected_ref
    _HH_FIT_CACHE[key] = result
    return result


def plot_hh_convergence(runs_dir: Path, output_dir: Path):
    """Fig: ΔpKa convergence from HH-fitted pKa (5-point least-squares fit).

    Uses all pH points simultaneously for a robust pKa estimate at each
    iteration. Ensemble mean ± SEM with faint individual traces.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), constrained_layout=True)

    systems = [
        ("lys", r"(a) LYS — 2-state", 10.4),
        ("glu", r"(b) GLU — 3-state ($N_D\!=\!2$)", 4.25),
        ("hsp", r"(c) HSP — 3-state (unequal)", 6.45),
    ]

    for ax, (sys_name, title, pka_ref) in zip(axes, systems):
        for condition, color, label in [
            ("on", C_ON, "corrected"),
            ("off", C_OFF, "uncorrected"),
        ]:
            base = f"{sys_name}_gshift_{condition}"
            seed_dirs = find_all_runs(runs_dir, base)

            all_data = []
            for sd in seed_dirs:
                it, pka, ref = _hh_fit_pka_trajectory(sd, pka_ref)
                if len(it) < 5:
                    continue
                delta = pka - ref
                sm = ewma(delta, span=12)
                all_data.append({"iters": it, "delta": delta, "smooth": sm})

            if not all_data:
                continue

            # Faint individual traces
            for d in all_data:
                ax.plot(d["iters"], d["delta"],
                        color=(*color[:3], 0.1), lw=0.4, zorder=1)

            # Ensemble mean ± SEM (ragged: use all data at each iter)
            n_s = len(all_data)
            max_len = max(len(d["iters"]) for d in all_data)
            iters_full = all_data[
                max(range(n_s), key=lambda i: len(all_data[i]["iters"]))
            ]["iters"]
            mean = np.full(max_len, np.nan)
            sem = np.full(max_len, np.nan)
            for j in range(max_len):
                vals = [d["smooth"][j] for d in all_data if j < len(d["smooth"])]
                if vals:
                    mean[j] = np.mean(vals)
                    if len(vals) > 1:
                        sem[j] = np.std(vals, ddof=1) / np.sqrt(len(vals))

            ax.plot(iters_full, mean, color=color, lw=2.5,
                    label=f"{label} ($n$={n_s})", zorder=5)
            valid = ~np.isnan(sem)
            ax.fill_between(iters_full[valid],
                            mean[valid] - sem[valid],
                            mean[valid] + sem[valid],
                            color=(*color[:3], 0.2), lw=0, zorder=3)

        ax.axhline(0, color="k", ls="--", lw=0.6, alpha=0.4)
        if sys_name == "glu":
            ax.axhline(-np.log10(2), color=C_OFF[:3], ls=":", lw=1.0, alpha=0.5)
            ax.text(0.97, 0.05, r"$-\log_{10}2$", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6, color=C_OFF[:3],
                    alpha=0.7)

        ax.set_xlabel("ALF iteration")
        if ax == axes[0]:
            ax.set_ylabel(r"$\Delta$pKa (HH fit $-$ reference)")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
        ax.set_ylim(-2.5, 2.5)
        clean_axes(ax)

    fig.savefig(output_dir / "fig_md_hh_convergence.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir / 'fig_md_hh_convergence.png'}")


def _read_hh_pop_rmsd(csv_path: Path) -> float | None:
    """Compute RMSD of observed vs theoretical populations from HH CSV.

    Uses all states and all pH points. Returns None if file is unreadable.
    """
    import csv
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            diffs = []
            for row in reader:
                diffs.append(float(row["population"]) - float(row["theoretical_population"]))
        if not diffs:
            return None
        return np.sqrt(np.mean(np.array(diffs) ** 2))
    except Exception:
        return None


def _hh_pop_rmsd_trajectory(run_dir: Path) -> tuple:
    """Extract population RMSD trajectory from HH CSVs for a single run.

    Returns (iters, rmsds) arrays.  Uses global cache if available.
    """
    key = str(run_dir)
    if key in _POP_RMSD_CACHE:
        return _POP_RMSD_CACHE[key]

    hh_dir = run_dir / "solvated" / "plots" / "hh_plots"
    if not hh_dir.exists():
        result = np.array([]), np.array([])
        _POP_RMSD_CACHE[key] = result
        return result
    iters, rmsds = [], []
    for i in range(1, 300):
        f = hh_dir / f"data_run{i}.csv"
        if not f.exists():
            continue
        rmsd = _read_hh_pop_rmsd(f)
        if rmsd is not None:
            iters.append(i)
            rmsds.append(rmsd)
    result = np.array(iters), np.array(rmsds)
    _POP_RMSD_CACHE[key] = result
    return result


def plot_pop_rmsd(runs_dir: Path, output_dir: Path):
    """Fig: Population RMSD convergence from HH fits.

    Compares observed protonation-state populations against theoretical
    Henderson-Hasselbalch predictions at each pH point. ON runs should
    converge to a small noise floor; OFF runs should plateau at a higher
    value reflecting the systematic gshift error in the populations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), constrained_layout=True)

    systems = [
        ("lys", "(a) LYS — 2-state"),
        ("glu", "(b) GLU — 3-state"),
        ("hsp", "(c) HSP — 3-state"),
    ]

    for ax, (sys_name, title) in zip(axes, systems):
        for condition, color, label in [
            ("on", C_ON, "corrected"),
            ("off", C_OFF, "uncorrected"),
        ]:
            base = f"{sys_name}_gshift_{condition}"
            seed_dirs = find_all_runs(runs_dir, base)

            all_data = []
            for sd in seed_dirs:
                it, rm = _hh_pop_rmsd_trajectory(sd)
                if len(it) < 5:
                    continue
                sm = ewma(rm, span=8)
                all_data.append({"iters": it, "vals": sm})

            if not all_data:
                continue

            n_s = len(all_data)
            iters_r, mean, sem = ragged_mean_sem(
                [d["vals"] for d in all_data],
                [d["iters"] for d in all_data])

            # Faint individual traces
            for d in all_data:
                ax.plot(d["iters"], d["vals"],
                        color=(*color[:3], 0.12), lw=0.4)

            ax.plot(iters_r, mean, color=color, lw=2.5,
                    label=f"{label} ($n$={n_s})")
            valid = ~np.isnan(sem)
            if valid.any():
                ax.fill_between(iters_r[valid],
                                mean[valid] - sem[valid],
                                mean[valid] + sem[valid],
                                color=(*color[:3], 0.2), lw=0)

        ax.axhline(0, color="k", ls="-", lw=0.4, alpha=0.3)
        ax.set_xlabel("ALF iteration")
        if ax == axes[0]:
            ax.set_ylabel("Population RMSD\n(observed vs HH theory)")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
        ax.set_ylim(-0.02, 0.7)
        clean_axes(ax)

    fig.savefig(output_dir / "fig_md_pop_rmsd.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir / 'fig_md_pop_rmsd.png'}")


def _get_K_base(k: int) -> str:
    """Map K value to base directory name for GLU uncorrected."""
    if k == 5:
        return "glu_gshift_off"
    return f"glu_K{k}_off"


def plot_pop_K_independence(runs_dir: Path, output_dir: Path):
    """Fig: pKa error vs K for GLU (uncorrected).

    Bar plot of final ΔpKa at K=3,5,7. Uses HH-fit pKa when HH CSVs are
    available; falls back to populations.dat-based ΔpKa otherwise.
    Right panel shows convergence trajectories.
    """
    pka_ref = 4.25  # GLU macroscopic pKa reference
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0), constrained_layout=True)

    # Left: bar chart of final ΔpKa
    ax = axes[0]
    K_values, means, sems, n_runs = [], [], [], []

    for k in [3, 5, 7]:
        base = _get_K_base(k)
        seed_dirs = find_all_runs(runs_dir, base)

        final_deltas = []
        for sd in seed_dirs:
            # Try HH-fit pKa first
            it, pka, ref = _hh_fit_pka_trajectory(sd, pka_ref)
            if len(pka) >= 10:
                final_deltas.append(np.mean(pka[-20:]) - ref)
                continue
            # Fallback: populations.dat-based ΔpKa
            data = extract_convergence(sd)
            if len(data["pka"]) >= 10:
                tail = data["pka"][-20:]
                tail = tail[~np.isnan(tail)]
                if len(tail) > 0:
                    final_deltas.append(np.mean(tail) - data["pka_ref"])

        if final_deltas:
            K_values.append(k)
            means.append(np.mean(final_deltas))
            n_runs.append(len(final_deltas))
            sems.append(np.std(final_deltas, ddof=1) / np.sqrt(len(final_deltas))
                        if len(final_deltas) > 1 else 0)

    if K_values:
        ax.bar(K_values, means, yerr=sems, capsize=4,
               color=(*C_OFF[:3], 0.7), edgecolor=C_OFF[:3],
               width=0.8, linewidth=0.8)
        # Theory line
        theory = -np.log10(2)
        ax.axhline(theory, color="k", ls="--", lw=1.0, alpha=0.6,
                   label=rf"theory: $-\log_{{10}}(N_D) = {theory:.3f}$")
        ax.axhline(0, color="k", ls="-", lw=0.4, alpha=0.3)
        ax.set_xlabel("Number of replicas ($K$)")
        ax.set_ylabel(r"$\Delta$pKa (last 20 iters)")
        ax.set_title("(a) GLU uncorrected: pKa error vs $K$",
                     fontsize=8, fontweight="bold")
        ax.set_xticks(K_values)
        ax.legend(fontsize=6, loc="lower left")
        # Annotate n per bar
        for kv, m, n in zip(K_values, means, n_runs):
            ax.text(kv, m + 0.02, f"$n$={n}", ha="center", va="bottom",
                    fontsize=6)
    else:
        ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes, ha="center")
    clean_axes(ax)

    # Right: convergence trajectories at each K
    ax = axes[1]
    K_list = [3, 5, 7]
    cmap = plt.cm.Oranges(np.linspace(0.4, 0.95, len(K_list)))

    for k, color in zip(K_list, cmap):
        base = _get_K_base(k)
        seed_dirs = find_all_runs(runs_dir, base)

        all_data = []
        for sd in seed_dirs:
            # Try HH-fit pKa trajectory
            it, pka, ref = _hh_fit_pka_trajectory(sd, pka_ref)
            if len(it) >= 5:
                delta = pka - ref
                sm = ewma(delta, span=12)
                all_data.append({"iters": it, "vals": sm})
                continue
            # Fallback: populations.dat-based ΔpKa
            data = extract_convergence(sd)
            if len(data["pka"]) >= 5:
                it2 = np.arange(1, len(data["pka"]) + 1)
                delta2 = data["pka"] - data["pka_ref"]
                sm2 = ewma(delta2, span=12)
                all_data.append({"iters": it2, "vals": sm2})

        if not all_data:
            continue

        n_s = len(all_data)
        iters_r, mean, sem = ragged_mean_sem(
            [d["vals"] for d in all_data],
            [d["iters"] for d in all_data])

        ax.plot(iters_r, mean, color=color, lw=2.5,
                label=f"$K\\!={k}$ ($n$={n_s})")
        valid = ~np.isnan(sem)
        if valid.any():
            ax.fill_between(iters_r[valid],
                            mean[valid] - sem[valid],
                            mean[valid] + sem[valid],
                            color=(*color[:3], 0.2), lw=0)

    ax.axhline(-np.log10(2), color="k", ls="--", lw=1.0, alpha=0.6)
    ax.axhline(0, color="k", ls="-", lw=0.4, alpha=0.3)
    ax.set_xlabel("ALF iteration")
    ax.set_ylabel(r"$\Delta$pKa")
    ax.set_title("(b) GLU uncorrected: $\\Delta$pKa trajectory",
                 fontsize=8, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
    ax.set_ylim(-1.5, 1.5)
    clean_axes(ax)

    fig.savefig(output_dir / "fig_md_pop_K_independence.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir / 'fig_md_pop_K_independence.png'}")


def plot_pop_W_dependence(runs_dir: Path, output_dir: Path):
    """Fig: ΔpKa vs W for LYS.

    Two panels: (a) uncorrected at W=1,3,5,10, (b) corrected at W=1,3,5,10.
    Uses HH-fit pKa when available, populations.dat-based ΔpKa as fallback.
    """
    pka_ref = 10.4  # LYS reference
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0), constrained_layout=True)

    W_list = [1, 3, 5, 10]
    off_cmap = plt.cm.Oranges(np.linspace(0.4, 0.95, len(W_list)))
    on_cmap = plt.cm.Blues(np.linspace(0.4, 0.95, len(W_list)))

    for ax, condition, cmap_colors, title in [
        (axes[0], "off", off_cmap, "(a) LYS uncorrected — varying $W$"),
        (axes[1], "on", on_cmap, "(b) LYS corrected — varying $W$"),
    ]:
        for w, color in zip(W_list, cmap_colors):
            base = f"lys_W{w}_{condition}"
            seed_dirs = find_all_runs(runs_dir, base)

            all_data = []
            for sd in seed_dirs:
                # Try HH-fit pKa first
                it, pka, ref = _hh_fit_pka_trajectory(sd, pka_ref)
                if len(it) >= 5:
                    delta = pka - ref
                    sm = ewma(delta, span=12)
                    all_data.append({"iters": it, "vals": sm})
                    continue
                # Fallback: populations.dat-based ΔpKa
                data = extract_convergence(sd)
                if len(data["pka"]) >= 5:
                    it2 = np.arange(1, len(data["pka"]) + 1)
                    delta2 = data["pka"] - data["pka_ref"]
                    sm2 = ewma(delta2, span=12)
                    all_data.append({"iters": it2, "vals": sm2})

            if not all_data:
                continue

            # Faint individual traces
            for d in all_data:
                ax.plot(d["iters"], d["vals"],
                        color=(*color[:3], 0.12), lw=0.4)

            n_s = len(all_data)
            iters_r, mean, sem = ragged_mean_sem(
                [d["vals"] for d in all_data],
                [d["iters"] for d in all_data])

            ax.plot(iters_r, mean, color=color, lw=2.5,
                    label=f"$W\\!={w}$ ($n$={n_s})")
            valid = ~np.isnan(sem)
            if valid.any():
                ax.fill_between(iters_r[valid],
                                mean[valid] - sem[valid],
                                mean[valid] + sem[valid],
                                color=(*color[:3], 0.2), lw=0)

        ax.axhline(0, color="k", ls="--", lw=0.6, alpha=0.4)
        ax.set_xlabel("ALF iteration")
        if ax == axes[0]:
            ax.set_ylabel(r"$\Delta$pKa (fitted $-$ reference)")
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
        ax.set_ylim(-2.5, 2.5)
        clean_axes(ax)

    fig.savefig(output_dir / "fig_md_pop_W_dependence.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_dir / 'fig_md_pop_W_dependence.png'}")


def print_ensemble_summary(runs_dir: Path):
    """Print ensemble statistics for conditions with seed dirs."""
    print("\n" + "=" * 80)
    print("ENSEMBLE SUMMARY")
    print("=" * 80)

    conditions = [
        "lys_gshift_on", "lys_gshift_off",
        "glu_gshift_on", "glu_gshift_off",
        "hsp_gshift_on", "hsp_gshift_off",
        "glu_K3_off", "glu_K7_off",
        "lys_W1_off", "lys_W3_off", "lys_W5_off", "lys_W10_off", "lys_W10_on",
    ]

    print(f"{'Condition':<22s} {'Seeds':>5s} {'MinIter':>8s} "
          f"{'ΔpKa(last20)':>14s} {'SEM':>8s}")
    print("-" * 65)

    for cond in conditions:
        ens = extract_ensemble(runs_dir, cond)
        if ens is None:
            n_dirs = len(find_seed_dirs(runs_dir, cond))
            if n_dirs > 0:
                print(f"{cond:<22s} {n_dirs:>5d}     (no data yet)")
            continue

        # Final delta from each seed
        final_deltas = []
        for seed_data in ens["individual"]:
            tail = seed_data["delta_raw"][-20:]
            final_deltas.append(np.nanmean(tail))

        mean_delta = np.mean(final_deltas)
        sem_delta = (np.std(final_deltas, ddof=1) / np.sqrt(len(final_deltas))
                     if len(final_deltas) > 1 else 0.0)
        min_iter = min(len(d["iters"]) for d in ens["individual"])

        print(f"{cond:<22s} {ens['n_seeds']:>5d} {min_iter:>8d} "
              f"{mean_delta:>+14.3f} {sem_delta:>8.3f}")

    print("=" * 80)


# ---------------------------------------------------------------------------
# Parallel pre-computation of HH trajectories
# ---------------------------------------------------------------------------

# PKA_REF lookup by system prefix
_SYS_PKA_REF = {"lys": 10.4, "glu": 4.25, "hsp": 6.45}


def _guess_pka_ref(dirname: str) -> float:
    """Guess pKa reference from directory name prefix."""
    for prefix, ref in _SYS_PKA_REF.items():
        if dirname.lower().startswith(prefix):
            return ref
    return 10.4  # default


def _precompute_hh_one(run_dir: Path) -> tuple[str, tuple, tuple]:
    """Worker: compute HH-fit pKa trajectory and pop RMSD for one run dir.

    Bypasses cache since each worker has its own memory space.
    """
    pka_ref = _guess_pka_ref(run_dir.name)

    # HH-fit pKa trajectory (duplicates core logic to avoid cache lookup)
    hh_dir = run_dir / "solvated" / "plots" / "hh_plots"
    hh_iters, hh_pkas = [], []
    detected_ref = pka_ref
    if hh_dir.exists():
        # Auto-detect macro-pKa reference from first available CSV
        for i in range(1, 300):
            f = hh_dir / f"data_run{i}.csv"
            if f.exists():
                ref = _extract_ref_pka_from_csv(f)
                if ref is not None:
                    detected_ref = ref
                break
        for i in range(1, 300):
            f = hh_dir / f"data_run{i}.csv"
            if not f.exists():
                continue
            pka = _fit_pka_from_hh(f, detected_ref)
            if pka is not None:
                hh_iters.append(i)
                hh_pkas.append(pka)
    hh_fit = (np.array(hh_iters), np.array(hh_pkas), detected_ref)

    # Pop RMSD trajectory
    rmsd_iters, rmsd_vals = [], []
    if hh_dir.exists():
        for i in range(1, 300):
            f = hh_dir / f"data_run{i}.csv"
            if not f.exists():
                continue
            rmsd = _read_hh_pop_rmsd(f)
            if rmsd is not None:
                rmsd_iters.append(i)
                rmsd_vals.append(rmsd)
    pop_rmsd = (np.array(rmsd_iters), np.array(rmsd_vals))

    return str(run_dir), hh_fit, pop_rmsd


def precompute_hh_caches(run_dirs: list[Path], n_workers: int = 8):
    """Pre-compute HH-fit and pop RMSD trajectories in parallel.

    Populates _HH_FIT_CACHE and _POP_RMSD_CACHE.
    """
    # Filter to dirs that have hh_plots
    hh_dirs = [d for d in run_dirs
               if (d / "solvated" / "plots" / "hh_plots").exists()]
    if not hh_dirs:
        print("  No HH plot data found.")
        return

    print(f"Pre-computing HH trajectories for {len(hh_dirs)} runs "
          f"({min(n_workers, len(hh_dirs))} workers) ...")

    with ProcessPoolExecutor(max_workers=min(n_workers, len(hh_dirs))) as pool:
        futures = {pool.submit(_precompute_hh_one, d): d for d in hh_dirs}
        for future in as_completed(futures):
            key, hh_fit, pop_rmsd = future.result()
            _HH_FIT_CACHE[key] = hh_fit
            _POP_RMSD_CACHE[key] = pop_rmsd

    print(f"  Cached {len(_HH_FIT_CACHE)} HH-fit + "
          f"{len(_POP_RMSD_CACHE)} pop-RMSD trajectories.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _extract_one(run_dir: Path) -> tuple[str, dict]:
    """Worker for parallel extraction of convergence data."""
    return run_dir.name, extract_convergence(run_dir)


def _list_run_dirs(runs_dir: Path) -> list[Path]:
    """List all valid run directories under runs_dir."""
    dirs = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name == "figures":
            continue
        if (d / "solvated").exists():
            dirs.append(d)
    return dirs


def main():
    parser = argparse.ArgumentParser(description="Analyze gshift validation runs")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"),
                        help="Directory containing run subdirectories")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for figures (default: runs/figures)")
    parser.add_argument("-j", "--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    args = parser.parse_args()

    apply_style()

    runs_dir = args.runs_dir
    output_dir = args.output_dir or runs_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = _list_run_dirs(runs_dir)
    n_workers = min(args.workers, len(run_dirs))

    # Extract convergence data in parallel
    print(f"Extracting convergence from {len(run_dirs)} runs "
          f"({n_workers} workers) ...")
    runs_data = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_extract_one, d): d for d in run_dirs}
        for future in as_completed(futures):
            name, data = future.result()
            runs_data[name] = data
            n = len(data["pka"])
            if n > 0:
                print(f"  {name}: {data['system']}, {n} iters, "
                      f"pKa_ref={data['pka_ref']:.2f}, "
                      f"last ΔpKa={data['pka'][-1] - data['pka_ref']:+.3f}")

    # Pre-compute HH-fit and pop RMSD trajectories in parallel
    precompute_hh_caches(run_dirs, n_workers)

    print_summary(runs_data)
    print_ensemble_summary(runs_dir)

    print("\nGenerating figures ...")
    plot_convergence(runs_data, runs_dir, output_dir)
    plot_K_independence(runs_data, runs_dir, output_dir)
    plot_W_dependence(runs_data, runs_dir, output_dir)
    plot_profile_convergence(runs_dir, output_dir)
    plot_bsum_convergence(runs_dir, output_dir)
    plot_profile_rmsd(runs_dir, output_dir)
    plot_profile_rmsd_to_ref(runs_dir, output_dir)
    plot_pop_rmsd(runs_dir, output_dir)
    plot_pop_K_independence(runs_dir, output_dir)
    plot_pop_W_dependence(runs_dir, output_dir)
    plot_hh_convergence(runs_dir, output_dir)

    # Also copy figures to paper repo if it exists and is a different directory
    paper_figs = Path("/home/stanislc/projects/gshift-correction-paper/paper/figures")
    if paper_figs.exists() and paper_figs.resolve() != output_dir.resolve():
        import shutil
        for png in output_dir.glob("fig_md_*.png"):
            dest = paper_figs / png.name
            shutil.copy2(png, dest)
            print(f"  Copied → {dest}")

    print("\nDone!")


if __name__ == "__main__":
    main()
