"""Expected population targets for convergence metrics.

Bridges TAG/HH model with convergence criteria. When pH is active,
Henderson-Hasselbalch predicts non-uniform equilibrium populations;
all population-based checks should compare against these targets
rather than assuming uniform 1/N.

When pH is off (ALF mode), expected populations are uniform and all
existing behavior is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class ExpectedPopulations:
    """Per-state expected population fractions from HH model.

    Attributes:
        pH: The pH value used for the prediction.
        per_state: Flat array of expected fractions, shape (nblocks,).
            Sums to 1.0 within each site (per-site normalization).
        nsubs: Number of substates per site (e.g. [2, 3]).
        source: "theoretical" (from TAG pKa + pH) or "uniform" (1/N fallback).
    """

    pH: float
    per_state: np.ndarray
    nsubs: list[int]
    source: str


def compute_expected_populations(
    patch_info: "pd.DataFrame",
    pH: float,
    nsubs: list[int],
) -> ExpectedPopulations:
    """Compute HH-predicted per-state expected populations from TAG pKa values.

    Extracts micro-pKa values from the TAG column of patch_info and uses
    the Boltzmann partition function (via compute_theoretical_populations)
    to predict equilibrium populations at the given pH.

    When TAG pKa information is unavailable for a site, falls back to
    uniform 1/N for that site.

    Args:
        patch_info: DataFrame with columns including 'site' and 'TAG'.
            TAG format: "NONE", "UPOS <pKa>", or "UNEG <pKa>".
        pH: Target pH for HH prediction.
        nsubs: Number of substates per site.

    Returns:
        ExpectedPopulations with per-state fractions.
    """
    from cphmd.analysis.henderson_hasselbalch import (
        MicrostateType,
        compute_theoretical_populations,
    )

    nblocks = sum(nsubs)
    per_state = np.zeros(nblocks)
    has_theoretical = False

    if patch_info is None or "site" not in patch_info.columns:
        # No site info — uniform for all
        return _uniform_expected(pH, nsubs)

    offset = 0
    for site_idx, site_id in enumerate(patch_info["site"].unique()):
        site_patches = patch_info[patch_info["site"] == site_id]
        n = nsubs[site_idx] if site_idx < len(nsubs) else len(site_patches)

        # Build microstates list from TAG column
        microstates: list[tuple[str, MicrostateType, float | None]] = []
        for idx, (_, row) in enumerate(site_patches.iterrows()):
            select_name = str(row.get("SELECT", f"s{site_id}s{idx}"))
            tag = str(row.get("TAG", "NONE")).strip().upper()
            parts = tag.split()

            tag_type = parts[0] if parts else "NONE"
            tag_pKa = None
            if len(parts) >= 2:
                try:
                    tag_pKa = float(parts[1])
                except ValueError:
                    pass

            mtype: MicrostateType = "NONE"
            if tag_type == "UPOS":
                mtype = "UPOS"
            elif tag_type == "UNEG":
                mtype = "UNEG"

            microstates.append((select_name, mtype, tag_pKa))

        # Check if any titratable state has a pKa value
        has_pka = any(pka is not None for _, _, pka in microstates)

        if has_pka:
            # Compute theoretical populations at this pH
            pH_grid = np.array([pH])
            theo = compute_theoretical_populations(pH_grid, microstates)

            # Extract populations in order
            for idx, (select_name, _, _) in enumerate(microstates):
                if idx < n and select_name in theo:
                    per_state[offset + idx] = theo[select_name][0]
            has_theoretical = True
        else:
            # No pKa info for this site — uniform
            per_state[offset : offset + n] = 1.0 / n

        # Ensure per-site normalization
        site_total = per_state[offset : offset + n].sum()
        if site_total > 0:
            per_state[offset : offset + n] /= site_total

        offset += n

    return ExpectedPopulations(
        pH=pH,
        per_state=per_state,
        nsubs=list(nsubs),
        source="theoretical" if has_theoretical else "uniform",
    )


def _uniform_expected(pH: float, nsubs: list[int]) -> ExpectedPopulations:
    """Create uniform 1/N expected populations (ALF mode fallback)."""
    nblocks = sum(nsubs)
    per_state = np.zeros(nblocks)
    offset = 0
    for n in nsubs:
        per_state[offset : offset + n] = 1.0 / n
        offset += n
    return ExpectedPopulations(
        pH=pH,
        per_state=per_state,
        nsubs=list(nsubs),
        source="uniform",
    )


def compute_model_deviation(
    measured: np.ndarray,
    expected: ExpectedPopulations,
) -> float:
    """Compute worst-site max|measured_i - expected_i|.

    Replaces max(p)-min(p) as the population balance metric.
    When expected populations are uniform (1/N), this reduces to
    max(p) - 1/N, which is equivalent to the old metric for
    symmetric distributions.

    Args:
        measured: Per-state measured population fractions (nblocks,).
            Should be normalized per site.
        expected: Expected populations from compute_expected_populations.

    Returns:
        Worst-site maximum absolute deviation in [0, 1].
    """
    if measured.size == 0:
        return 1.0

    nsubs = expected.nsubs
    expected_arr = expected.per_state

    worst = 0.0
    offset = 0
    for n in nsubs:
        m = measured[offset : offset + n]
        e = expected_arr[offset : offset + n]
        site_dev = float(np.max(np.abs(m - e)))
        worst = max(worst, site_dev)
        offset += n

    return worst
