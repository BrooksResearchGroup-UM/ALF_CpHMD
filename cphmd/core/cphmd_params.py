"""
CpHMD parameter calculations for constant pH molecular dynamics.

This module implements the mathematical framework for computing pH-dependent
bias parameters from micro-pKa values specified in the TAG field of patches.

Henderson-Hasselbalch Framework:
- UPOS (s=+1): Population = P / (1 + 10^(pH - pKa)) - decreases with pH
- UNEG (s=-1): Population = P / (1 + 10^(pKa - pH)) - increases with pH
- NONE (s=0): Reference state with no pH dependence

The pH₀ (isoelectric point) is computed from micro-pKa values:
- For sites with both UPOS and UNEG: pH₀ = 0.5 * (pKa_pos + pKa_neg)
- For acidic sites (UNEG only): pH₀ = pKa_neg
- For basic sites (UPOS only): pH₀ = pKa_pos
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


# Physical constants
KB = 0.0019872041  # Boltzmann constant in kcal·mol⁻¹·K⁻¹


@dataclass
class SiteParameters:
    """CpHMD parameters for a single titratable site.

    Attributes:
        site_id: Site identifier (e.g., "1", "2")
        pH0: Isoelectric point for this site
        pKa_pos: Macroscopic pKa for protonation (+↔0)
        pKa_neg: Macroscopic pKa for deprotonation (0↔−)
        subsite_shifts: Dictionary mapping SELECT names to pKa shifts
        site_type: Classification ("three_state", "acidic", "basic")
    """
    site_id: str
    pH0: float
    pKa_pos: float | None = None
    pKa_neg: float | None = None
    subsite_shifts: dict[str, float] = field(default_factory=dict)
    site_type: Literal["three_state", "acidic", "basic"] = "acidic"


@dataclass
class CpHMDParameters:
    """Complete CpHMD parameters for a simulation.

    Attributes:
        temperature: Temperature in Kelvin
        target_pH: User-specified target pH
        kTln10: Thermal energy factor (kB * T * ln(10))
        sites: Dictionary mapping site IDs to SiteParameters
        effective_pH: Computed effective pH for PHMD command
    """
    temperature: float
    target_pH: float
    kTln10: float = field(init=False)
    sites: dict[str, SiteParameters] = field(default_factory=dict)
    effective_pH: float = field(init=False)

    def __post_init__(self):
        """Compute derived values."""
        self.kTln10 = KB * self.temperature * np.log(10.0)
        self.effective_pH = self.target_pH


def parse_tag_value(tag: str) -> tuple[str, float | None]:
    """Parse a TAG field from patches.dat.

    Args:
        tag: TAG string like "UPOS 6.0", "UNEG 4.0", or "NONE"

    Returns:
        Tuple of (tag_type, pKa_value) where pKa_value is None for NONE tags
    """
    tag = str(tag).strip()
    parts = tag.split()

    if parts[0].upper() == "NONE":
        return ("NONE", None)
    elif parts[0].upper() == "UPOS":
        return ("UPOS", float(parts[1]))
    elif parts[0].upper() == "UNEG":
        return ("UNEG", float(parts[1]))
    else:
        # Unknown tag type - treat as NONE
        return ("NONE", None)


def compute_site_parameters(
    site_id: str,
    site_patches: pd.DataFrame,
) -> SiteParameters:
    """Compute CpHMD parameters for a single titratable site.

    This function analyzes the TAG values for all subsites and computes:
    - The isoelectric point (pH₀)
    - Macroscopic pKa values
    - Per-subsite pKa shifts

    Args:
        site_id: Site identifier
        site_patches: DataFrame rows for this site from patches.dat

    Returns:
        SiteParameters with computed values
    """
    # Categorize micro-pKa values by tag type
    pKa_upos: list[float] = []
    pKa_uneg: list[float] = []

    for _, row in site_patches.iterrows():
        tag_type, pKa = parse_tag_value(row["TAG"])
        if tag_type == "UPOS" and pKa is not None:
            pKa_upos.append(pKa)
        elif tag_type == "UNEG" and pKa is not None:
            pKa_uneg.append(pKa)

    # Compute macroscopic pKa values and pH₀
    params = SiteParameters(site_id=site_id, pH0=7.0)

    if pKa_upos and pKa_uneg:
        # Three-state system (e.g., histidine)
        # pKa_pos = -log10(Σ 10^(-pKa_i)) for UPOS states
        # pKa_neg = log10(Σ 10^(pKa_i)) for UNEG states
        params.pKa_pos = -np.log10(np.sum(10 ** (-np.array(pKa_upos))))
        params.pKa_neg = np.log10(np.sum(10 ** (np.array(pKa_uneg))))
        params.pH0 = 0.5 * (params.pKa_pos + params.pKa_neg)
        params.site_type = "three_state"

    elif pKa_uneg:
        # Acidic site (e.g., Asp, Glu)
        params.pKa_neg = np.log10(np.sum(10 ** (np.array(pKa_uneg))))
        params.pH0 = params.pKa_neg
        params.site_type = "acidic"

    elif pKa_upos:
        # Basic site (e.g., Lys, Arg)
        params.pKa_pos = -np.log10(np.sum(10 ** (-np.array(pKa_upos))))
        params.pH0 = params.pKa_pos
        params.site_type = "basic"

    # Compute per-subsite pKa shifts (relative to site pH₀)
    for _, row in site_patches.iterrows():
        tag_type, pKa = parse_tag_value(row["TAG"])
        select_name = row["SELECT"]

        if pKa is not None:
            params.subsite_shifts[select_name] = params.pH0 - pKa
        else:
            params.subsite_shifts[select_name] = 0.0

    return params


def compute_all_site_parameters(
    patch_info: pd.DataFrame,
    temperature: float,
    target_pH: float,
) -> CpHMDParameters:
    """Compute CpHMD parameters for all titratable sites.

    Args:
        patch_info: DataFrame from patches.dat with site/sub columns
        temperature: Simulation temperature in Kelvin
        target_pH: User-specified target pH

    Returns:
        CpHMDParameters containing all site parameters
    """
    cphmd = CpHMDParameters(temperature=temperature, target_pH=target_pH)

    # Process each site
    for site_id in patch_info["site"].unique():
        site_patches = patch_info[patch_info["site"] == site_id]
        site_params = compute_site_parameters(site_id, site_patches)
        cphmd.sites[site_id] = site_params

    # Determine effective pH for PHMD command
    pH0_values = [s.pH0 for s in cphmd.sites.values()]

    if len(set(pH0_values)) == 1:
        # Single unique pH₀ - use it as effective pH
        cphmd.effective_pH = pH0_values[0]
    elif len(set(pH0_values)) > 1:
        # Multiple different pH₀ values - use neutral reference
        cphmd.effective_pH = 0.0
    # else: keep target_pH as effective_pH

    return cphmd


def compute_bias_shifts(
    cphmd: CpHMDParameters,
    patch_info: pd.DataFrame,
    delta_pKa: float,
    replica_idx: int = 0,
) -> tuple[list[float], list[float]]:
    """Compute bias shift values for dynamics.

    Creates two bias arrays:
    1. b_shift: pH-dependent bias for dynamics (includes replica shifts)
    2. b_fix_shift: Original pKa shifts for ALF analysis

    Args:
        cphmd: CpHMD parameters
        patch_info: DataFrame from patches.dat
        delta_pKa: pH increment between replicas
        replica_idx: Index of current replica (0-based)

    Returns:
        Tuple of (b_shift, b_fix_shift) arrays
    """
    kTln10 = cphmd.kTln10
    ncentral = len(cphmd.sites) // 2

    # Compute replica-specific pH shift
    replica_shift = delta_pKa * (replica_idx - ncentral)
    total_delta_pH = (cphmd.target_pH - cphmd.effective_pH) + replica_shift

    b_shift: list[float] = []
    b_fix_shift: list[float] = []

    for _, row in patch_info.iterrows():
        tag_type, _ = parse_tag_value(row["TAG"])
        site_id = row["site"]
        select_name = row["SELECT"]

        # Determine sign based on tag type
        if tag_type == "UPOS":
            sign = +1
        elif tag_type == "UNEG":
            sign = -1
        else:
            sign = 0

        # b_shift: bias for dynamics (pH + replica effects)
        b_shift.append(sign * kTln10 * total_delta_pH)

        # b_fix_shift: original pKa shifts for analysis
        if site_id in cphmd.sites:
            pKa_shift = cphmd.sites[site_id].subsite_shifts.get(select_name, 0.0)
            b_fix_shift.append(sign * kTln10 * pKa_shift)
        else:
            b_fix_shift.append(0.0)

    return b_shift, b_fix_shift


def write_bias_files(
    output_dir: Path,
    b_shift: list[float],
    b_fix_shift: list[float],
) -> None:
    """Write bias shift files for dynamics.

    Args:
        output_dir: Directory to write files (creates nbshift/ subdirectory)
        b_shift: pH-dependent bias shifts
        b_fix_shift: Original pKa shifts for analysis
    """
    nbshift_dir = Path(output_dir) / "nbshift"
    nbshift_dir.mkdir(exist_ok=True)

    np.savetxt(
        nbshift_dir / "b_shift.dat",
        np.reshape(np.array(b_shift), (1, -1)),
        fmt="%.18e",
    )

    np.savetxt(
        nbshift_dir / "b_fix_shift.dat",
        np.reshape(np.array(b_fix_shift), (1, -1)),
        fmt="%.18e",
    )


def get_delta_pKa_for_phase(phase: int) -> float:
    """Get the pH increment between replicas for a given phase.

    Args:
        phase: Simulation phase (1, 2, or 3)

    Returns:
        delta_pKa value
    """
    if phase == 1:
        return 1.0
    elif phase == 2:
        return 0.5
    else:
        return 0.25
