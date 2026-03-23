"""Initial bias estimation from single-point energy perturbation.

Estimates ALF linear biases (b) and quadratic barriers (c) by evaluating
CHARMM energies at pure-state and midpoint lambda configurations on a
minimized structure. Used when presets are unavailable.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_b_from_endpoints(
    endpoint_energies: dict[int, np.ndarray],
    nsubs: list[int],
) -> np.ndarray:
    """Compute linear biases (b) from endpoint energies.

    For each site, b[i] = -(E[i] - E[0]) where E[0] is the reference
    (first substate). This gives b[0] = 0 per site, matching the ALF
    convention where the first substate is the reference.

    Args:
        endpoint_energies: {site_index: array of energies per substate}
        nsubs: Number of substates per site.

    Returns:
        b array with shape (1, nblocks) matching ALF convention.
    """
    nblocks = sum(nsubs)
    b = np.zeros(nblocks)

    offset = 0
    for site_idx, n in enumerate(nsubs):
        energies = endpoint_energies[site_idx]
        b[offset : offset + n] = -(energies - energies[0])
        offset += n

    return b.reshape(1, -1)


def compute_c_from_midpoints(
    midpoint_energies: dict[int, dict[tuple[int, int], float]],
    endpoint_energies: dict[int, np.ndarray],
    nsubs: list[int],
) -> np.ndarray:
    """Compute quadratic barriers (c) from midpoint energies.

    For each pair (i,j) within a site:
        c[i][j] = -(E_mid[ij] - 0.5*(E[i] + E[j]))

    The excess energy above linear interpolation estimates the barrier height.

    Args:
        midpoint_energies: {site_index: {(sub_i, sub_j): midpoint_energy}}
        endpoint_energies: {site_index: array of energies per substate}
        nsubs: Number of substates per site.

    Returns:
        c array with shape (nblocks, nblocks), symmetric, zero diagonal.
    """
    nblocks = sum(nsubs)
    c = np.zeros((nblocks, nblocks))

    offset = 0
    for site_idx, n in enumerate(nsubs):
        e = endpoint_energies[site_idx]
        mids = midpoint_energies.get(site_idx, {})

        for (i, j), e_mid in mids.items():
            linear_interp = 0.5 * (e[i] + e[j])
            val = -(e_mid - linear_interp)
            c[offset + i, offset + j] = val
            c[offset + j, offset + i] = val

        offset += n

    return c


def generate_lambda_configs(
    nsubs: list[int],
) -> list[dict]:
    """Generate lambda configurations for endpoint and midpoint evaluations.

    For each site, generates:
    - N endpoint configs: lambda_i = 1, others = 0
    - N*(N-1)/2 midpoint configs: lambda_i = lambda_j = 0.5, others = 0

    Args:
        nsubs: Number of substates per site.

    Returns:
        List of dicts (one per site), each with:
        - "endpoints": list of lambda arrays (length N)
        - "midpoints": list of ((i, j), lambda_array) tuples
    """
    from itertools import combinations

    configs = []
    for n in nsubs:
        endpoints = []
        for i in range(n):
            lam = np.zeros(n)
            lam[i] = 1.0
            endpoints.append(lam)

        midpoints = []
        for i, j in combinations(range(n), 2):
            lam = np.zeros(n)
            lam[i] = 0.5
            lam[j] = 0.5
            midpoints.append(((i, j), lam))

        configs.append({"endpoints": endpoints, "midpoints": midpoints})
    return configs


def _synthetic_patch_info(nsubs: list[int]) -> pd.DataFrame:
    """Build a minimal patch_info DataFrame from nsubs alone.

    Used for legacy prep format where no patches.dat exists.
    The SELECT column uses the standard s{site}s{sub} naming,
    and site/sub columns are pre-populated.
    """
    rows = []
    for site_idx, n in enumerate(nsubs):
        for sub_idx in range(n):
            rows.append({
                "SELECT": f"s{site_idx + 1}s{sub_idx + 1}",
                "site": site_idx + 1,
                "sub": sub_idx + 1,
            })
    return pd.DataFrame(rows)


def _generate_legacy_call_statements(nsubs: list[int]) -> str:
    """Generate CALL statements using legacy siteX_subY named selections.

    Legacy (msld-py-prep) setup scripts define atom selections named
    ``site1_sub1``, ``site2_sub3``, etc. After ``setup_legacy()`` runs,
    these selections are in CHARMM memory and can be referenced directly.
    """
    lines = []
    block_idx = 2  # block 1 = environment
    for site_idx, n in enumerate(nsubs):
        for sub_idx in range(n):
            lines.append(
                f"CALL {block_idx} sele site{site_idx + 1}_sub{sub_idx + 1} end"
            )
            block_idx += 1
    return "\n".join(lines) + "\n\n"


def _generate_zero_ldbv(patch_info: pd.DataFrame, fnex: float = 5.5) -> str:
    """Generate LDBI/LDBV statements with zero coefficients.

    Required to allocate CHARMM's internal bias arrays (ibvidi, ibvidj, etc.)
    even when we want zero bias contribution.  Without these, the Fortran
    module-level allocatables in lambdadyn.F90 are never allocated, and
    ``msld_add_biasenergy`` segfaults on access.

    Args:
        patch_info: DataFrame from patches.dat (must have site/sub columns).
        fnex: FNEX parameter (used for type-8/10 REF values).

    Returns:
        CHARMM LDBI + LDBV string.
    """
    from .bias_constants import derive_bias_constants

    constants = derive_bias_constants(fnex)
    ldbv_lines: list[str] = []
    idx = 0

    # Type 6: quadratic barriers (c) — one per intra-site pair
    for site in patch_info["site"].unique():
        site_data = patch_info[patch_info["site"] == site]
        for i1, row1 in site_data.iterrows():
            for i2, row2 in site_data.iterrows():
                if i2 > i1:
                    idx += 1
                    ldbv_lines.append(
                        f"ldbv {idx:<3} {i1 + 2:<2} {i2 + 2:<2} {6:<4} {0.0:<8} {0.0:<6} {0:<1}"
                    )

    # Type 8: endpoint potentials (s) — one per ordered intra-site pair
    for site in patch_info["site"].unique():
        site_data = patch_info[patch_info["site"] == site]
        for i1, row1 in site_data.iterrows():
            for i2, row2 in site_data.iterrows():
                if i2 != i1:
                    idx += 1
                    ldbv_lines.append(
                        f"ldbv {idx:<3} {i1 + 2:<2} {i2 + 2:<2} {8:<4} "
                        f"{constants.chi_offset:<8.5f} {0.0:<6} {0:<1}"
                    )

    # Type 10: skew potentials (x) — one per ordered intra-site pair
    for site in patch_info["site"].unique():
        site_data = patch_info[patch_info["site"] == site]
        for i1, row1 in site_data.iterrows():
            for i2, row2 in site_data.iterrows():
                if i2 != i1:
                    idx += 1
                    ldbv_lines.append(
                        f"ldbv {idx:<3} {i1 + 2:<2} {i2 + 2:<2} {10:<4} "
                        f"{-constants.omega_decay:<8} {0.0:<6} {0:<1}"
                    )

    header = f"LDBI {idx}"
    return header + "\n" + "\n".join(ldbv_lines) + "\n\n"


def _build_energy_block_command(
    patch_info: pd.DataFrame | None,
    site_lambdas: dict[int, np.ndarray],
    nsubs: list[int],
    fnex: float = 5.5,
    legacy: bool = False,
) -> str:
    """Build a BLOCK command for single-point energy evaluation.

    Sets up MSLD with LDIN, RMLA MSLD FNEX, and zero-coefficient LDBV.
    The LDBV section is required to allocate CHARMM's internal bias arrays
    (prevents segfault in ``msld_add_biasenergy``).  All LDBV coefficients
    are zero so they contribute no bias energy.

    BLADE must be OFF before calling this (caller's responsibility).

    Args:
        patch_info: DataFrame from patches.dat, or None for legacy mode.
        site_lambdas: {site_index: lambda array for that site's substates}
            Sites not in dict use equipartition (1/N).
        nsubs: Number of substates per site.
        fnex: FNEX constraint parameter.
        legacy: If True, use siteX_subY named selections for CALL
            statements (legacy msld-py-prep format).

    Returns:
        CHARMM BLOCK command string.
    """
    from .block_builder import (
        generate_block_header,
        generate_call_statements,
        generate_exclusions,
        generate_rmla_msld,
    )

    # For legacy mode, build synthetic patch_info from nsubs
    if legacy or patch_info is None:
        patch_info = _synthetic_patch_info(nsubs)

    # Ensure site/sub columns exist (derived from SELECT, e.g. "s1s2" → site=1, sub=2)
    if "site" not in patch_info.columns:
        patch_info = patch_info.copy()
        patch_info[["site", "sub"]] = patch_info["SELECT"].str.extract(r"(?i)s(\d+)s(\d+)")
        patch_info["site"] = patch_info["site"].astype(int)
        patch_info["sub"] = patch_info["sub"].astype(int)

    n_blocks = len(patch_info) + 1  # +1 for environment

    # Build lambda vector for all blocks
    full_lambda = []
    for site_idx, n in enumerate(nsubs):
        if site_idx in site_lambdas:
            full_lambda.extend(site_lambdas[site_idx])
        else:
            # Equipartition for unperturbed sites
            full_lambda.extend([1.0 / n] * n)

    # Build LDIN lines with zero bias, specified lambda
    ldin_lines = [
        "!--- LDIN for energy evaluation ---",
        f"LDIN {1:<4} {1.0:.4f} {0.0:.4f} {12.0:.1f} {0.0:.4f} {5.0:.1f}",  # Environment
    ]
    for idx, row in patch_info.iterrows():
        l0 = full_lambda[idx]
        ldin_lines.append(f"LDIN {idx + 2:<4} {l0:.4f} {0.0:.4f} {12.0:.1f} {0.0:.4f} {5.0:.1f}")
    ldin_str = "\n".join(ldin_lines) + "\n\n"

    # MSLD setup (no actual dynamics — just for energy evaluation)
    dynamics_setup = "QLDM THETA\nLANG TEMP 298.15\nSOFT ON\n\n"

    # CALL statements: legacy uses siteX_subY selections, default uses patches.dat
    call_str = (
        _generate_legacy_call_statements(nsubs) if legacy
        else generate_call_statements(patch_info)
    )

    parts = [
        generate_block_header(n_blocks),
        call_str,
        generate_exclusions(patch_info),
        dynamics_setup,
        ldin_str,
        generate_rmla_msld(patch_info, no_constraint=True),
        _generate_zero_ldbv(patch_info, fnex=fnex),
        "END",
    ]
    return "\n".join(parts)


def _evaluate_energy() -> float:
    """Evaluate current CHARMM energy on CPU (no BLADE) and return total."""
    import pycharmm.energy as energy

    energy.show()
    return energy.get_total()


def guess_initial_biases(
    patch_info: pd.DataFrame | None,
    nsubs: list[int],
    fnex: float = 5.5,
    legacy: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate initial b and c biases from single-point energy evaluations.

    Requires pyCHARMM to be initialized with topology, PSF, coordinates,
    crystal, and non-bonded settings already configured (done by the caller
    in alf_runner before the first dynamics run).

    For each titratable site, evaluates:
    - N endpoint energies (lambda_i = 1, others = 0)
    - N*(N-1)/2 midpoint energies (lambda_i = lambda_j = 0.5)

    Other sites are held at equipartition during evaluation.

    Args:
        patch_info: DataFrame from patches.dat, or None for legacy mode.
        nsubs: Number of substates per site.
        fnex: FNEX constraint value.
        legacy: If True, use legacy siteX_subY named selections.

    Returns:
        (b, c) where b has shape (1, nblocks) and c has shape (nblocks, nblocks).
    """
    import pycharmm.lingo as lingo

    from .charmm_utils import clear_block, execute_block_command

    # Ensure BLADE GPU is off — we evaluate energy on CPU only
    lingo.charmm_script("blade off")

    configs = generate_lambda_configs(nsubs)
    endpoint_energies: dict[int, np.ndarray] = {}
    midpoint_energies: dict[int, dict[tuple[int, int], float]] = {}

    for site_idx, site_cfg in enumerate(configs):
        n = nsubs[site_idx]
        logger.info("Bias guess: evaluating site %d (%d substates)", site_idx, n)

        # --- Endpoint evaluations ---
        site_e = np.zeros(n)
        for sub_idx, lam in enumerate(site_cfg["endpoints"]):
            clear_block()
            block_cmd = _build_energy_block_command(
                patch_info,
                {site_idx: lam},
                nsubs,
                fnex=fnex,
                legacy=legacy,
            )
            execute_block_command(block_cmd)
            site_e[sub_idx] = _evaluate_energy()
            logger.info("  endpoint[%d] E = %.3f", sub_idx, site_e[sub_idx])

        endpoint_energies[site_idx] = site_e

        # --- Midpoint evaluations ---
        site_mids: dict[tuple[int, int], float] = {}
        for (i, j), lam in site_cfg["midpoints"]:
            clear_block()
            block_cmd = _build_energy_block_command(
                patch_info,
                {site_idx: lam},
                nsubs,
                fnex=fnex,
                legacy=legacy,
            )
            execute_block_command(block_cmd)
            e_mid = _evaluate_energy()
            site_mids[(i, j)] = e_mid
            logger.info("  midpoint[%d,%d] E = %.3f", i, j, e_mid)

        midpoint_energies[site_idx] = site_mids

    # Clear BLOCK after all evaluations
    clear_block()

    b = compute_b_from_endpoints(endpoint_energies, nsubs)
    c = compute_c_from_midpoints(midpoint_energies, endpoint_energies, nsubs)

    logger.info(
        "Bias guess complete: b range [%.2f, %.2f], c range [%.2f, %.2f]",
        b.min(),
        b.max(),
        c.min(),
        c.max(),
    )

    return b, c


def _setup_vacuum_nonbonded(
    patch_info: pd.DataFrame | None,
    nsubs: list[int],
    fnex: float = 5.5,
    legacy: bool = False,
):
    """Delete solvent/ions and switch to vacuum nonbonded settings.

    Sets up a full BLOCK (CALL, LDIN, exclusions, MSLD) before NBONDS
    so CHARMM knows block assignments and excluded atom pairs during
    the nonbond list build (MAKINB).
    """
    import pycharmm.lingo as lingo
    import pycharmm.settings as settings

    from .charmm_utils import clear_block, execute_block_command

    # Lower bomb level through entire transition
    settings.set_bomb_level(-6)

    lingo.charmm_script(
        "delete atom sele resn TIP3 .or. segid SOLV .or. segid IONS end"
    )

    # Clear crystal (removes PBC and PME)
    lingo.charmm_script("CRYSTAL FREE")

    # Full BLOCK setup before NBONDS — CHARMM needs CALL assignments
    # and exclusions to allocate per-atom bond arrays correctly.
    clear_block()
    block_cmd = _build_energy_block_command(
        patch_info, {}, nsubs, fnex=fnex, legacy=legacy,
    )
    execute_block_command(block_cmd)

    # Set up vacuum nonbonded: force-shift, large cutoffs, no PME.
    lingo.charmm_script(
        "NBONDS ELEC ATOM CDIE EPS 1 NOEWald "
        "CUTNB 999.0 CUTIM 999.0 CTOFNB 998.0 CTONNB 990.0 "
        "FSHIFT VSWITCH "
        "INBFRQ -1 IMGFRQ -1 NBXMOD 5"
    )

    settings.set_bomb_level(0)


def guess_initial_biases_vacuum(
    patch_info: pd.DataFrame | None,
    nsubs: list[int],
    fnex: float = 5.5,
    legacy: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate initial biases in vacuum (no solvent).

    Same as guess_initial_biases() but first removes solvent/ions and
    switches to vacuum nonbonded. This gives cleaner electrostatic
    differences between protonation states without solvent screening.

    WARNING: This permanently modifies the CHARMM session (atoms deleted).
    The caller must reload PSF/CRD afterward if further simulation is needed.

    Args:
        patch_info: DataFrame from patches.dat, or None for legacy mode.
        nsubs: Number of substates per site.
        fnex: FNEX constraint value.
        legacy: If True, use legacy siteX_subY named selections.

    Returns:
        (b, c) where b has shape (1, nblocks) and c has shape (nblocks, nblocks).
    """
    import pycharmm.settings as settings

    _setup_vacuum_nonbonded(patch_info, nsubs, fnex=fnex, legacy=legacy)

    # Keep bomb level low during vacuum energy evaluations — the first
    # nonbond list rebuild after clearing PME may still trigger residual
    # colfft warnings from the invalidated FFT grid.
    settings.set_bomb_level(-6)
    try:
        result = guess_initial_biases(patch_info, nsubs, fnex=fnex, legacy=legacy)
    finally:
        settings.set_bomb_level(0)

    return result


# Scale factor for c (barrier) biases: at the FNEX midpoint, λ_i ≈ λ_j ≈ 0.5,
# so the LDBV barrier V_c = COEF × λ_i × λ_j ≈ COEF/4. The single-point
# evaluation measures the barrier height directly, so COEF ≈ 4 × barrier.
C_MIDPOINT_SCALE = 4.0


def guess_initial_biases_combined(
    patch_info: pd.DataFrame | None,
    nsubs: list[int],
    fnex: float = 5.5,
    legacy: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate initial biases from solvated−vacuum energy difference.

    Computes ΔΔE_solvation = (vacuum bias) − (solvated bias) for both b and c.
    This isolates the differential solvation energy between protonation states,
    giving correct sign for ~7/9 residue types (b) and 9/9 (c).

    The c values are scaled by C_MIDPOINT_SCALE (~4) to account for the
    λ_i × λ_j ≈ 0.25 product at the FNEX transition midpoint.

    WARNING: Vacuum evaluation permanently modifies the CHARMM session
    (deletes solvent atoms). Must be called before dynamics.

    Args:
        patch_info: DataFrame from patches.dat, or None for legacy mode.
        nsubs: Number of substates per site.
        fnex: FNEX constraint value.
        legacy: If True, use legacy siteX_subY named selections.

    Returns:
        (b, c) where b has shape (1, nblocks) and c has shape (nblocks, nblocks).
    """
    from .charmm_utils import clear_block

    # Solvated evaluation first (non-destructive)
    b_solv, c_solv = guess_initial_biases(patch_info, nsubs, fnex=fnex, legacy=legacy)
    clear_block()

    # Vacuum evaluation (destructive — deletes solvent)
    b_vac, c_vac = guess_initial_biases_vacuum(patch_info, nsubs, fnex=fnex, legacy=legacy)
    clear_block()

    # ΔΔE_solvation: isolates differential solvation contribution
    b = b_vac - b_solv
    c = C_MIDPOINT_SCALE * (c_vac - c_solv)

    logger.info(
        "Combined bias guess (ΔΔE_solv): b range [%.2f, %.2f], c range [%.2f, %.2f]",
        b.min(), b.max(), c.min(), c.max(),
    )

    return b, c
