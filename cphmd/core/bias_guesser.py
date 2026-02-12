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


def _build_energy_block_command(
    patch_info: pd.DataFrame,
    site_lambdas: dict[int, np.ndarray],
    nsubs: list[int],
    fnex: float = 5.5,
) -> str:
    """Build a minimal BLOCK command for single-point energy evaluation.

    Sets LDIN with specified lambda values per site, zero biases, and FNEX
    constraint. No LDBV terms (we want raw physical energy, not biased).

    Args:
        patch_info: DataFrame from patches.dat.
        site_lambdas: {site_index: lambda array for that site's substates}
            Sites not in dict use equipartition (1/N).
        nsubs: Number of substates per site.
        fnex: FNEX constraint parameter.

    Returns:
        CHARMM BLOCK command string.
    """
    from .block_builder import (
        generate_block_header,
        generate_call_statements,
        generate_exclusions,
        generate_rmla_msld,
    )

    # Ensure site/sub columns exist (derived from SELECT, e.g. "s1s2" → site=1, sub=2)
    if "site" not in patch_info.columns:
        patch_info = patch_info.copy()
        patch_info[["site", "sub"]] = patch_info["SELECT"].str.extract(r"s(\d+)s(\d+)")
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
        f"LDIN {1:<4} {1:<4} {0.0:<4} {12.0:<4} {0.0:<2} {5.0:<4}",  # Environment
    ]
    for idx, row in patch_info.iterrows():
        l0 = full_lambda[idx]
        ldin_lines.append(f"LDIN {idx + 2:<4} {l0:<8.5f} {0.0:<4} {12.0:<4} {0.0:<2} {5.0:<4}")
    ldin_str = "\n".join(ldin_lines) + "\n\n"

    # Minimal dynamics setup (needed for BLOCK to work, but no actual dynamics)
    dynamics_setup = "QLDM THETA\nLANG TEMP 298.15\nSOFT ON\n\n"

    parts = [
        generate_block_header(n_blocks),
        generate_call_statements(patch_info),
        generate_exclusions(patch_info),
        dynamics_setup,
        ldin_str,
        generate_rmla_msld(patch_info, constraint_type="fnex", fnex=fnex),
        # NO LDBV — we want raw physical energy differences
        "END",
    ]
    return "\n".join(parts)


def _evaluate_energy() -> float:
    """Evaluate current CHARMM energy and return total."""
    import pycharmm.energy as energy

    energy.show()
    return energy.get_total()


def guess_initial_biases(
    patch_info: pd.DataFrame,
    nsubs: list[int],
    fnex: float = 5.5,
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
        patch_info: DataFrame from patches.dat.
        nsubs: Number of substates per site.
        fnex: FNEX constraint value.

    Returns:
        (b, c) where b has shape (1, nblocks) and c has shape (nblocks, nblocks).
    """
    from .charmm_utils import clear_block, execute_block_command

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
