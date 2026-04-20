"""Initial bias estimation from single-point energy perturbation.

Estimates ALF linear biases (b) and quadratic barriers (c) by evaluating
CHARMM energies at pure-state and midpoint lambda configurations on a
minimized structure. Used when presets are unavailable.
"""

from __future__ import annotations

import logging
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from cphmd.native import system
from cphmd.native.types import AtomSelection

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
        c array with shape (nblocks, nblocks), upper triangle populated and
        zero elsewhere. The quadratic c terms are consumed once per pair.
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
            rows.append(
                {
                    "SELECT": f"s{site_idx + 1}s{sub_idx + 1}",
                    "site": site_idx + 1,
                    "sub": sub_idx + 1,
                }
            )
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
            lines.append(f"CALL {block_idx} sele site{site_idx + 1}_sub{sub_idx + 1} end")
            block_idx += 1
    return "\n".join(lines) + "\n\n"


def _generate_named_call_statements(selection_names: list[str]) -> str:
    """Generate CALL statements from pre-defined CHARMM selection names."""
    lines = [
        "!----------------------------------------",
        "! Set up l-dynamics by setting BLOCK parameters",
        "!----------------------------------------\n",
    ]
    for idx, selection_name in enumerate(selection_names):
        lines.append(f"CALL {idx + 2} sele {selection_name} end")
    return "\n".join(lines) + "\n\n"


def _generate_block_header(n_blocks: int) -> str:
    """Generate a BLOCK command header."""
    return f"BLOCK {n_blocks}\n\n"


def _generate_call_statements(patch_info: pd.DataFrame) -> str:
    """Generate CALL statements for atom selections."""
    lines = [
        "!----------------------------------------",
        "! Set up l-dynamics by setting BLOCK parameters",
        "!----------------------------------------\n",
    ]

    for idx, row in patch_info.iterrows():
        lines.append(
            f"CALL {idx + 2} SELEct segid {row['SEGID']} .and. resid {row['RESID']} "
            f".and. resname {row['PATCH']} END"
        )

    return "\n".join(lines) + "\n\n"


def _generate_exclusions(patch_info: pd.DataFrame) -> str:
    """Generate ADEXCL statements for intra-site exclusions."""
    lines = [
        "!----------------------------------------",
        "! Exclude blocks from each other",
        "!----------------------------------------\n",
    ]

    for idx1, row1 in patch_info.iterrows():
        for idx2, row2 in patch_info.iterrows():
            if idx2 > idx1 and row1["site"] == row2["site"]:
                lines.append(f"adexcl {idx1 + 2:<3} {idx2 + 2:<3}")

    return "\n".join(lines) + "\n\n"


def _generate_rmla_msld(
    patch_info: pd.DataFrame,
    constraint_type: str = "fnex",
    fnex: float = 5.5,
    fpie_width: float = 1.0,
    fpie_force: float = 100.0,
    no_constraint: bool = False,
    electrostatics: str = "pmeex",
) -> str:
    """Generate RMLA and MSLD/MSMA statements."""
    lines = [
        "!------------------------------------------",
        "! Bond/angle/improper at full strength; dihedrals lambda-scaled",
        "! (Hayes & Brooks 2024: scale dihedrals, unscale bond+angle)",
        "!------------------------------------------\n",
        "rmla bond thet impr\n",
        "!------------------------------------------",
        "! MSLD - numbers assign each block to the specified site",
        "!------------------------------------------\n",
        "MSLD 0 -",
    ]

    for i, select in enumerate(patch_info["SELECT"]):
        site_num = select.lower().split("s")[1]
        if i < len(patch_info) - 1:
            lines.append(f"{site_num}  -")
        else:
            lines.append(f"{site_num} -")

    if no_constraint:
        constraint_str = ""
    elif constraint_type == "fpie":
        constraint_str = f"fpie widt {fpie_width} forc {fpie_force}"
    else:
        constraint_str = f"fnex {fnex}"

    lines.extend([
        f"{constraint_str} \n",
        "!------------------------------------------",
        "! Constructs the interaction matrix",
        "!------------------------------------------\n",
        "MSMA\n",
    ])

    if electrostatics in ("pmeex", "pme_ex"):
        lines.extend([
            "!------------------------------------------",
            "! PME exclusions for electrostatics",
            "!------------------------------------------\n",
            "PMEL EX\n",
        ])
    elif electrostatics in ("pmeon", "pme_on"):
        lines.extend([
            "!------------------------------------------",
            "! PME ON for electrostatics",
            "!------------------------------------------\n",
            "PMEL ON\n",
        ])
    elif electrostatics in ("pmenn", "pme_nn"):
        lines.extend([
            "!------------------------------------------",
            "! PME no-exclusions for electrostatics",
            "!------------------------------------------\n",
            "PMEL NN\n",
        ])

    return "\n".join(lines)


def _ensure_site_columns(patch_info: pd.DataFrame) -> pd.DataFrame:
    """Return patch_info with integer site/sub columns and contiguous index."""
    df = patch_info.copy()
    if "site" not in df.columns or "sub" not in df.columns:
        df[["site", "sub"]] = df["SELECT"].str.extract(r"(?i)s(\d+)s(\d+)")
    df["site"] = df["site"].astype(int)
    df["sub"] = df["sub"].astype(int)
    return df.reset_index(drop=True)


def _site_numbers(patch_info: pd.DataFrame) -> list[int]:
    """Return site numbers in deterministic ALF order."""
    return sorted(int(site) for site in patch_info["site"].unique())


def derive_site_keep_segids(
    patch_info: pd.DataFrame | None,
    nsubs: list[int],
    *,
    legacy: bool = False,
    legacy_keep_segid: str = "LIG",
) -> list[tuple[str, ...]]:
    """Derive the vacuum keep-segids for each titratable site."""
    if legacy or patch_info is None:
        keep = legacy_keep_segid.strip().upper()
        if not keep:
            raise ValueError("legacy_keep_segid must not be empty")
        return [(keep,) for _ in nsubs]

    df = _ensure_site_columns(patch_info)
    if "SEGID" not in df.columns:
        raise ValueError("patch_info must contain SEGID for segment-isolated vacuum")

    site_keep_segids: list[tuple[str, ...]] = []
    for site in _site_numbers(df):
        segids: list[str] = []
        for raw in df.loc[df["site"] == site, "SEGID"]:
            segid = str(raw).strip().upper()
            if segid and segid not in segids:
                segids.append(segid)
        if not segids:
            raise ValueError(f"No SEGID values found for site {site}")
        site_keep_segids.append(tuple(segids))

    if len(site_keep_segids) != len(nsubs):
        raise ValueError(
            "patch_info site count does not match nsubs: "
            f"{len(site_keep_segids)} != {len(nsubs)}"
        )
    return site_keep_segids


def parse_legacy_ligseg(setup_script: str | Path, default: str = "LIG") -> str:
    """Parse the legacy ligand segment variable from an msld-py-prep script."""
    script_path = Path(setup_script)
    match = re.search(
        r"(?im)^\s*set\s+ligseg\s*=\s*([^\s!]+)",
        script_path.read_text(),
    )
    segid = match.group(1) if match else default
    return segid.strip().upper()


def _delete_except_segids_selection(keep_segids: tuple[str, ...] | list[str]) -> str:
    """Build a CHARMM selection that keeps only the selected segids."""
    segids = [str(segid).strip().upper() for segid in keep_segids if str(segid).strip()]
    if not segids:
        raise ValueError("At least one segid must be kept for vacuum evaluation")
    clause = " .or. ".join(f"segid {segid}" for segid in segids)
    return f".not. ({clause})"


def _delete_except_segids_command(keep_segids: tuple[str, ...] | list[str]) -> str:
    """Build CHARMM command that keeps only the selected segids."""
    return f"delete atom sele {_delete_except_segids_selection(keep_segids)} end"


def _context_site_indices(
    target_site_idx: int,
    site_keep_segids: list[tuple[str, ...]],
) -> list[int]:
    """Sites retained in the same segment context as the target site."""
    target = set(site_keep_segids[target_site_idx])
    return [
        site_idx for site_idx, segids in enumerate(site_keep_segids) if target.intersection(segids)
    ]


def _local_default_context(
    patch_info: pd.DataFrame,
    context_site_indices: list[int],
) -> tuple[pd.DataFrame, list[int]]:
    """Build a contiguous local patch_info/nsubs view for default prep."""
    df = _ensure_site_columns(patch_info)
    site_numbers = _site_numbers(df)
    context_sites = [site_numbers[idx] for idx in context_site_indices]

    rows = []
    local_nsubs: list[int] = []
    for local_site, site in enumerate(context_sites, start=1):
        site_rows = df[df["site"] == site].sort_values("sub").copy()
        local_nsubs.append(len(site_rows))
        for local_sub, (_, row) in enumerate(site_rows.iterrows(), start=1):
            row = row.copy()
            row["site"] = local_site
            row["sub"] = local_sub
            row["SELECT"] = f"s{local_site}s{local_sub}"
            rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True), local_nsubs


def _local_legacy_context(
    nsubs: list[int],
    context_site_indices: list[int],
) -> tuple[pd.DataFrame, list[int], list[str]]:
    """Build local patch_info plus original legacy selection names."""
    rows = []
    call_selection_names: list[str] = []
    local_nsubs: list[int] = []
    for local_site, original_site_idx in enumerate(context_site_indices, start=1):
        n = nsubs[original_site_idx]
        local_nsubs.append(n)
        for local_sub in range(1, n + 1):
            rows.append(
                {
                    "SELECT": f"s{local_site}s{local_sub}",
                    "site": local_site,
                    "sub": local_sub,
                }
            )
            call_selection_names.append(f"site{original_site_idx + 1}_sub{local_sub}")
    return pd.DataFrame(rows), local_nsubs, call_selection_names


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
                        f"{constants.omega_decay:<8} {0.0:<6} {0:<1}"
                    )

    header = f"LDBI {idx}"
    return header + "\n" + "\n".join(ldbv_lines) + "\n\n"


def _build_energy_block_command(
    patch_info: pd.DataFrame | None,
    site_lambdas: dict[int, np.ndarray],
    nsubs: list[int],
    fnex: float = 5.5,
    legacy: bool = False,
    call_selection_names: list[str] | None = None,
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
    # For legacy mode, build synthetic patch_info from nsubs
    if legacy or patch_info is None:
        patch_info = _synthetic_patch_info(nsubs)

    # Ensure site/sub columns exist (derived from SELECT, e.g. "s1s2" → site=1, sub=2)
    patch_info = _ensure_site_columns(patch_info)

    if call_selection_names is not None and len(call_selection_names) != len(patch_info):
        raise ValueError("call_selection_names length must match patch_info rows")

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
    if call_selection_names is not None:
        call_str = _generate_named_call_statements(call_selection_names)
    elif legacy:
        call_str = _generate_legacy_call_statements(nsubs)
    else:
        call_str = _generate_call_statements(patch_info)

    parts = [
        _generate_block_header(n_blocks),
        call_str,
        _generate_exclusions(patch_info),
        dynamics_setup,
        ldin_str,
        _generate_rmla_msld(patch_info, no_constraint=True),
        _generate_zero_ldbv(patch_info, fnex=fnex),
        "END",
    ]
    return "\n".join(parts)


def _execute_block_command(block_cmd: str) -> None:
    """Execute generated BLOCK input through the native CHARMM stream boundary."""
    with tempfile.TemporaryDirectory(prefix="cphmd_bias_") as tmp:
        script_path = Path(tmp) / "block.inp"
        script_path.write_text(block_cmd.rstrip() + "\n")
        system.stream_file(script_path)


def _evaluate_energy() -> float:
    """Evaluate current CHARMM energy on CPU (no BLADE) and return total."""
    system.energy_show()
    return system.energy_get_total()


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
    # Ensure BLADE GPU is off — we evaluate energy on CPU only
    system.blade_off()

    configs = generate_lambda_configs(nsubs)
    endpoint_energies: dict[int, np.ndarray] = {}
    midpoint_energies: dict[int, dict[tuple[int, int], float]] = {}

    for site_idx, site_cfg in enumerate(configs):
        n = nsubs[site_idx]
        logger.info("Bias guess: evaluating site %d (%d substates)", site_idx, n)

        # --- Endpoint evaluations ---
        site_e = np.zeros(n)
        for sub_idx, lam in enumerate(site_cfg["endpoints"]):
            system.clear_block()
            block_cmd = _build_energy_block_command(
                patch_info,
                {site_idx: lam},
                nsubs,
                fnex=fnex,
                legacy=legacy,
            )
            _execute_block_command(block_cmd)
            site_e[sub_idx] = _evaluate_energy()
            logger.info("  endpoint[%d] E = %.3f", sub_idx, site_e[sub_idx])

        endpoint_energies[site_idx] = site_e

        # --- Midpoint evaluations ---
        site_mids: dict[tuple[int, int], float] = {}
        for (i, j), lam in site_cfg["midpoints"]:
            system.clear_block()
            block_cmd = _build_energy_block_command(
                patch_info,
                {site_idx: lam},
                nsubs,
                fnex=fnex,
                legacy=legacy,
            )
            _execute_block_command(block_cmd)
            e_mid = _evaluate_energy()
            site_mids[(i, j)] = e_mid
            logger.info("  midpoint[%d,%d] E = %.3f", i, j, e_mid)

        midpoint_energies[site_idx] = site_mids

    # Clear BLOCK after all evaluations
    system.clear_block()

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


def _evaluate_site_in_context(
    patch_info: pd.DataFrame,
    nsubs: list[int],
    local_site_idx: int,
    fnex: float = 5.5,
    call_selection_names: list[str] | None = None,
) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    """Evaluate one local site while keeping the rest of its context in BLOCK."""
    configs = generate_lambda_configs(nsubs)
    site_cfg = configs[local_site_idx]
    n = nsubs[local_site_idx]

    site_e = np.zeros(n)
    for sub_idx, lam in enumerate(site_cfg["endpoints"]):
        system.clear_block()
        block_cmd = _build_energy_block_command(
            patch_info,
            {local_site_idx: lam},
            nsubs,
            fnex=fnex,
            call_selection_names=call_selection_names,
        )
        _execute_block_command(block_cmd)
        site_e[sub_idx] = _evaluate_energy()

    site_mids: dict[tuple[int, int], float] = {}
    for (i, j), lam in site_cfg["midpoints"]:
        system.clear_block()
        block_cmd = _build_energy_block_command(
            patch_info,
            {local_site_idx: lam},
            nsubs,
            fnex=fnex,
            call_selection_names=call_selection_names,
        )
        _execute_block_command(block_cmd)
        site_mids[(i, j)] = _evaluate_energy()

    system.clear_block()
    return site_e, site_mids


def _setup_vacuum_nonbonded(
    patch_info: pd.DataFrame | None,
    nsubs: list[int],
    fnex: float = 5.5,
    legacy: bool = False,
    keep_segids: tuple[str, ...] | list[str] | None = None,
    call_selection_names: list[str] | None = None,
):
    """Delete non-kept segments and switch to vacuum nonbonded settings.

    Sets up a full BLOCK (CALL, LDIN, exclusions, MSLD) before NBONDS
    so CHARMM knows block assignments and excluded atom pairs during
    the nonbond list build (MAKINB).
    """
    # Lower bomb level through entire transition
    system.set_bomb_level(-6)

    if keep_segids:
        system.delete_atoms(AtomSelection(raw=_delete_except_segids_selection(keep_segids)))
    else:
        system.delete_atoms(AtomSelection(raw="resn TIP3 .or. segid SOLV .or. segid IONS"))

    # Clear crystal (removes PBC and PME)
    system.crystal_free()

    # Full BLOCK setup before NBONDS — CHARMM needs CALL assignments
    # and exclusions to allocate per-atom bond arrays correctly.
    system.clear_block()
    block_cmd = _build_energy_block_command(
        patch_info,
        {},
        nsubs,
        fnex=fnex,
        legacy=legacy,
        call_selection_names=call_selection_names,
    )
    _execute_block_command(block_cmd)

    # Set up vacuum nonbonded: force-shift, large cutoffs, no PME.
    system.nbonds_setup(
        cutnb=999.0,
        cutim=999.0,
        ctofnb=998.0,
        ctonnb=990.0,
        eps=1.0,
        atom=True,
        cdie=True,
        switch=False,
        vswitch=True,
        fshift=True,
        inbfrq=-1,
        imgfrq=-1,
        nbxmod=5,
    )

    system.set_bomb_level(0)


def guess_initial_biases_segment_vacuum(
    patch_info: pd.DataFrame | None,
    nsubs: list[int],
    *,
    reload_system: Callable[[], None],
    site_keep_segids: list[tuple[str, ...]] | None = None,
    legacy: bool = False,
    legacy_keep_segid: str = "LIG",
    fnex: float = 5.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate vacuum biases while keeping only the current site's segment context."""
    system.blade_off()

    if site_keep_segids is None:
        site_keep_segids = derive_site_keep_segids(
            patch_info,
            nsubs,
            legacy=legacy,
            legacy_keep_segid=legacy_keep_segid,
        )

    nblocks = sum(nsubs)
    b = np.zeros(nblocks)
    c = np.zeros((nblocks, nblocks))
    offsets = np.cumsum([0] + nsubs[:-1])

    for target_site_idx, keep_segids in enumerate(site_keep_segids):
        context_indices = _context_site_indices(target_site_idx, site_keep_segids)
        local_site_idx = context_indices.index(target_site_idx)

        if legacy:
            local_patch_info, local_nsubs, call_names = _local_legacy_context(
                nsubs, context_indices
            )
        else:
            if patch_info is None:
                raise ValueError("patch_info is required for default segment vacuum")
            local_patch_info, local_nsubs = _local_default_context(patch_info, context_indices)
            call_names = None

        reload_system()
        _setup_vacuum_nonbonded(
            local_patch_info,
            local_nsubs,
            fnex=fnex,
            keep_segids=keep_segids,
            call_selection_names=call_names,
        )

        system.set_bomb_level(-6)
        try:
            site_e, site_mids = _evaluate_site_in_context(
                local_patch_info,
                local_nsubs,
                local_site_idx,
                fnex=fnex,
                call_selection_names=call_names,
            )
        finally:
            system.set_bomb_level(0)
            system.clear_block()

        site_nsubs = [nsubs[target_site_idx]]
        b_site = compute_b_from_endpoints({0: site_e}, site_nsubs).reshape(-1)
        c_site = compute_c_from_midpoints({0: site_mids}, {0: site_e}, site_nsubs)

        offset = offsets[target_site_idx]
        n = nsubs[target_site_idx]
        b[offset : offset + n] = b_site
        c[offset : offset + n, offset : offset + n] = c_site

    return b.reshape(1, -1), c


def guess_initial_biases_vacuum(
    patch_info: pd.DataFrame | None,
    nsubs: list[int],
    fnex: float = 5.5,
    legacy: bool = False,
    keep_segids: tuple[str, ...] | list[str] | None = None,
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
    _setup_vacuum_nonbonded(
        patch_info,
        nsubs,
        fnex=fnex,
        legacy=legacy,
        keep_segids=keep_segids,
    )

    # Keep bomb level low during vacuum energy evaluations — the first
    # nonbond list rebuild after clearing PME may still trigger residual
    # colfft warnings from the invalidated FFT grid.
    system.set_bomb_level(-6)
    try:
        result = guess_initial_biases(patch_info, nsubs, fnex=fnex, legacy=legacy)
    finally:
        system.set_bomb_level(0)

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
    reload_system: Callable[[], None] | None = None,
    site_keep_segids: list[tuple[str, ...]] | None = None,
    legacy_keep_segid: str = "LIG",
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
    # Solvated evaluation first (non-destructive)
    b_solv, c_solv = guess_initial_biases(patch_info, nsubs, fnex=fnex, legacy=legacy)
    system.clear_block()

    # Vacuum evaluation (destructive — deletes atoms)
    if reload_system is None:
        b_vac, c_vac = guess_initial_biases_vacuum(
            patch_info,
            nsubs,
            fnex=fnex,
            legacy=legacy,
        )
    else:
        b_vac, c_vac = guess_initial_biases_segment_vacuum(
            patch_info,
            nsubs,
            reload_system=reload_system,
            site_keep_segids=site_keep_segids,
            legacy=legacy,
            legacy_keep_segid=legacy_keep_segid,
            fnex=fnex,
        )
    system.clear_block()

    # ΔΔE_solvation: isolates differential solvation contribution
    b = b_vac - b_solv
    c = C_MIDPOINT_SCALE * (c_vac - c_solv)

    logger.info(
        "Combined bias guess (ΔΔE_solv): b range [%.2f, %.2f], c range [%.2f, %.2f]",
        b.min(),
        b.max(),
        c.min(),
        c.max(),
    )

    return b, c
