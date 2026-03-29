"""Parse CHARMM block.str LDIN definitions for titratable site information.

Extracts CALL and LDIN lines from block.str files to determine:
- Number of states per titratable site
- Slope direction (UPOS/UNEG) for Henderson-Hasselbalch fitting
- Model pKa values (micro and macro)
- Tautomer fractions for multi-state sites
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# CALL line: CALL <idx> SELEct segid <segid> .and. resid <resid> .and. resname <resname> end
_CALL_RE = re.compile(
    r"^\s*CALL\s+(\d+)\s+SELEct\s+.*segid\s+(\w+)\s+.*resid\s+(\d+)\s+.*resname\s+(\w+)\s+end",
    re.IGNORECASE,
)

# LDIN line: LDIN <idx> <lambda_init> <val> <val> <val> <val> <tag> [<pKa>]
_LDIN_RE = re.compile(
    r"^\s*LDIN\s+(\d+)\s+(\S+)\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)\s*(\S+)?",
    re.IGNORECASE,
)


@dataclass
class StateInfo:
    """Information for a single lambda state within a titratable site."""

    block_idx: int
    resname: str
    lambda_init: float
    tag: str
    model_pka: float | None


@dataclass
class SiteInfo:
    """Aggregated information for a titratable site parsed from block.str."""

    resid: str
    segid: str
    site_index: int
    n_states: int
    main_slope_sign: int
    pka_macro: float | None
    f_taut: float
    states: list[StateInfo] = field(default_factory=list)


def parse_block_str(block_path: Path) -> dict[str, SiteInfo]:
    """Parse a CHARMM block.str file to extract titratable site information.

    Parameters
    ----------
    block_path : Path
        Path to the block.str file.

    Returns
    -------
    dict[str, SiteInfo]
        Dictionary keyed by resid string, with SiteInfo for each titratable site.
        Duplicate resids are disambiguated with a suffix (e.g., "45", "45_site3").
    """
    block_path = Path(block_path)
    if not block_path.exists():
        logger.warning("block.str not found: %s", block_path)
        return {}

    lines = block_path.read_text().splitlines()

    # Parse CALL lines: idx -> (segid, resid, resname)
    call_data: dict[int, tuple[str, str, str]] = {}
    # Parse LDIN lines: idx -> (lambda_init, tag, model_pka)
    ldin_data: dict[int, tuple[float, str, float | None]] = {}

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("!") or not stripped:
            continue

        m = _CALL_RE.match(stripped)
        if m:
            idx = int(m.group(1))
            segid = m.group(2)
            resid = m.group(3)
            resname = m.group(4)
            call_data[idx] = (segid, resid, resname)
            continue

        m = _LDIN_RE.match(stripped)
        if m:
            idx = int(m.group(1))
            lambda_init = float(m.group(2))
            tag = m.group(3).upper()
            model_pka = float(m.group(4)) if m.group(4) and tag != "NONE" else None
            ldin_data[idx] = (lambda_init, tag, model_pka)

    # Skip LDIN index 1 (environment block).
    # Group by resid: consecutive CALL indices sharing the same resid form a site.
    sites_raw: dict[str, list[dict]] = {}
    segid_map: dict[str, str] = {}
    for idx in sorted(call_data.keys()):
        segid, resid, resname = call_data[idx]
        if idx not in ldin_data:
            continue
        lambda_init, tag, model_pka = ldin_data[idx]
        if resid not in sites_raw:
            sites_raw[resid] = []
            segid_map[resid] = segid
        sites_raw[resid].append(
            {
                "idx": idx,
                "resname": resname,
                "lambda_init": lambda_init,
                "tag": tag,
                "model_pka": model_pka,
            }
        )

    # Build SiteInfo dict keyed by resid, with duplicate disambiguation.
    result: dict[str, SiteInfo] = {}
    site_counter = 0
    resid_seen: dict[str, int] = {}

    for resid, raw_states in sites_raw.items():
        site_counter += 1
        n_states = len(raw_states)

        # Determine slope direction from first non-NONE tag.
        non_none = [s for s in raw_states if s["tag"] != "NONE"]
        if non_none:
            flag_type = non_none[0]["tag"]
            main_slope_sign = -1 if flag_type == "UNEG" else +1
        else:
            main_slope_sign = -1  # default for acids

        # Compute macro pKa and tautomer fraction.
        micro_pkas = [s["model_pka"] for s in non_none if s["model_pka"] is not None]
        if n_states >= 3 and len(micro_pkas) >= 2:
            kas = [10.0 ** (-pka) for pka in micro_pkas]
            ka_sum = sum(kas)
            pka_macro = -np.log10(ka_sum)
            f_taut = kas[0] / ka_sum
        elif len(micro_pkas) == 1:
            pka_macro = micro_pkas[0]
            f_taut = 0.5
        else:
            pka_macro = None
            f_taut = 0.5

        # Build StateInfo list.
        states = [
            StateInfo(
                block_idx=s["idx"],
                resname=s["resname"],
                lambda_init=s["lambda_init"],
                tag=s["tag"],
                model_pka=s["model_pka"],
            )
            for s in raw_states
        ]

        # Handle duplicate resids.
        if resid in resid_seen:
            resid_seen[resid] += 1
            key = f"{resid}_site{resid_seen[resid]}"
        else:
            resid_seen[resid] = 1
            key = resid

        result[key] = SiteInfo(
            resid=resid,
            segid=segid_map[resid],
            site_index=site_counter,
            n_states=n_states,
            main_slope_sign=main_slope_sign,
            pka_macro=pka_macro,
            f_taut=f_taut,
            states=states,
        )

    logger.info("LDIN parser: found %d sites in %s", len(result), block_path)
    for key, site in result.items():
        pka_str = f"{site.pka_macro:.2f}" if site.pka_macro is not None else "N/A"
        logger.info(
            "  resid %s: %d-state, slope=%d, macro_pKa=%s, f_taut=%.3f",
            key,
            site.n_states,
            site.main_slope_sign,
            pka_str,
            site.f_taut,
        )

    return result
