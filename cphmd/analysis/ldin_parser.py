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

# Standard CALL line:
# CALL <idx> SELEct segid <segid> .and. resid <resid> .and. resname <resname> end
_CALL_RE = re.compile(
    r"^\s*CALL\s+(\d+)\s+SELEct\s+.*segid\s+(\w+)\s+.*resid\s+(\d+)\s+.*resname\s+(\w+)\s+end",
    re.IGNORECASE,
)

# Legacy msld-py-prep CALL line:
# CALL <idx> sele site<site>_sub<sub> show end
_LEGACY_CALL_RE = re.compile(
    r"^\s*CALL\s+(\d+)\s+SELE(?:ct)?\s+site(\d+)_sub(\d+)\b",
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

    # Parse CALL lines: idx -> metadata
    call_data: dict[int, dict[str, str | int]] = {}
    # Parse LDIN lines: idx -> (lambda_init, tag, model_pka)
    ldin_data: dict[int, tuple[float, str, float | None]] = {}

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("!") or not stripped:
            continue

        m = _CALL_RE.match(stripped)
        if m:
            idx = int(m.group(1))
            call_data[idx] = {
                "segid": m.group(2),
                "resid": m.group(3),
                "resname": m.group(4),
                "group": m.group(3),
            }
            continue

        m = _LEGACY_CALL_RE.match(stripped)
        if m:
            idx = int(m.group(1))
            site = int(m.group(2))
            sub = int(m.group(3))
            call_data[idx] = {
                "segid": "LIG",
                "resid": str(site),
                "resname": f"site{site}_sub{sub}",
                "group": f"site{site}",
                "legacy_site": site,
            }
            continue

        parsed_ldin = _parse_ldin_line(stripped)
        if parsed_ldin is not None:
            idx, lambda_init, tag, model_pka = parsed_ldin
            ldin_data[idx] = (lambda_init, tag, model_pka)

    # Skip LDIN index 1 (environment block).
    # Group by resid: consecutive CALL indices sharing the same resid form a site.
    sites_raw: dict[str, list[dict]] = {}
    site_meta: dict[str, dict[str, str | int]] = {}
    for idx in sorted(call_data.keys()):
        call = call_data[idx]
        if idx not in ldin_data:
            continue
        lambda_init, tag, model_pka = ldin_data[idx]
        group = str(call["group"])
        if group not in sites_raw:
            sites_raw[group] = []
            site_meta[group] = call
        sites_raw[group].append(
            {
                "idx": idx,
                "resname": str(call["resname"]),
                "lambda_init": lambda_init,
                "tag": tag,
                "model_pka": model_pka,
            }
        )

    # Build SiteInfo dict keyed by resid, with duplicate disambiguation.
    result: dict[str, SiteInfo] = {}
    site_counter = 0
    resid_seen: dict[str, int] = {}

    for group, raw_states in sites_raw.items():
        site_counter += 1
        meta = site_meta[group]
        resid = str(meta["resid"])
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
        if "legacy_site" in meta:
            key = f"site{meta['legacy_site']}"
        elif resid in resid_seen:
            resid_seen[resid] += 1
            key = f"{resid}_site{resid_seen[resid]}"
        else:
            resid_seen[resid] = 1
            key = resid

        result[key] = SiteInfo(
            resid=resid,
            segid=str(meta["segid"]),
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


def _parse_ldin_line(line: str) -> tuple[int, float, str, float | None] | None:
    """Parse standard LDIN lines and legacy ``!RX!`` annotations."""
    if not line.lower().startswith("ldin"):
        return None

    if "!RX!" in line:
        head, rx = line.split("!RX!", 1)
        head_tokens = head.split()
        if len(head_tokens) < 3:
            return None
        idx = int(head_tokens[1])
        lambda_init = _parse_float(head_tokens[2])
        rx_tokens = rx.split()
        tag = rx_tokens[0].upper() if rx_tokens else "NONE"
        model_pka = _parse_model_pka(tag, rx_tokens[1] if len(rx_tokens) > 1 else None)
        return idx, lambda_init, tag, model_pka

    tokens = line.split()
    if len(tokens) < 3:
        return None
    idx = int(tokens[1])
    lambda_init = _parse_float(tokens[2])
    tag = tokens[7].upper() if len(tokens) > 7 else "NONE"
    model_pka = _parse_model_pka(tag, tokens[8] if len(tokens) > 8 else None)
    return idx, lambda_init, tag, model_pka


def _parse_model_pka(tag: str, token: str | None) -> float | None:
    if tag == "NONE" or token is None:
        return None
    return _parse_float(token)


def _parse_float(token: str) -> float:
    try:
        return float(token)
    except ValueError:
        return float("nan")
