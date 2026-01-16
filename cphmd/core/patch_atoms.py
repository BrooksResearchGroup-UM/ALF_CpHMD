"""Auto-detection of atoms from patch definitions.

This module provides functions to automatically determine which atoms
are affected by a protonation patch, eliminating the need for manual
atom specification in patches.dat.

For standard titratable residues (ASP, GLU, HIS, LYS, etc.), a built-in
lookup table is used. For custom ligands, the atoms can be parsed from
the patch definition in STR/RTF files.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Built-in lookup for standard titratable residues
# Maps patch name to list of atoms involved in the protonation state
STANDARD_PATCH_ATOMS: dict[str, list[str]] = {
    # Aspartate (ASP) - pKa ~4.0
    "ASPP": ["OD1", "OD2", "HD1", "HD2"],  # Protonated ASP
    "ASP": ["OD1", "OD2"],  # Deprotonated ASP (standard)

    # Glutamate (GLU) - pKa ~4.3
    "GLUP": ["OE1", "OE2", "HE1", "HE2"],  # Protonated GLU
    "GLU": ["OE1", "OE2"],  # Deprotonated GLU (standard)

    # Histidine (HIS) - pKa ~6.0
    "HSP": ["ND1", "NE2", "HD1", "HE2"],   # Doubly protonated (charged)
    "HSD": ["ND1", "NE2", "HD1"],          # Delta-protonated (neutral)
    "HSE": ["ND1", "NE2", "HE2"],          # Epsilon-protonated (neutral)
    "HIS": ["ND1", "NE2", "HD1", "HE2"],   # Generic histidine

    # Lysine (LYS) - pKa ~10.5
    "LYS": ["NZ", "HZ1", "HZ2", "HZ3"],    # Protonated (charged)
    "LSN": ["NZ", "HZ1", "HZ2"],           # Neutral lysine

    # Arginine (ARG) - pKa ~12.5
    "ARG": ["NE", "NH1", "NH2", "HE", "HH11", "HH12", "HH21", "HH22"],  # Charged
    "ARGN": ["NE", "NH1", "NH2", "HE", "HH11", "HH21"],  # Neutral

    # Cysteine (CYS) - pKa ~8.3
    "CYS": ["SG", "HG1"],                  # Protonated (thiol)
    "CYM": ["SG"],                         # Deprotonated (thiolate)

    # Tyrosine (TYR) - pKa ~10.5
    "TYR": ["OH", "HH"],                   # Protonated (phenol)
    "TYM": ["OH"],                         # Deprotonated (phenolate)

    # Serine (SER) - rarely titratable, pKa ~13
    "SER": ["OG", "HG1"],                  # Protonated (alcohol)
    "SEM": ["OG"],                         # Deprotonated (alkoxide)

    # N-terminus patches
    "NTER": ["N", "HT1", "HT2", "HT3"],    # Protonated N-terminus
    "ACE": ["CAY", "HY1", "HY2", "HY3", "CY", "OY"],  # Acetylated N-terminus

    # C-terminus patches
    "CTER": ["C", "OT1", "OT2"],           # Standard C-terminus (COO-)
    "CT2": ["C", "OT2", "HT2"],            # Protonated C-terminus (COOH)
}


# Alternative names for patches
PATCH_ALIASES: dict[str, str] = {
    "ASPH": "ASPP",
    "GLUH": "GLUP",
    "HISD": "HSD",
    "HISE": "HSE",
    "HISP": "HSP",
    "LYSH": "LYS",
    "ARGH": "ARG",
    "CYSH": "CYS",
    "TYRH": "TYR",
    "SERH": "SER",
}


@dataclass
class PatchAtomInfo:
    """Information about atoms in a patch.

    Attributes:
        atoms: List of atom names involved in the patch.
        added_atoms: Atoms added by the patch (from ATOM statements).
        deleted_atoms: Atoms deleted by the patch (from DELETE ATOM).
        modified_atoms: Atoms with modified properties.
    """

    atoms: list[str] = field(default_factory=list)
    added_atoms: list[str] = field(default_factory=list)
    deleted_atoms: list[str] = field(default_factory=list)
    modified_atoms: list[str] = field(default_factory=list)


def get_patch_atoms(
    patch_name: str,
    custom_patches: dict[str, list[str]] | None = None,
) -> list[str]:
    """Get atoms affected by a protonation patch.

    First checks custom_patches dict, then standard lookup table.
    Raises ValueError if patch is unknown.

    Args:
        patch_name: Name of the patch (e.g., "ASPP", "GLUP", "HSP").
        custom_patches: Optional dict mapping patch names to atom lists.

    Returns:
        List of atom names involved in the patch.

    Raises:
        ValueError: If patch name is unknown and not in custom_patches.

    Example:
        >>> get_patch_atoms("ASPP")
        ['OD1', 'OD2', 'HD1', 'HD2']
        >>> get_patch_atoms("MYLIG", custom_patches={"MYLIG": ["C1", "O1", "H1"]})
        ['C1', 'O1', 'H1']
    """
    # Normalize patch name (uppercase)
    patch_name = patch_name.upper()

    # Check for alias
    if patch_name in PATCH_ALIASES:
        patch_name = PATCH_ALIASES[patch_name]

    # Check custom patches first
    if custom_patches and patch_name in custom_patches:
        return custom_patches[patch_name]

    # Check standard lookup
    if patch_name in STANDARD_PATCH_ATOMS:
        return STANDARD_PATCH_ATOMS[patch_name]

    raise ValueError(
        f"Unknown patch: {patch_name}. "
        "Provide atoms explicitly via custom_patches parameter."
    )


def parse_patch_atoms_from_str(str_file: str | Path) -> dict[str, PatchAtomInfo]:
    """Parse atom names from patch definitions in STR file.

    Parses CHARMM stream files to extract atom information from
    PRES (patch residue) definitions.

    Args:
        str_file: Path to STR file containing patch definitions.

    Returns:
        Dictionary mapping patch name to PatchAtomInfo.

    Example:
        >>> patches = parse_patch_atoms_from_str("my_ligand.str")
        >>> patches["MYLIG"].atoms
        ['C1', 'O1', 'H1']
    """
    str_file = Path(str_file)
    patches: dict[str, PatchAtomInfo] = {}

    if not str_file.exists():
        raise FileNotFoundError(f"STR file not found: {str_file}")

    content = str_file.read_text()

    # Pattern to match PRES definitions
    # PRES PATCHNAME charge
    pres_pattern = re.compile(
        r"^\s*PRES\s+(\w+)\s+[\d.-]+",
        re.MULTILINE | re.IGNORECASE
    )

    # Find all PRES definitions
    pres_matches = list(pres_pattern.finditer(content))

    for i, match in enumerate(pres_matches):
        patch_name = match.group(1).upper()
        start_pos = match.end()

        # Find end of this patch (next PRES, RESI, or END)
        if i + 1 < len(pres_matches):
            end_pos = pres_matches[i + 1].start()
        else:
            # Find END or next major section
            end_match = re.search(
                r"^\s*(END|RESI|PRES)\s",
                content[start_pos:],
                re.MULTILINE | re.IGNORECASE
            )
            end_pos = start_pos + end_match.start() if end_match else len(content)

        patch_content = content[start_pos:end_pos]

        # Parse atoms in this patch
        info = _parse_patch_section(patch_content)
        info.atoms = list(set(
            info.added_atoms + info.deleted_atoms + info.modified_atoms
        ))

        patches[patch_name] = info
        logger.debug(f"Parsed patch {patch_name}: {len(info.atoms)} atoms")

    return patches


def _parse_patch_section(content: str) -> PatchAtomInfo:
    """Parse atom definitions from a patch section.

    Args:
        content: Text content of a single patch definition.

    Returns:
        PatchAtomInfo with parsed atoms.
    """
    info = PatchAtomInfo()

    # Pattern for ATOM statements: ATOM atomname type charge
    atom_pattern = re.compile(
        r"^\s*ATOM\s+(\w+)\s+\w+\s+[\d.-]+",
        re.MULTILINE | re.IGNORECASE
    )

    # Pattern for DELETE ATOM statements
    delete_pattern = re.compile(
        r"^\s*DELETE\s+ATOM\s+(\w+)",
        re.MULTILINE | re.IGNORECASE
    )

    # Find all ATOM statements (added atoms)
    for match in atom_pattern.finditer(content):
        atom_name = match.group(1).upper()
        info.added_atoms.append(atom_name)

    # Find all DELETE ATOM statements
    for match in delete_pattern.finditer(content):
        atom_name = match.group(1).upper()
        info.deleted_atoms.append(atom_name)

    # Modified atoms appear in both ATOM and bond/angle statements
    # without being in DELETE - we can detect them by looking at
    # atoms that already exist (lowercase 1, 2 suffixes typically)

    return info


def parse_patch_atoms_from_rtf(rtf_file: str | Path) -> dict[str, PatchAtomInfo]:
    """Parse atom names from patch definitions in RTF file.

    Similar to parse_patch_atoms_from_str but for pure RTF files.

    Args:
        rtf_file: Path to RTF file containing patch definitions.

    Returns:
        Dictionary mapping patch name to PatchAtomInfo.
    """
    # RTF files have the same format as the RTF section of STR files
    return parse_patch_atoms_from_str(rtf_file)


def load_custom_patches(
    files: list[str | Path],
) -> dict[str, list[str]]:
    """Load custom patch atoms from multiple topology files.

    Convenience function to parse patches from multiple STR/RTF files
    and return a combined dictionary.

    Args:
        files: List of STR or RTF file paths.

    Returns:
        Dictionary mapping patch name to atom list.

    Example:
        >>> custom = load_custom_patches(["ligand1.str", "ligand2.str"])
        >>> custom["LIG1_PROT"]
        ['O1', 'H1']
    """
    combined: dict[str, list[str]] = {}

    for file_path in files:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"File not found, skipping: {file_path}")
            continue

        try:
            patches = parse_patch_atoms_from_str(file_path)
            for name, info in patches.items():
                combined[name] = info.atoms
                logger.debug(f"Loaded patch {name} from {file_path}")
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")

    return combined


def get_atoms_for_residue_type(
    resname: str,
    protonation_state: str = "standard",
) -> list[str]:
    """Get titratable atoms for a standard residue type.

    Convenience function to get atoms based on residue name and
    desired protonation state.

    Args:
        resname: Residue name (ASP, GLU, HIS, LYS, ARG, CYS, TYR, SER).
        protonation_state: Either "standard", "protonated", or "deprotonated".

    Returns:
        List of atom names for the titratable group.

    Example:
        >>> get_atoms_for_residue_type("ASP")
        ['OD1', 'OD2']
        >>> get_atoms_for_residue_type("ASP", "protonated")
        ['OD1', 'OD2', 'HD1', 'HD2']
    """
    resname = resname.upper()

    # Mapping of residue to patches
    residue_patches = {
        "ASP": {"standard": "ASP", "protonated": "ASPP", "deprotonated": "ASP"},
        "GLU": {"standard": "GLU", "protonated": "GLUP", "deprotonated": "GLU"},
        "HIS": {"standard": "HSD", "protonated": "HSP", "deprotonated": "HSD"},
        "LYS": {"standard": "LYS", "protonated": "LYS", "deprotonated": "LSN"},
        "ARG": {"standard": "ARG", "protonated": "ARG", "deprotonated": "ARGN"},
        "CYS": {"standard": "CYS", "protonated": "CYS", "deprotonated": "CYM"},
        "TYR": {"standard": "TYR", "protonated": "TYR", "deprotonated": "TYM"},
        "SER": {"standard": "SER", "protonated": "SER", "deprotonated": "SEM"},
    }

    if resname not in residue_patches:
        raise ValueError(f"Unknown residue type: {resname}")

    state = protonation_state.lower()
    if state not in residue_patches[resname]:
        raise ValueError(f"Unknown protonation state: {protonation_state}")

    patch_name = residue_patches[resname][state]
    return get_patch_atoms(patch_name)
