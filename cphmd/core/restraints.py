"""
Restraint generators for titratable atoms in MSLD simulations.

In MSLD simulations, titratable atoms can exist in multiple chemical states
simultaneously. To prevent these "ghost" atoms from drifting apart during
dynamics, spatial restraints are applied.

Two restraint methods are supported:
1. SCAT (Soft-core constraint) - Harmonic restraint on atom positions
2. NOE (Nuclear Overhauser Effect) - Distance restraints between equivalent atoms

SCAT is generally preferred for its simplicity and efficiency.
NOE can be useful when explicit distance control is needed.
"""

import itertools
from pathlib import Path

import pandas as pd

from .generate_block import _wrap_cats_line


def generate_scat_restraints(
    patch_info: pd.DataFrame,
    include_hydrogen: bool = False,
    force_constant: float = 300.0,
) -> str:
    """Generate SCAT (soft-core constraint) restraints.

    SCAT applies harmonic position restraints to atoms within each
    titratable site, preventing ghost atoms from drifting.

    Args:
        patch_info: DataFrame from patches.dat with ATOMS column
        include_hydrogen: Whether to include hydrogen atoms in restraints
        force_constant: Restraint force constant (kcal/mol/Å²)

    Returns:
        CHARMM SCAT command string
    """
    lines = [
        "BLOCK",
        " scat on",
        f"scat k {force_constant}",
    ]

    for site in patch_info["site"].unique():
        site_data = patch_info[patch_info["site"] == site]

        # Collect all unique atoms for this site
        all_atoms = set()
        for atoms_str in site_data["ATOMS"]:
            all_atoms.update(atoms_str.split())

        # Separate heavy atoms and hydrogens
        heavy_atoms = [a for a in all_atoms if not a.startswith("H")]
        h_atoms = [a for a in all_atoms if a.startswith("H")]

        # Build inline selection from SEGID, RESID, and PATCH columns
        segid = site_data["SEGID"].iloc[0]
        resid = site_data["RESID"].iloc[0]
        resnames = site_data["PATCH"].tolist()
        resname_clause = " .or. ".join(f"resname {r}" for r in resnames)

        # Add restraints for heavy atoms
        for atom in heavy_atoms:
            lines.append(_wrap_cats_line(atom, segid, resid, resname_clause))

        # Optionally add hydrogen restraints
        if include_hydrogen:
            for atom in h_atoms:
                lines.append(_wrap_cats_line(atom, segid, resid, resname_clause))

    lines.append("END")
    return "\n".join(lines) + "\n"


def generate_noe_restraints(
    patch_info: pd.DataFrame,
    include_hydrogen: bool = False,
) -> str:
    """Generate NOE distance restraints for titratable atoms.

    NOE restraints maintain distances between equivalent atoms in
    different chemical states (e.g., same atom in protonated vs
    deprotonated forms).

    Args:
        patch_info: DataFrame from patches.dat with ATOMS column
        include_hydrogen: Whether to include hydrogen atoms

    Returns:
        CHARMM NOE command string
    """
    lines = ["NOE"]
    group_idx = 1

    for site, site_data in patch_info.groupby("site", sort=False):
        segid = site_data["SEGID"].iloc[0]
        resid = site_data["RESID"].iloc[0]
        first_patch = site_data["PATCH"].iloc[0]
        resname = first_patch[:3]  # First 3 characters

        # Get atoms and find those appearing in multiple patches
        atom_counts = site_data["ATOMS"].str.split().explode().value_counts()
        repeated_atoms = atom_counts[atom_counts > 1].index.tolist()

        if not repeated_atoms:
            continue

        # Filter atoms based on hydrogen setting
        if not include_hydrogen:
            repeated_atoms = [a for a in repeated_atoms if not a.startswith("H")]

        if not repeated_atoms:
            continue

        lines.extend([
            "!---------------------------------------------------------------",
            f"! Restraints for {segid} {resname} {resid}, SITE {site}, GROUP {group_idx}",
            "!---------------------------------------------------------------",
        ])
        group_idx += 1

        for atom in repeated_atoms:
            # Find patches containing this atom
            atom_patches = site_data[
                site_data["ATOMS"].str.contains(atom, na=False)
            ]

            if len(atom_patches) < 2:
                continue

            # Create pairwise restraints
            for i1, i2 in itertools.combinations(atom_patches.index, 2):
                patch1 = atom_patches.loc[i1, "PATCH"]
                patch2 = atom_patches.loc[i2, "PATCH"]

                lines.append(
                    f"assign sele segid {segid} .and. resid {resid} .and. resn {patch1} "
                    f".and. type {atom} end "
                    f"sele segid {segid} .and. resid {resid} .and. resn {patch2} "
                    f".and. type {atom} end -"
                )
                lines.append(
                    "kmin 100.0 rmin 0.0 kmax 100.0 rmax 0.0 fmax 2.0 rswitch 99999 sexp 1.0"
                )

    lines.append("END")
    return "\n".join(lines) + "\n"


def write_restraint_file(
    output_path: Path,
    patch_info: pd.DataFrame,
    method: str = "SCAT",
    include_hydrogen: bool = False,
) -> str:
    """Generate and write restraint file.

    Args:
        output_path: Path for output restraint file
        patch_info: DataFrame from patches.dat
        method: "SCAT" or "NOE"
        include_hydrogen: Whether to include hydrogen atoms

    Returns:
        The generated restraint command string
    """
    if method.upper() == "NOE":
        restraint_cmd = generate_noe_restraints(patch_info, include_hydrogen)
    else:
        restraint_cmd = generate_scat_restraints(patch_info, include_hydrogen)

    with open(output_path, "w") as f:
        f.write(restraint_cmd)

    return restraint_cmd
