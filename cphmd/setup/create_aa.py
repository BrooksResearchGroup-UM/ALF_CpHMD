"""
Create amino acid and nucleic acid template structures.

This module provides functions to generate PDB, CRD, and PSF files
for single residue building blocks using CHARMM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from cphmd import TOPPAR_DIR
from cphmd.native import system
from cphmd.native.types import AtomSelection

# Default titratable amino acids for CpHMD
TITRATABLE_AMINO_ACIDS = ["HSP", "LYS", "ARG", "ASP", "GLU", "TYR", "SER", "CYS"]

# Standard nucleic acids
NUCLEIC_ACIDS = ["ADE", "THY", "GUA", "CYT", "URA"]


def _load_topology(toppar_dir: Path | None = None) -> None:
    """Load CHARMM topology and parameter files.

    Args:
        toppar_dir: Path to topology directory. Defaults to project toppar/.
    """
    if toppar_dir is None:
        toppar_dir = TOPPAR_DIR

    toppar_dir = Path(toppar_dir)

    # Suppress warnings during topology loading
    system.set_bomb_level(-1)

    system.read_rtf(toppar_dir / "top_all36_prot.rtf")
    system.read_rtf(toppar_dir / "top_all36_na.rtf", append=True)
    system.read_param(toppar_dir / "par_all36m_prot.prm")
    system.read_param(toppar_dir / "par_all36_na.prm", append=True)
    system.stream_file(toppar_dir / "toppar_water_ions.str")

    # Restore bomb level
    system.set_bomb_level(0)


def create_amino_acid(
    residue: str,
    output_dir: Path | str = "pdb",
    toppar_dir: Path | None = None,
    template: str = "ALA ALA {res} ALA ALA",
    overwrite: bool = False,
) -> Path | None:
    """Create a single amino acid template structure.

    Generates PDB, CRD, and PSF files for a single amino acid residue
    capped with ALA residues and acetyl/CT3 termini.

    Args:
        residue: Three-letter amino acid code (e.g., "ASP", "GLU").
        output_dir: Directory to write output files.
        toppar_dir: Path to topology files. Defaults to project toppar/.
        template: Sequence template with {res} placeholder.
        overwrite: If True, overwrite existing files.

    Returns:
        Path to the generated PDB file, or None if skipped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = output_dir / f"{residue.lower()}.pdb"

    if pdb_path.exists() and not overwrite:
        print(f"File {pdb_path} already exists, skipping")
        return None

    # Load topology if not already loaded
    _load_topology(toppar_dir)

    # Generate sequence
    seq = template.format(res=residue)
    system.read_sequence_string(seq)

    # Build structure
    system.generate_segment("PROA", first="ACE", last="CT3", setup=True)
    system.ic_prm_fill(comp=False)
    system.ic_seed(((1, "CAY"), (1, "CY"), (1, "N")))
    system.ic_build()
    system.coor_orient()

    # Write output files
    system.write_coor(output_dir / f"{residue.lower()}.crd")
    system.write_coor_pdb(pdb_path)
    system.write_psf(output_dir / f"{residue.lower()}.psf")

    # Clean up for next run
    system.delete_atoms(AtomSelection())

    print(f"Created {pdb_path}")
    return pdb_path


def create_nucleic_acid(
    residue: str,
    output_dir: Path | str = "pdb",
    toppar_dir: Path | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Create a single nucleic acid template structure.

    Generates PDB, CRD, and PSF files for a single nucleic acid residue
    with 5TER/3TER termini.

    Args:
        residue: Three-letter nucleic acid code (e.g., "ADE", "GUA").
        output_dir: Directory to write output files.
        toppar_dir: Path to topology files. Defaults to project toppar/.
        overwrite: If True, overwrite existing files.

    Returns:
        Path to the generated PDB file, or None if skipped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = output_dir / f"{residue.lower()}.pdb"

    if pdb_path.exists() and not overwrite:
        print(f"File {pdb_path} already exists, skipping")
        return None

    # Load topology if not already loaded
    _load_topology(toppar_dir)

    # Generate sequence (single nucleotide)
    system.read_sequence_string(residue)

    # Build structure
    system.generate_segment("PROA", first="5TER", last="3TER", setup=True)
    system.ic_prm_fill(comp=False)
    system.ic_seed(((1, "C1'"), (1, "C2'"), (1, "C3'")))
    system.ic_build()
    system.coor_orient()

    # Write output files
    system.write_coor(output_dir / f"{residue.lower()}.crd")
    system.write_coor_pdb(pdb_path)
    system.write_psf(output_dir / f"{residue.lower()}.psf")

    # Clean up for next run
    system.delete_atoms(AtomSelection())

    print(f"Created {pdb_path}")
    return pdb_path


def create_all_templates(
    output_dir: Path | str = "pdb",
    toppar_dir: Path | None = None,
    molecule_type: Literal["amino", "nucleic", "both"] = "both",
    amino_acids: list[str] | None = None,
    nucleic_acids: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """Create template structures for multiple residues.

    Args:
        output_dir: Directory to write output files.
        toppar_dir: Path to topology files.
        molecule_type: Which types to create ("amino", "nucleic", or "both").
        amino_acids: List of amino acids to create. Defaults to titratable residues.
        nucleic_acids: List of nucleic acids to create. Defaults to standard bases.
        overwrite: If True, overwrite existing files.

    Returns:
        Dictionary with "amino" and "nucleic" keys containing lists of created paths.
    """
    results: dict[str, list[Path]] = {"amino": [], "nucleic": []}

    if amino_acids is None:
        amino_acids = TITRATABLE_AMINO_ACIDS
    if nucleic_acids is None:
        nucleic_acids = NUCLEIC_ACIDS

    if molecule_type in ("amino", "both"):
        print(f"Creating amino acid templates: {amino_acids}")
        for aa in amino_acids:
            path = create_amino_acid(aa, output_dir, toppar_dir, overwrite=overwrite)
            if path:
                results["amino"].append(path)

    if molecule_type in ("nucleic", "both"):
        print(f"Creating nucleic acid templates: {nucleic_acids}")
        for na in nucleic_acids:
            path = create_nucleic_acid(na, output_dir, toppar_dir, overwrite=overwrite)
            if path:
                results["nucleic"].append(path)

    return results
