"""
CpHMD patching module.

This module applies titratable residue patches to create multiple protonation
states for constant pH molecular dynamics simulations.
"""

from __future__ import annotations

import itertools
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cphmd import TOPPAR_DIR
from cphmd.core.patch_applier import PatchApplier, ensure_patches_streamed
from cphmd.native import system
from cphmd.native.types import AtomSelection


@dataclass
class LigandPatchDef:
    """Definition of titratable patches for a custom ligand.

    Attributes:
        resname: Residue name of the ligand in the structure.
        patch_file: Path to RTF file containing PRES definitions for this ligand.
        patches: List of patch names to apply (e.g., ["p1_1", "p1_3", "p1_4"]).
        pka_values: Dict mapping patch name to pKa value (auto-detected from file if not provided).
        reference_patch: Name of the reference state patch (default: first in patches).

    Note:
        Atoms are automatically parsed from the patch file's PRES definitions.
        pKa values can also be auto-detected from "! pKa = X.X" comments in the patch file.

    Example:
        LigandPatchDef(
            resname="RIBO",
            patch_file="/path/to/master-patch.rtf",
            patches=["p1_1", "p1_3", "p1_4"],
        )
    """

    resname: str
    patch_file: str | Path
    patches: list[str] = field(default_factory=list)
    pka_values: dict[str, float] = field(default_factory=dict)
    reference_patch: str | None = None
    sites: list[LigandSiteDef] | None = None

    def __post_init__(self):
        if self.sites:
            self.sites = [LigandSiteDef(**s) if isinstance(s, dict) else s for s in self.sites]


@dataclass
class LigandSiteDef:
    """Definition of a single titratable site within a multi-site ligand.

    Attributes:
        patches: List of patch names for this site (e.g., ["FOLA_W", "FOLA_U"]).
        pka_values: Dict mapping patch name to pKa value for this site.
        reference_patch: Patch that becomes the NONE reference state. When set,
            the original state and tautomers get UNEG with this patch's pKa,
            matching the ASP/GLU convention (deprotonated = reference).
    """

    patches: list[str]
    pka_values: dict[str, float] = field(default_factory=dict)
    reference_patch: str | None = None


@dataclass
class PatchConfig:
    """Configuration for patching titratable residues.

    Attributes:
        input_folder: Input folder containing structure files (used with structure_file).
        structure_file: Base name of structure file (without extension).
        psf_file: Direct path to PSF file (overrides input_folder/structure_file).
        crd_file: Direct path to CRD file (overrides input_folder/structure_file).
        output_folder: Output folder for patched files (defaults to input_folder).
        hmr: Enable hydrogen mass repartitioning.
        hmr_waters: Apply HMR to water molecules too.
        selected_residues: List of residue selections (e.g., ["ASP", "PROA:15"]).
        toppar_dir: Path to topology directory.
        topology_files: List of topology files to load (relative to toppar_dir).
        extra_files: Additional topology files as absolute paths (for custom ligands).
        ligand_patches: List of LigandPatchDef for custom titratable ligands.

    Note:
        Either (input_folder + structure_file) OR (psf_file + crd_file) must be provided.
        If both are provided, psf_file/crd_file take precedence.

        For custom ligands, provide:
        - extra_files: List of RTF, PRM, or STR files with ligand topology/parameters
        - ligand_patches: LigandPatchDef objects defining titratable sites

    Example for ligand with custom patches:
        config = PatchConfig(
            input_folder="my_system/prep",
            extra_files=[
                "/path/to/ligand.rtf",
                "/path/to/ligand.prm",
            ],
            ligand_patches=[
                LigandPatchDef(
                    resname="RIBO",
                    patch_file="/path/to/master-patch.rtf",
                    patches=["p1_1", "p1_3", "p1_4"],
                    # atoms and pKa values are auto-detected from patch file
                ),
            ],
        )
    """

    input_folder: str | Path | None = None
    structure_file: str = "solvated"
    psf_file: str | Path | None = None
    crd_file: str | Path | None = None
    output_folder: str | Path | None = None
    hmr: bool = True
    hmr_waters: bool = False
    selected_residues: list[str] = field(default_factory=list)
    toppar_dir: Path | None = None
    topology_files: list[str] = field(
        default_factory=lambda: [
            "top_all36_prot.rtf",
            "par_all36m_prot.prm",
            "top_all36_na.rtf",
            "par_all36_na.prm",
            "toppar_water_ions.str",
            "top_all36_cgenff.rtf",
            "par_all36_cgenff.prm",
            "my_files/titratable_residues.str",
            "my_files/nucleic_c36.str",
        ]
    )
    extra_files: list[str | Path] = field(default_factory=list)
    ligand_patches: list[LigandPatchDef] = field(default_factory=list)

    def __post_init__(self):
        """Validate and resolve file paths."""
        # Check that we have valid input specification
        has_folder = self.input_folder is not None
        has_files = self.psf_file is not None and self.crd_file is not None
        if self.toppar_dir is not None:
            self.toppar_dir = Path(self.toppar_dir)

        if not has_folder and not has_files:
            raise ValueError("Must provide either input_folder or both psf_file and crd_file")

        # Resolve paths
        if has_files:
            self.psf_file = Path(self.psf_file)
            self.crd_file = Path(self.crd_file)
            if self.output_folder is None:
                self.output_folder = self.psf_file.parent
        else:
            self.input_folder = Path(self.input_folder)
            if self.output_folder is None:
                self.output_folder = self.input_folder

        self.output_folder = Path(self.output_folder)

        # Resolve extra_files to Path objects
        self.extra_files = [Path(f) for f in self.extra_files]

        # Validate ligand patch definitions
        for ligand_def in self.ligand_patches:
            patch_file = Path(ligand_def.patch_file)
            if not patch_file.exists():
                raise FileNotFoundError(f"Ligand patch file not found: {patch_file}")
            if not ligand_def.patches and not ligand_def.sites:
                raise ValueError(f"Ligand {ligand_def.resname}: must specify patches or sites")


class PatchParser:
    """Parse patch definitions from CHARMM topology files.

    This class reads titratable residue stream files and topology files
    to extract patch definitions, atom groups, and pKa values.

    Attributes:
        residues: List of titratable residue names.
        patches: Dict mapping residue name to list of patch names.
        atom_groups: Dict mapping patch/residue name to list of atom names.
        pka: Dict mapping patch name to pKa value.
    """

    def __init__(
        self,
        segment_path: str | Path = "toppar/my_files/titratable_residues.str",
        topology_path: str | Path = "toppar/top_all36_prot.rtf",
    ):
        self.segment_path = Path(segment_path)
        self.topology_path = Path(topology_path)
        self.atom_groups: dict[str, list[str]] = {}
        self.patches: dict[str, list[str]] = {}
        self.pka: dict[str, str] = {}
        self.residues: list[str] = []
        self.default_patches: list[str] = []
        # Maps resname -> reference_patch name (None means use default RESNAME+"O")
        self.reference_patches: dict[str, str | None] = {}
        # Maps resname -> list of site_keys for multi-site ligands
        self.site_keys: dict[str, list[str]] = {}

        self._load_patches()
        self._load_topology()
        self._load_default_patches()

        # Remove duplicates from atom groups
        for group in self.atom_groups:
            self.atom_groups[group] = list(set(self.atom_groups[group]))

    def _load_patches(self) -> None:
        """Load patch definitions from the segment stream file."""
        if not self.segment_path.exists():
            raise FileNotFoundError(f"{self.segment_path} not found")

        with open(self.segment_path, "r") as f:
            lines = [line.upper() for line in f.readlines()]

        for i, line in enumerate(lines):
            if line.startswith("!") and line.endswith("PATCHES\n"):
                resname_match = re.search(r"\((\w+)\)", line)
                if resname_match:
                    resname = resname_match.group(1).upper()
                    self.residues.append(resname)
                    self.patches.setdefault(resname, [])
                    self.atom_groups.setdefault(resname, [])

                    for j in range(i + 1, len(lines)):
                        if lines[j].startswith("!") and lines[j].endswith("PATCHES\n"):
                            break
                        if lines[j].startswith("PRES"):
                            patch_name = lines[j].split()[1].upper()
                            if patch_name not in self.patches[resname]:
                                self.patches[resname].append(patch_name)
                            self.atom_groups.setdefault(patch_name, [])

                            for k in range(j + 1, len(lines)):
                                if lines[k].startswith("PRES"):
                                    break
                                if "pka" in lines[k].lower():
                                    self.pka[patch_name] = lines[k].split("=")[1].strip()
                                if lines[k].startswith("ATOM"):
                                    atom = lines[k].split()[1].upper()
                                    self.atom_groups[patch_name].append(atom)
                                    if atom not in self.atom_groups[resname]:
                                        self.atom_groups[resname].append(atom)

    def _load_default_patches(self) -> None:
        """Load default patches from topology file."""
        if not self.topology_path.exists():
            raise FileNotFoundError(f"{self.topology_path} not found")

        with open(self.topology_path, "r") as f:
            lines = [line.upper() for line in f.readlines()]

        for i, line in enumerate(lines):
            if line.startswith("PRES"):
                patch_name = line.split()[1].upper()
                self.default_patches.append(patch_name)
                self.atom_groups.setdefault(patch_name, [])

                for k in range(i + 1, len(lines)):
                    if lines[k].startswith("PRES") or lines[k].startswith("RESI"):
                        break
                    if lines[k].startswith("ATOM"):
                        atom = lines[k].split()[1].upper()
                        self.atom_groups[patch_name].append(atom)

    def _load_topology(self) -> None:
        """Load atom definitions from topology file for residues."""
        if not self.topology_path.exists():
            return

        with open(self.topology_path, "r") as f:
            lines = [line.upper() for line in f.readlines()]

        for i, line in enumerate(lines):
            if line.startswith("RESI"):
                resname = line.split()[1].upper()
                if resname not in self.residues:
                    continue

                atoms = []
                bonds = []
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith("RESI") or lines[j].startswith("PRES"):
                        break
                    if lines[j].startswith("ATOM"):
                        atoms.append(lines[j].split()[1])
                    if lines[j].startswith("BOND"):
                        bonds.extend(lines[j].split()[1:])

                # Filter atoms to only those in topology
                self.atom_groups[resname] = [
                    atom for atom in self.atom_groups[resname] if atom in atoms
                ]

                # Add hydrogen atoms bonded to heavy atoms in the group
                for k in range(0, len(bonds), 2):
                    if k + 1 >= len(bonds):
                        break
                    if bonds[k] in self.atom_groups[resname] and not bonds[k].startswith("H"):
                        if bonds[k + 1] not in self.atom_groups[resname] and bonds[
                            k + 1
                        ].startswith("H"):
                            self.atom_groups[resname].append(bonds[k + 1])
                    if bonds[k + 1] in self.atom_groups[resname] and not bonds[k + 1].startswith(
                        "H"
                    ):
                        if bonds[k] not in self.atom_groups[resname] and bonds[k].startswith("H"):
                            self.atom_groups[resname].append(bonds[k])

    def print_atoms(self) -> None:
        """Print atom groups with pKa values."""
        for name in self.atom_groups:
            if name in self.pka:
                print(f"{name}, pKa {self.pka[name]}: {self.atom_groups[name]}\n")
            else:
                print(f"{name}: {self.atom_groups[name]}\n")

    def register_ligand(self, ligand_def: LigandPatchDef) -> None:
        """Register a custom ligand with its patch definitions.

        Atoms and pKa values are automatically parsed from the patch file.
        This allows adding titratable ligands with custom patch files
        to the patching workflow.

        Supports two modes:
        - Single-site: ligand_def.patches lists all patches for one MSLD site.
        - Multi-site: ligand_def.sites defines independent titratable sites,
          each registered under a site_key (e.g., "FOL_1", "FOL_2").

        Args:
            ligand_def: LigandPatchDef containing resname, patch_file, and patches/sites.
        """
        resname = ligand_def.resname.upper()
        patch_file = Path(ligand_def.patch_file)

        # Add to residues list (for structure search)
        if resname not in self.residues:
            self.residues.append(resname)

        if ligand_def.sites:
            self._register_multisite_ligand(resname, patch_file, ligand_def)
        else:
            self._register_singlesite_ligand(resname, patch_file, ligand_def)

    def _register_multisite_ligand(
        self, resname: str, patch_file: Path, ligand_def: LigandPatchDef
    ) -> None:
        """Register a multi-site ligand with independent titratable sites.

        Each site gets a site_key (e.g., "FOL_1", "FOL_2") used as the
        resname in titratable_list. Patches and atom_groups are registered
        per site_key so _apply_patches() can process each site independently.
        """
        site_key_list: list[str] = []

        for i, site in enumerate(ligand_def.sites):
            site_key = f"{resname}_{i + 1}"
            site_key_list.append(site_key)

            # Parse atoms from patch file for this site's patches
            all_patches = list(site.patches)
            patch_atoms, patch_pka, deleted_atoms = self._parse_ligand_patch_file(
                patch_file, all_patches
            )

            # Register patches under site_key
            self.patches[site_key] = [p.upper() for p in site.patches]

            # Collect all atoms for this site's REPLICATE selection
            all_atoms: set[str] = set()

            for patch in site.patches:
                patch_upper = patch.upper()
                atoms = patch_atoms.get(patch_upper, [])
                self.atom_groups[patch_upper] = atoms
                all_atoms.update(atoms)

                del_atoms = deleted_atoms.get(patch_upper, [])
                all_atoms.update(del_atoms)

                # Set pKa (user-provided takes precedence over auto-detected)
                if patch in site.pka_values:
                    self.pka[patch_upper] = str(site.pka_values[patch])
                elif patch_upper in patch_pka:
                    self.pka[patch_upper] = patch_pka[patch_upper]

            self.atom_groups[site_key] = list(all_atoms)

            if site.reference_patch:
                self.reference_patches[site_key] = site.reference_patch.upper()

            print(f"Registered site {site_key} with patches: {self.patches[site_key]}")
            print(f"  Atoms for REPLICATE: {self.atom_groups[site_key]}")

        self.site_keys[resname] = site_key_list
        print(f"Multi-site ligand {resname}: {len(site_key_list)} sites registered")

    def _register_singlesite_ligand(
        self, resname: str, patch_file: Path, ligand_def: LigandPatchDef
    ) -> None:
        """Register a single-site ligand (original behavior)."""
        # Parse all patches to get atoms, pKa values, and deleted atoms
        patch_atoms, patch_pka, deleted_atoms = self._parse_ligand_patch_file(
            patch_file, list(ligand_def.patches)
        )

        # Set patches for this residue (only alternates, reference is auto-generated)
        self.patches[resname] = [patch.upper() for patch in ligand_def.patches]

        # Collect all atoms across ALL patches for REPLICATE
        all_atoms: set[str] = set()

        # Include atoms from alternate patches
        for patch in ligand_def.patches:
            patch_upper = patch.upper()
            atoms = patch_atoms.get(patch_upper, [])
            self.atom_groups[patch_upper] = atoms
            all_atoms.update(atoms)

            # Include deleted atoms in the reference state
            del_atoms = deleted_atoms.get(patch_upper, [])
            all_atoms.update(del_atoms)

            # Set pKa (user-provided takes precedence over auto-detected)
            if patch in ligand_def.pka_values:
                self.pka[patch_upper] = str(ligand_def.pka_values[patch])
            elif patch_upper in patch_pka:
                self.pka[patch_upper] = patch_pka[patch_upper]

        # Set all atoms for the residue (union of atoms + deleted atoms)
        self.atom_groups[resname] = list(all_atoms)
        if ligand_def.reference_patch:
            self.reference_patches[resname] = ligand_def.reference_patch.upper()

        print(f"Registered ligand {resname} with patches: {self.patches[resname]}")
        print(f"  Deleted atoms to include: {[a for p in deleted_atoms.values() for a in p]}")
        print(f"  All atoms for REPLICATE: {self.atom_groups[resname]}")

    def _parse_ligand_patch_file(
        self, patch_file: Path, patches: list[str]
    ) -> tuple[dict[str, list[str]], dict[str, str], dict[str, list[str]]]:
        """Parse a ligand patch file to extract atoms and pKa values.

        Args:
            patch_file: Path to the RTF file with PRES definitions.
            patches: List of patch names to extract.

        Returns:
            Tuple of (patch_atoms, patch_pka, deleted_atoms) dicts.
            deleted_atoms contains atoms that are deleted by each patch.
        """
        patch_atoms: dict[str, list[str]] = {}
        patch_pka: dict[str, str] = {}
        deleted_atoms: dict[str, list[str]] = {}
        patches_upper = [p.upper() for p in patches]

        with open(patch_file, "r") as f:
            lines = [line.upper() for line in f.readlines()]

        current_patch = None
        for i, line in enumerate(lines):
            if line.startswith("PRES"):
                parts = line.split()
                if len(parts) >= 2:
                    patch_name = parts[1]
                    if patch_name in patches_upper:
                        current_patch = patch_name
                        patch_atoms[current_patch] = []
                        deleted_atoms[current_patch] = []
                    else:
                        current_patch = None
            elif current_patch:
                # Check for pKa comment
                if "PKA" in line and "=" in line:
                    try:
                        pka_val = line.split("=")[1].strip().split()[0]
                        patch_pka[current_patch] = pka_val
                    except (IndexError, ValueError):
                        pass
                # Extract atoms from ATOM lines
                if line.startswith("ATOM"):
                    parts = line.split()
                    if len(parts) >= 2:
                        atom_name = parts[1]
                        if atom_name not in patch_atoms[current_patch]:
                            patch_atoms[current_patch].append(atom_name)
                # Extract atoms from DELETE ATOM lines (DELE or DELETE)
                if line.startswith("DELE") or line.startswith("DELETE"):
                    parts = line.split()
                    # Format: DELETE ATOM <atomname> or DELE ATOM <atomname>
                    if len(parts) >= 3 and parts[1] == "ATOM":
                        atom_name = parts[2]
                        if atom_name not in deleted_atoms[current_patch]:
                            deleted_atoms[current_patch].append(atom_name)
                # End of patch section
                if line.startswith("END") or (line.startswith("PRES") and i > 0):
                    current_patch = None

        return patch_atoms, patch_pka, deleted_atoms


class Universe:
    """Helper class for working with CHARMM atom data as pandas DataFrames."""

    def __init__(self):
        self.universe = self._update_universe()

    def _update_universe(self) -> pd.DataFrame:
        """Get current system state as DataFrame."""
        try:
            snapshot = system.get_topology_snapshot()
        except Exception as e:
            raise RuntimeError("No atoms in system") from e

        if snapshot.natom <= 0:
            raise RuntimeError("No atoms in system")

        self._atoms = snapshot.atoms
        self._natom = snapshot.natom
        return pd.DataFrame(
            {
                "index": list(range(1, snapshot.natom + 1)),
                "res_name": [atom.resname for atom in snapshot.atoms],
                "res_id": [atom.resid for atom in snapshot.atoms],
                "seg_id": [atom.segid for atom in snapshot.atoms],
                "chem_type": [atom.atom_name for atom in snapshot.atoms],
                "atom_type": [atom.atom_name for atom in snapshot.atoms],
                "x": [atom.x for atom in snapshot.atoms],
                "y": [atom.y for atom in snapshot.atoms],
                "z": [atom.z for atom in snapshot.atoms],
            }
        )

    def resid_of_resname(self, resname: str) -> np.ndarray:
        """Get unique residue IDs for a given residue name."""
        return self.universe[self.universe["res_name"] == resname]["res_id"].unique()

    def resname_of_resid(self, resid: int) -> np.ndarray:
        """Get residue names for a given residue ID."""
        return self.universe[self.universe["res_id"] == resid]["res_name"].unique()


def _parse_selection_criteria(selections: list[str]) -> list[tuple[str, str | tuple[str, str]]]:
    """Parse selection criteria strings into typed tuples.

    Args:
        selections: List of selection strings (e.g., ["ASP", "15", "PROA:15"]).

    Returns:
        List of (type, value) tuples where type is "resname", "resid", or "segid_resid".
    """
    parsed = []
    for item in selections:
        item_upper = item.upper()
        if ":" in item:
            parts = item.split(":", 1)
            segid, resid_str = parts[0].upper(), parts[1]
            if resid_str.isdigit():
                parsed.append(("segid_resid", (segid, resid_str)))
            else:
                print(f"Warning: Invalid segid:resid format '{item}'")
        elif item_upper.isalpha() and len(item_upper) <= 4:
            parsed.append(("resname", item_upper))
        elif item.isdigit():
            parsed.append(("resid", item))
        else:
            parsed.append(("resname", item_upper))
    return parsed


def _should_patch_residue(
    seg_id: str,
    res_id: int,
    res_name: str,
    criteria: list[tuple[str, str | tuple[str, str]]],
) -> bool:
    """Check if a residue should be patched based on selection criteria.

    Args:
        seg_id: Segment ID.
        res_id: Residue ID.
        res_name: Residue name.
        criteria: Parsed selection criteria.

    Returns:
        True if residue should be patched.
    """
    if not criteria:
        return True

    res_id_str = str(res_id)
    res_name_upper = res_name.upper()

    for criterion_type, value in criteria:
        if criterion_type == "resname" and res_name_upper == value:
            return True
        elif criterion_type == "resid" and res_id_str == value:
            return True
        elif criterion_type == "segid_resid":
            sel_seg, sel_res = value  # type: ignore
            if seg_id.upper() == sel_seg and res_id_str == sel_res:
                return True
    return False


def _read_topology_files(config: PatchConfig, verbose: bool = True) -> None:
    """Load CHARMM topology and parameter files.

    Loads standard topology files from toppar_dir, then any extra_files
    specified as absolute paths for custom residues or parameters.
    """
    toppar_dir = config.toppar_dir or TOPPAR_DIR

    if not verbose:
        system.set_prnlev(-1)

    prm_files = [f for f in config.topology_files if f.endswith(".prm")]
    rtf_files = [f for f in config.topology_files if f.endswith(".rtf")]
    str_files = [f for f in config.topology_files if f.endswith(".str")]

    system.set_bomb_level(-2)
    system.set_warn_level(-1)

    if rtf_files:
        system.read_rtf(toppar_dir / rtf_files[0])
        for f in rtf_files[1:]:
            system.read_rtf(toppar_dir / f, append=True)

    if prm_files:
        system.read_param(toppar_dir / prm_files[0])
        for f in prm_files[1:]:
            system.read_param(toppar_dir / f, append=True)

    for f in str_files:
        system.stream_file(toppar_dir / f)

    # Load extra files (absolute paths for custom residues/parameters)
    for extra_file in config.extra_files:
        extra_path = Path(extra_file)
        if not extra_path.exists():
            raise FileNotFoundError(f"Extra topology file not found: {extra_path}")

        suffix = extra_path.suffix.lower()
        if suffix == ".rtf":
            system.read_rtf(extra_path, append=True)
        elif suffix == ".prm":
            system.read_param(extra_path, append=True)
        elif suffix == ".str":
            system.stream_file(extra_path)
        else:
            # Try streaming unknown file types
            system.stream_file(extra_path)

    # Load ligand patch files
    for ligand_def in config.ligand_patches:
        patch_path = Path(ligand_def.patch_file)
        suffix = patch_path.suffix.lower()
        if suffix == ".rtf":
            system.read_rtf(patch_path, append=True)
        elif suffix == ".str":
            system.stream_file(patch_path)
        else:
            # Try reading as RTF for unknown patch file types
            system.read_rtf(patch_path, append=True)
        print(f"Loaded ligand patch file: {patch_path}")

    system.set_warn_level(5)
    system.set_bomb_level(0)

    if not verbose:
        system.set_prnlev(5)


def _declares_histidine_patches(path: Path) -> bool:
    if path.name.lower() == "his_patches.str":
        return True
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").upper()
    except OSError:
        return False
    return "PRES HSDP" in text and "PRES HSEP" in text


def _config_loads_histidine_patches(config: PatchConfig, toppar_dir: Path) -> bool:
    paths: list[Path] = []
    for file_name in config.topology_files:
        path = Path(file_name)
        paths.append(path if path.is_absolute() else toppar_dir / path)
    paths.extend(Path(path) for path in config.extra_files)
    return any(_declares_histidine_patches(path) for path in paths)


def _detect_disulfide_bonds(
    uni: Universe, residues: list[tuple[str, int, str]]
) -> tuple[list[tuple[str, int, str]], set[tuple[str, int, str]]]:
    """Detect and filter out disulfide-bonded cysteines.

    Args:
        uni: Universe object with current system.
        residues: List of (seg_id, res_id, res_name) tuples for CYS residues.

    Returns:
        Tuple of (filtered_residues, bonded_residues).
    """
    cys_universe = uni.universe[
        (uni.universe["res_name"] == "CYS") & (uni.universe["atom_type"] == "SG")
    ]

    bonded_res: set[tuple[str, int, str]] = set()

    if not cys_universe.empty:
        num_cys_sg = len(cys_universe)
        for i in range(num_cys_sg):
            cys1 = cys_universe.iloc[i]
            cys1_pos = np.array([cys1["x"], cys1["y"], cys1["z"]])
            for j in range(i + 1, num_cys_sg):
                cys2 = cys_universe.iloc[j]

                if cys1["res_id"] == cys2["res_id"]:
                    continue

                cys2_pos = np.array([cys2["x"], cys2["y"], cys2["z"]])
                distance = np.linalg.norm(cys1_pos - cys2_pos)

                if distance < 2.6:
                    print(f"CYS {cys1['res_id']} is disulfide bonded to {cys2['res_id']}")
                    bonded_res.update(
                        [
                            (cys1["seg_id"], cys1["res_id"], cys1["res_name"]),
                            (cys2["seg_id"], cys2["res_id"], cys2["res_name"]),
                        ]
                    )

    filtered = [res for res in residues if res not in bonded_res]
    for res in filtered:
        print(f"Patching CYS {res[1]} not involved in a disulfide bond.")

    return filtered, bonded_res


def _fft_number(n: float) -> int:
    """Find the smallest FFT-friendly number >= n.

    FFT-friendly numbers are products of 2, 3, and 5 only,
    and must be even (for CHARMM FFT requirements).
    """
    n = int(n)
    if n <= 2:
        return 2

    # Generate all smooth numbers (only factors 2, 3, 5) up to 2*n
    smooth_numbers = []
    limit = max(n * 2, 256)

    for i in range(2, limit + 1):
        temp = i
        for factor in [2, 3, 5]:
            while temp % factor == 0:
                temp //= factor
        if temp == 1 and i % 2 == 0:  # Must be smooth and even
            smooth_numbers.append(i)

    # Return the smallest one >= n
    for num in smooth_numbers:
        if num >= n:
            return num

    return n  # Fallback


def _stream_charmm_script(script: str, directory: Path, prefix: str = "patching") -> None:
    """Stream a generated CHARMM snippet without leaving persistent run artifacts."""
    directory.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        suffix=".inp",
        prefix=f"{prefix}_",
        delete=False,
    )
    path = Path(handle.name)
    try:
        with handle:
            handle.write(script.rstrip())
            handle.write("\n")
        system.stream_file(path)
    finally:
        path.unlink(missing_ok=True)


def _convert_histidines_for_segment(
    titratable_list: list[tuple[str, int, str]],
    his_residues: list[tuple[str, int, str]],
    tmp_dir: Path | None,
) -> None:
    """Convert HSD/HSE residues to HSP before selection filtering."""
    applier = PatchApplier()
    segids = list(dict.fromkeys(segid for segid, _, _ in his_residues))
    for seg_id in segids:
        results = applier.apply_to_topology(
            candidate_resnames=frozenset({"HSD", "HSE"}),
            segid_filter=seg_id,
        )
        for result in results:
            print(f"Patching {result.patch_name} {result.segid} {result.resid}")
            titratable_list.append((result.segid, result.resid, result.target_resname))
        if tmp_dir is not None and results:
            system.hbuild(AtomSelection(segid=seg_id, hydrogens=True))


def _filter_titratable_residues(
    titratable_list: list[tuple[str, int, str]],
    criteria: list[tuple[str, str | tuple[str, str]]],
) -> list[tuple[str, int, str]]:
    return [
        item
        for item in titratable_list
        if _should_patch_residue(item[0], item[1], item[2], criteria)
    ]


def patch_system(config: PatchConfig) -> Path:
    """Apply CpHMD patches to titratable residues.

    This function transforms a standard molecular structure into one with
    multiple protonation states for each titratable residue, suitable for
    constant pH molecular dynamics.

    Args:
        config: Patching configuration.

    Returns:
        Path to the output folder containing output files.

    Output files:
        - system.pdb, system.psf, system.crd: Patched system
        - system_hmr.psf, system_hmr.crd: With HMR (if enabled)
        - patches.dat: Patch information for each site
        - select.str: CHARMM selection definitions
        - box.dat, size.dat, fft.dat: Box parameters
    """
    toppar_dir = config.toppar_dir or TOPPAR_DIR
    output_folder = config.output_folder

    # Resolve input file paths
    if config.psf_file is not None and config.crd_file is not None:
        # Direct file paths provided
        psf_path = Path(config.psf_file)
        crd_path = Path(config.crd_file)
        print(f"Using direct file paths: PSF={psf_path}, CRD={crd_path}")
    else:
        # Folder-based paths
        input_folder = Path(config.input_folder)
        psf_path = input_folder / f"{config.structure_file}.psf"
        crd_path = input_folder / f"{config.structure_file}.crd"
        print(f"Using folder-based paths: {input_folder}/{config.structure_file}.*")

    # Validate input files
    for fpath in [psf_path, crd_path]:
        if not fpath.exists():
            raise FileNotFoundError(f"File not found: {fpath}")
        if fpath.stat().st_size == 0:
            raise ValueError(f"File is empty: {fpath}")

    # Ensure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Parse selection criteria
    criteria = _parse_selection_criteria(config.selected_residues)

    # Load topology
    _read_topology_files(config)
    if not _config_loads_histidine_patches(config, toppar_dir):
        ensure_patches_streamed(force=True)
    system.set_iofmt(extended=True)

    # Ligand fragments may have fractional charge — suppress PSF charge warning
    if config.ligand_patches:
        system.set_bomb_level(-2)

    # Load structure
    system.read_psf(psf_path)
    system.read_coor(crd_path)

    uni = Universe()
    patches_topology = PatchParser(
        segment_path=toppar_dir / "my_files" / "titratable_residues.str",
        topology_path=toppar_dir / "top_all36_prot.rtf",
    )

    # Register custom ligand patches
    for ligand_def in config.ligand_patches:
        patches_topology.register_ligand(ligand_def)

    patches_topology.print_atoms()

    # Find titratable residues
    titratable_list: list[tuple[str, int, str]] = []
    for resname in patches_topology.residues:
        residues = [
            (row["seg_id"], row["res_id"], row["res_name"])
            for _, row in uni.universe[uni.universe["res_name"] == resname][
                ["seg_id", "res_id", "res_name"]
            ]
            .drop_duplicates()
            .iterrows()
        ]

        # Handle CYS disulfide bonds
        if resname == "CYS":
            residues, _ = _detect_disulfide_bonds(uni, residues)

        # Expand multi-site ligands into separate entries per site
        if resname in patches_topology.site_keys:
            for seg_id, resid, _ in residues:
                for site_key in patches_topology.site_keys[resname]:
                    titratable_list.append((seg_id, resid, site_key))
        else:
            titratable_list.extend(residues)

    # Process segments
    seg_ids = uni.universe["seg_id"].unique().tolist()
    print(f"Structure segids: {seg_ids}")

    tmp_dir = output_folder / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    for seg_id in seg_ids:
        if seg_id != seg_ids[0]:
            system.delete_atoms(AtomSelection())
            system.reset_io_unit(91)
            system.read_psf(psf_path)
            system.read_coor(crd_path)

        if len(seg_ids) > 1:
            system.delete_atoms(AtomSelection(raw=f".not. segid {seg_id}"))

        # Handle HSD/HSE → HSP conversion
        his_residues = [
            (row["seg_id"], row["res_id"], row["res_name"])
            for _, row in uni.universe[
                (uni.universe["res_name"].isin(["HSD", "HSE"])) & (uni.universe["seg_id"] == seg_id)
            ][["seg_id", "res_id", "res_name"]]
            .drop_duplicates()
            .iterrows()
        ]

        _convert_histidines_for_segment(titratable_list, his_residues, tmp_dir)

        system.hbuild()
        system.write_psf(tmp_dir / f"{seg_id.lower()}.psf")
        system.write_coor(tmp_dir / f"{seg_id.lower()}.crd")

    # Filter titratable residues based on selection criteria
    filtered_list = _filter_titratable_residues(titratable_list, criteria)
    dropped_count = len(titratable_list) - len(filtered_list)
    print(f"Filtered {dropped_count} residues based on selection criteria.")
    titratable_list = filtered_list

    print("Final list of residues selected for patching:")
    for seg, resid, resn in titratable_list:
        print(f"  Segment: {seg}, Residue ID: {resid}, Residue Name: {resn}")

    # Initialize patches.dat
    patches_file = output_folder / "patches.dat"
    with open(patches_file, "w") as f:
        f.write("SEGID,RESID,PATCH,SELECT,ATOMS,TAG\n")

    # Apply patches per segment
    # Keep bomblevel -2 for entire patching loop to handle non-integer charge warnings
    system.set_bomb_level(-2)
    global_idx = 0
    for seg_id in seg_ids:
        with open(patches_file, "a") as f:
            system.delete_atoms(AtomSelection())
            system.reset_io_unit(91)
            system.read_psf(tmp_dir / f"{seg_id.lower()}.psf")
            system.read_coor(tmp_dir / f"{seg_id.lower()}.crd")

            sublist = [res for res in titratable_list if res[0] == seg_id]
            print(f"Patching segment {seg_id}: {sublist}")

            global_idx = _apply_patches(output_folder, patches_topology, sublist, global_idx, f)

        system.ic_build()
        system.ic_prm_fill(comp=True)
        # Build coordinates for newly added hydrogens (e.g., H36, H37 from ligand patches)
        system.hbuild()
        system.write_coor(tmp_dir / f"{seg_id.lower()}.crd")
        system.write_psf(tmp_dir / f"{seg_id.lower()}.psf")

    # Combine segments
    system.delete_atoms(AtomSelection())
    for i, seg_id in enumerate(seg_ids):
        system.reset_io_unit(91)
        if i == 0:
            system.read_psf(tmp_dir / f"{seg_id.lower()}.psf")
            system.read_coor(tmp_dir / f"{seg_id.lower()}.crd")
        else:
            system.read_psf(tmp_dir / f"{seg_id.lower()}.psf", append=True)
            system.read_coor(tmp_dir / f"{seg_id.lower()}.crd", append=True)

    # Reset bomblevel after all patching is complete
    system.set_bomb_level(0)

    # Write output files
    system.write_coor_pdb(output_folder / "system.pdb")
    system.write_psf(output_folder / "system.psf")
    system.write_coor(output_folder / "system.crd")

    # Apply HMR
    if config.hmr:
        if config.hmr_waters:
            system.psf_hmr(resnames_exclude=[])
            print("HMR enabled for system")
        else:
            system.psf_hmr()
            print("HMR enabled for system without waters")
        system.write_psf(output_folder / "system_hmr.psf")
        system.write_coor(output_folder / "system_hmr.crd")

    # Copy box.dat from input folder if it exists (preserves crystal type from solvation)
    if config.input_folder is not None:
        src_box = Path(config.input_folder) / "box.dat"
        dst_box = output_folder / "box.dat"
        if src_box.exists() and not dst_box.exists():
            shutil.copy(src_box, dst_box)

    # Calculate box parameters (only creates box.dat if not already present)
    _write_box_params(output_folder)

    # Write selection file
    _write_selections(output_folder, patches_topology, titratable_list)

    # Cleanup
    shutil.rmtree(tmp_dir)

    print(f"Patching completed. Output files in: {output_folder}")
    return output_folder


def _apply_patches(
    input_folder: Path,
    patches_topology: PatchParser,
    titratable_list: list[tuple[str, int, str]],
    start_idx: int,
    f,
) -> int:
    """Apply patches to titratable residues using CHARMM REPLICATE command."""
    tmp_dir = input_folder / "tmp"
    system.disable_autogen()
    idx = start_idx

    # Track patches processed at each (seg_id, resid) for multi-site ligands.
    # When processing site 2+, these must be excluded from AUTO ANGLES to prevent
    # re-creating cross-block angles for earlier sites whose REPLICATE was reset.
    processed_patches: dict[tuple[str, int], list[str]] = {}

    for res_idx, residue in enumerate(titratable_list):
        seg_id, resid, resname = residue
        idx += 1

        # Check if first occurrence of this segment
        if [res[0] for res in titratable_list].index(seg_id) == titratable_list.index(residue):
            system.reset_io_unit(91)
            system.read_psf(tmp_dir / f"{seg_id.lower()}.psf")
            system.read_coor(tmp_dir / f"{seg_id.lower()}.crd")

        # Replicate the titratable atom group into one segment per patch state.
        n_rep = len(patches_topology.patches[resname]) + 1
        atoms = patches_topology.atom_groups[resname]
        system.replicate_atoms(
            seg_id,
            resid,
            atoms,
            replica_segid=str(resid),
            nreplica=n_rep,
            setup=True,
        )

        # Delete atoms in original residue
        system.delete_atoms_by_names(seg_id, resid, atoms)

        # Process each patch
        # Check if a reference_patch is defined (ligand with explicit reference state)
        ref_patch = patches_topology.reference_patches.get(resname)
        ref_pka = patches_topology.pka.get(ref_patch) if ref_patch else None

        # Compute micro-pKa for UNEG states sharing ref_pka.
        # User-provided pKa is the macro-pKa; when n tautomers share it,
        # each micro-pKa = macro - log10(n) so the Boltzmann model
        # reproduces the correct macro midpoint.
        ref_micro_pka = ref_pka
        if ref_patch and ref_pka:
            import math

            n_uneg = 1  # original state always gets UNEG
            for p in patches_topology.patches[resname]:
                if p.upper() != ref_patch and p not in patches_topology.pka:
                    n_uneg += 1
            if n_uneg > 1:
                ref_micro_pka = f"{float(ref_pka) - math.log10(n_uneg):.2f}"

        j = 1
        for patch in [resname + "O"] + patches_topology.patches[resname]:
            if patch != resname + "O":
                system.patch(patch, f"{resid}{j}", resid, setup=True)

                atoms_str = " ".join(patches_topology.atom_groups[patch])
                if ref_patch and patch.upper() == ref_patch:
                    # Explicit reference patch → NONE
                    f.write(f"{seg_id},{resid},{patch},s{idx}s{j},{atoms_str},NONE\n")
                elif patch not in patches_topology.pka:
                    if ref_patch and ref_micro_pka:
                        # Tautomer without pKa, but reference is defined → UNEG
                        f.write(
                            f"{seg_id},{resid},{patch},s{idx}s{j},{atoms_str},"
                            f"UNEG {ref_micro_pka}\n"
                        )
                    else:
                        f.write(f"{seg_id},{resid},{patch},s{idx}s{j},{atoms_str},NONE\n")
                elif (
                    len(patches_topology.atom_groups[patch])
                    - len(patches_topology.atom_groups[resname])
                    > 0
                ):
                    f.write(
                        f"{seg_id},{resid},{patch},s{idx}s{j},{atoms_str},"
                        f"UNEG {patches_topology.pka[patch]}\n"
                    )
                else:
                    f.write(
                        f"{seg_id},{resid},{patch},s{idx}s{j},{atoms_str},"
                        f"UPOS {patches_topology.pka[patch]}\n"
                    )
            else:
                atoms_str = " ".join(patches_topology.atom_groups[patch[:-1]])
                if ref_patch and ref_micro_pka:
                    # Original state with reference defined → UNEG
                    f.write(
                        f"{seg_id},{resid},{patch},s{idx}s{j},{atoms_str},UNEG {ref_micro_pka}\n"
                    )
                else:
                    f.write(f"{seg_id},{resid},{patch},s{idx}s{j},{atoms_str},NONE\n")

            system.rename_residues(AtomSelection(segid=f"{resid}{j}"), patch)
            system.join_segments(seg_id, f"{resid}{j}")
            j += 1

        # Build AUTO ANGLES selection, excluding patches from previously processed
        # sites at the same (seg_id, resid). Without exclusion, AUTO ANGLES
        # re-creates cross-block angles for earlier sites whose REPLICATE was reset.
        key = (seg_id, resid)
        prev_patches = processed_patches.get(key, [])
        if prev_patches:
            exclude = " .or. ".join([f"resname {p}" for p in prev_patches])
            auto_sel = f"sele segid {seg_id} .and. resid {resid} .and. .not. ({exclude}) end"
        else:
            auto_sel = f"sele segid {seg_id} .and. resid {resid} end"

        _stream_charmm_script(
            "\n".join(
                [
                    f"AUTO ANGLES DIHEDRALS {auto_sel}",
                    f"IC PARAM {auto_sel}",
                    f"IC FILL PRESERVE {auto_sel}",
                    "REPLIcate RESEt",
                ]
            ),
            tmp_dir,
            "auto_ic",
        )

        # Delete connections between different patches within this site
        patches = [resname + "O"] + patches_topology.patches[resname]
        for pair in itertools.combinations(patches, 2):
            system.delete_connectivity(
                AtomSelection(segid=seg_id, resid=resid, resname=pair[0]),
                AtomSelection(segid=seg_id, resid=resid, resname=pair[1]),
            )

        # Track this site's patches for future exclusion
        processed_patches.setdefault(key, []).extend(patches)

    return idx


def _write_box_params(input_folder: Path) -> None:
    """Write box parameters and FFT numbers."""
    box_file = input_folder / "box.dat"

    if not box_file.exists():
        stats = system.coor_stat()
        xmax, xmin = stats["xmax"], stats["xmin"]
        ymax, ymin = stats["ymax"], stats["ymin"]
        zmax, zmin = stats["zmax"], stats["zmin"]
        max_size = max(int(xmax - xmin), int(ymax - ymin), int(zmax - zmin))
        box_size = [max_size, max_size, max_size]

        with open(box_file, "w") as f:
            f.write("CUBic\n")
            f.write(f"{box_size[0]} {box_size[1]} {box_size[2]}\n")
            f.write("90.0 90.0 90.0\n")

    with open(box_file, "r") as f:
        lines = f.readlines()
    box_size = [float(x) for x in lines[1].split()]

    with open(input_folder / "size.dat", "w") as f:
        f.write(str(box_size[0]))

    fft = [_fft_number(x) for x in box_size]
    with open(input_folder / "fft.dat", "w") as f:
        f.write(" ".join(str(x) for x in fft))


def _write_selections(
    input_folder: Path,
    patches_topology: PatchParser,
    titratable_list: list[tuple[str, int, str]],
) -> None:
    """Write CHARMM selection definitions file."""
    with open(input_folder / "select.str", "w") as f:
        for i, residue in enumerate(titratable_list, 1):
            seg_id, resid, resname = residue
            patches = [resname + "O"] + patches_topology.patches[resname]
            for j, patch in enumerate(patches, 1):
                f.write(
                    f"DEFIne s{i}s{j} SELEct segid {seg_id} .and. resid {resid} "
                    f".and. resname {patch} END\n"
                )
