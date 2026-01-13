"""
CpHMD patching module.

This module applies titratable residue patches to create multiple protonation
states for constant pH molecular dynamics simulations.
"""

from __future__ import annotations

import itertools
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import pycharmm
import pycharmm.coor as coor
import pycharmm.ic as ic
import pycharmm.lingo as lingo
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.select as select
import pycharmm.settings as settings
import pycharmm.write as write

from cphmd import TOPPAR_DIR


@dataclass
class PatchConfig:
    """Configuration for patching titratable residues.

    Attributes:
        input_folder: Input folder containing structure files.
        structure_file: Base name of structure file (without extension).
        hmr: Enable hydrogen mass repartitioning.
        hmr_waters: Apply HMR to water molecules too.
        selected_residues: List of residue selections (e.g., ["ASP", "PROA:15"]).
        toppar_dir: Path to topology directory.
        topology_files: List of topology files to load.
    """

    input_folder: str | Path
    structure_file: str = "solvated"
    hmr: bool = True
    hmr_waters: bool = False
    selected_residues: list[str] = field(default_factory=list)
    toppar_dir: Path | None = None
    topology_files: list[str] = field(default_factory=lambda: [
        "top_all36_prot.rtf",
        "par_all36m_prot.prm",
        "top_all36_na.rtf",
        "par_all36_na.prm",
        "toppar_water_ions.str",
        "top_all36_cgenff.rtf",
        "par_all36_cgenff.prm",
        "my_files/titratable_residues.str",
        "my_files/nucleic_c36.str",
        "my_files/his_patches.str",
    ])


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
                        bonds = lines[j].split()[1:]

                # Filter atoms to only those in topology
                self.atom_groups[resname] = [
                    atom for atom in self.atom_groups[resname] if atom in atoms
                ]

                # Add hydrogen atoms bonded to heavy atoms in the group
                for k in range(0, len(bonds), 2):
                    if k + 1 >= len(bonds):
                        break
                    if bonds[k] in self.atom_groups[resname] and not bonds[k].startswith("H"):
                        if bonds[k + 1] not in self.atom_groups[resname] and bonds[k + 1].startswith("H"):
                            self.atom_groups[resname].append(bonds[k + 1])
                    if bonds[k + 1] in self.atom_groups[resname] and not bonds[k + 1].startswith("H"):
                        if bonds[k] not in self.atom_groups[resname] and bonds[k].startswith("H"):
                            self.atom_groups[resname].append(bonds[k])

    def print_atoms(self) -> None:
        """Print atom groups with pKa values."""
        for name in self.atom_groups:
            if name in self.pka:
                print(f"{name}, pKa {self.pka[name]}: {self.atom_groups[name]}\n")
            else:
                print(f"{name}: {self.atom_groups[name]}\n")


class Universe:
    """Helper class for working with CHARMM atom data as pandas DataFrames."""

    def __init__(self):
        self.universe = self._update_universe()

    def _update_universe(self) -> pd.DataFrame:
        """Get current system state as DataFrame."""
        try:
            select_all = pycharmm.SelectAtoms().all_atoms()
        except Exception as e:
            raise RuntimeError("No atoms in system") from e

        crds = coor.get_positions()
        return pd.DataFrame({
            "index": select_all._atom_indexes,
            "res_name": select_all._res_names,
            "res_id": select_all._res_ids,
            "seg_id": select_all._seg_ids,
            "chem_type": select_all._chem_types,
            "atom_type": select_all._atom_types,
            "x": crds["x"].values,
            "y": crds["y"].values,
            "z": crds["z"].values,
        })

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
    """Load CHARMM topology and parameter files."""
    toppar_dir = config.toppar_dir or TOPPAR_DIR

    if not verbose:
        lingo.charmm_script("prnlev -1")

    prm_files = [f for f in config.topology_files if f.endswith(".prm")]
    rtf_files = [f for f in config.topology_files if f.endswith(".rtf")]
    str_files = [f for f in config.topology_files if f.endswith(".str")]

    lingo.charmm_script("bomblevel -2")
    settings.set_warn_level(-1)

    if rtf_files:
        read.rtf(str(toppar_dir / rtf_files[0]))
        for f in rtf_files[1:]:
            read.rtf(str(toppar_dir / f), append=True)

    if prm_files:
        read.prm(str(toppar_dir / prm_files[0]), flex=True)
        for f in prm_files[1:]:
            read.prm(str(toppar_dir / f), flex=True, append=True)

    for f in str_files:
        lingo.charmm_script(f"stream {toppar_dir / f}")

    settings.set_warn_level(5)
    lingo.charmm_script("bomblevel 0")

    if not verbose:
        lingo.charmm_script("prnlev 5")


def _detect_disulfide_bonds(uni: Universe, residues: list[tuple[str, int, str]]) -> tuple[
    list[tuple[str, int, str]], set[tuple[str, int, str]]
]:
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
                    bonded_res.update([
                        (cys1["seg_id"], cys1["res_id"], cys1["res_name"]),
                        (cys2["seg_id"], cys2["res_id"], cys2["res_name"]),
                    ])

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


def patch_system(config: PatchConfig) -> Path:
    """Apply CpHMD patches to titratable residues.

    This function transforms a standard molecular structure into one with
    multiple protonation states for each titratable residue, suitable for
    constant pH molecular dynamics.

    Args:
        config: Patching configuration.

    Returns:
        Path to the input folder containing output files.

    Output files:
        - system.pdb, system.psf, system.crd: Patched system
        - system_hmr.psf, system_hmr.crd: With HMR (if enabled)
        - patches.dat: Patch information for each site
        - select.str: CHARMM selection definitions
        - box.dat, size.dat, fft.dat: Box parameters
    """
    input_folder = Path(config.input_folder)
    toppar_dir = config.toppar_dir or TOPPAR_DIR

    # Validate input files
    required_files = [f"{config.structure_file}.crd", f"{config.structure_file}.psf"]
    for f in required_files:
        fpath = input_folder / f
        if not fpath.exists():
            raise FileNotFoundError(f"{f} not found in {input_folder}")
        if fpath.stat().st_size == 0:
            raise ValueError(f"{f} is empty")

    # Parse selection criteria
    criteria = _parse_selection_criteria(config.selected_residues)

    # Load topology
    _read_topology_files(config)
    lingo.charmm_script("IOFOrmat EXTEnded")

    # Load structure
    read.psf_card(str(input_folder / f"{config.structure_file}.psf"))
    read.coor_card(str(input_folder / f"{config.structure_file}.crd"))

    uni = Universe()
    patches_topology = PatchParser(
        segment_path=toppar_dir / "my_files" / "titratable_residues.str",
        topology_path=toppar_dir / "top_all36_prot.rtf",
    )
    patches_topology.print_atoms()

    # Find titratable residues
    titratable_list: list[tuple[str, int, str]] = []
    for resname in patches_topology.residues:
        residues = [
            (row["seg_id"], row["res_id"], row["res_name"])
            for _, row in uni.universe[uni.universe["res_name"] == resname][
                ["seg_id", "res_id", "res_name"]
            ].drop_duplicates().iterrows()
        ]

        # Handle CYS disulfide bonds
        if resname == "CYS":
            residues, _ = _detect_disulfide_bonds(uni, residues)

        titratable_list.extend(residues)

    # Process segments
    seg_ids = uni.universe["seg_id"].unique().tolist()
    print(f"Structure segids: {seg_ids}")

    tmp_dir = input_folder / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    for seg_id in seg_ids:
        if seg_id != seg_ids[0]:
            psf.delete_atoms(pycharmm.SelectAtoms().all_atoms())
            read.psf_card(str(input_folder / f"{config.structure_file}.psf"))
            read.coor_card(str(input_folder / f"{config.structure_file}.crd"))

        if len(seg_ids) > 1:
            lingo.charmm_script(f"DELEte ATOMs SELEct .not. segid {seg_id} END")

        # Handle HSD/HSE → HSP conversion
        his_residues = [
            (row["seg_id"], row["res_id"], row["res_name"])
            for _, row in uni.universe[
                (uni.universe["res_name"].isin(["HSD", "HSE"])) &
                (uni.universe["seg_id"] == seg_id)
            ][["seg_id", "res_id", "res_name"]].drop_duplicates().iterrows()
        ]

        for res in his_residues:
            print(f"Patching HSP {res[0]} {res[1]}")
            pycharmm.charmm_script(f"rename resn HSP sele segid {res[0]} .and. resid {res[1]} end")
            pycharmm.charmm_script(f"patch {res[2]}P {res[0]} {res[1]}")
            pycharmm.charmm_script(f"HBUILD sele segid {res[0]} .and. hydrogen end")
            titratable_list.append((res[0], res[1], "HSP"))

        pycharmm.charmm_script("HBuild")
        write.psf_card(str(tmp_dir / f"{seg_id}.psf"))
        write.coor_card(str(tmp_dir / f"{seg_id}.crd"))

    # Filter titratable residues based on selection criteria
    filtered_list = [
        item for item in titratable_list
        if _should_patch_residue(item[0], item[1], item[2], criteria)
    ]
    print(f"Filtered {len(titratable_list) - len(filtered_list)} residues based on selection criteria.")
    titratable_list = filtered_list

    print("Final list of residues selected for patching:")
    for seg, resid, resn in titratable_list:
        print(f"  Segment: {seg}, Residue ID: {resid}, Residue Name: {resn}")

    # Update HSD/HSE → HSP in universe
    uni.universe.loc[uni.universe["res_name"] == "HSD", "res_name"] = "HSP"
    uni.universe.loc[uni.universe["res_name"] == "HSE", "res_name"] = "HSP"

    # Initialize patches.dat
    patches_file = input_folder / "patches.dat"
    with open(patches_file, "w") as f:
        f.write("SEGID,RESID,PATCH,SELECT,ATOMS,TAG\n")

    # Apply patches per segment
    global_idx = 0
    for seg_id in seg_ids:
        with open(patches_file, "a") as f:
            psf.delete_atoms(pycharmm.SelectAtoms().all_atoms())
            read.psf_card(str(tmp_dir / f"{seg_id}.psf"))
            read.coor_card(str(tmp_dir / f"{seg_id}.crd"))

            sublist = [res for res in titratable_list if res[0] == seg_id]
            print(f"Patching segment {seg_id}: {sublist}")

            global_idx = _apply_patches(
                input_folder, patches_topology, sublist, global_idx, f
            )

        ic.build()
        ic.prm_fill(replace_all=True)
        write.coor_card(str(tmp_dir / f"{seg_id}.crd"))
        write.psf_card(str(tmp_dir / f"{seg_id}.psf"))

    # Combine segments
    psf.delete_atoms(pycharmm.SelectAtoms().all_atoms())
    for i, seg_id in enumerate(seg_ids):
        if i == 0:
            read.psf_card(str(tmp_dir / f"{seg_id}.psf"))
            read.coor_card(str(tmp_dir / f"{seg_id}.crd"))
        else:
            read.psf_card(str(tmp_dir / f"{seg_id}.psf"), append=True)
            read.coor_card(str(tmp_dir / f"{seg_id}.crd"), append=True)

    # Write output files
    write.coor_pdb(str(input_folder / "system.pdb"))
    write.psf_card(str(input_folder / "system.psf"))
    write.coor_card(str(input_folder / "system.crd"))

    # Apply HMR
    if config.hmr:
        if config.hmr_waters:
            psf.hmr(resnames_exclude=[])
            print("HMR enabled for system")
        else:
            psf.hmr()
            print("HMR enabled for system without waters")
        write.psf_card(str(input_folder / "system_hmr.psf"))
        write.coor_card(str(input_folder / "system_hmr.crd"))

    # Calculate box parameters
    _write_box_params(input_folder)

    # Write selection file
    _write_selections(input_folder, patches_topology, titratable_list)

    # Cleanup
    shutil.rmtree(tmp_dir)

    # Move files to prep folder
    prep_dir = input_folder / "prep"
    prep_dir.mkdir(exist_ok=True)
    for f in input_folder.iterdir():
        if f.is_dir():
            continue
        if f.suffix in [".out", ".err"]:
            continue
        shutil.move(str(f), str(prep_dir / f.name))

    print("Patching completed. Files moved to prep/")
    return input_folder


def _apply_patches(
    input_folder: Path,
    patches_topology: PatchParser,
    titratable_list: list[tuple[str, int, str]],
    start_idx: int,
    f,
) -> int:
    """Apply patches to titratable residues using CHARMM REPLICATE command."""
    pycharmm.charmm_script("AUTO NOPAtch")
    idx = start_idx

    for residue in titratable_list:
        seg_id, resid, resname = residue
        idx += 1

        # Check if first occurrence of this segment
        if [res[0] for res in titratable_list].index(seg_id) == titratable_list.index(residue):
            read.psf_card(str(input_folder / "tmp" / f"{seg_id}.psf"))
            read.coor_card(str(input_folder / "tmp" / f"{seg_id}.crd"))

        # Build REPLICATE command
        n_rep = len(patches_topology.patches[resname]) + 1
        charmm_cmd = f"REPLicate {resid} NREP {n_rep} SETUP -\n"

        # Build selection for atoms to replicate
        atoms = patches_topology.atom_groups[resname]
        select_cmd = f"SELEct segid {seg_id} .and. resid {resid} .and. ("
        select_cmd += " .or. ".join([f"-\ntype {atom}" for atom in atoms])
        select_cmd += ") END\n"

        pycharmm.charmm_script(charmm_cmd + select_cmd)

        # Delete atoms in original residue
        pycharmm.charmm_script(f"DELEte ATOMs {select_cmd}")

        # Process each patch
        j = 1
        for patch in [resname + "O"] + patches_topology.patches[resname]:
            cmd = f"! Working on {patch} {seg_id}:{resid}{j}:{resid}\n"

            if patch != resname + "O":
                cmd += f"PATCH {patch} {resid}{j} {resid} SETUP\n"

                atoms_str = " ".join(patches_topology.atom_groups[patch])
                if len(patches_topology.atom_groups[patch]) - len(patches_topology.atom_groups[resname]) > 0:
                    f.write(f"{seg_id},{resid},{patch},s{idx}s{j},{atoms_str},UNEG {patches_topology.pka[patch]}\n")
                else:
                    f.write(f"{seg_id},{resid},{patch},s{idx}s{j},{atoms_str},UPOS {patches_topology.pka[patch]}\n")
            else:
                atoms_str = " ".join(patches_topology.atom_groups[patch[:-1]])
                f.write(f"{seg_id},{resid},{patch},s{idx}s{j},{atoms_str},NONE\n")

            cmd += f"RENAMe RESName {patch} SELE segid {resid}{j} END\n"
            cmd += f"JOIN {seg_id} {resid}{j}\n"
            j += 1
            pycharmm.charmm_script(cmd)

        # Finalize residue
        pycharmm.charmm_script(f"AUTO ANGLES DIHEDRALS sele segid {seg_id} .and. resid {resid} end")
        pycharmm.charmm_script(f"IC PARAM sele segid {seg_id} .and. resid {resid} end")
        pycharmm.charmm_script(f"IC FILL PRESERVE sele segid {seg_id} .and. resid {resid} end")
        pycharmm.charmm_script("REPLIcate RESEt")

        # Delete connections between different patches
        patches = [resname + "O"] + patches_topology.patches[resname]
        for pair in itertools.combinations(patches, 2):
            pycharmm.charmm_script(
                f"DELETE CONN ATOMs SELEct segid {seg_id} .and. resid {resid} .and. resname {pair[0]} END "
                f"SELEct segid {seg_id} .and. resid {resid} .and. resname {pair[1]} END"
            )

    return idx


def _write_box_params(input_folder: Path) -> None:
    """Write box parameters and FFT numbers."""
    box_file = input_folder / "box.dat"

    if not box_file.exists():
        stats = coor.stat()
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
                f.write(f"DEFIne s{i}s{j} SELEct segid {seg_id} .and. resid {resid} .and. resname {patch} END\n")
