"""
Solvation module for CpHMD system preparation.

This module provides functions to solvate molecular systems in various
crystal box types with optional ion placement.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pycharmm
import pycharmm.coor as coor
import pycharmm.crystal as crystal
import pycharmm.energy as energy
import pycharmm.generate as gen
import pycharmm.image as image
import pycharmm.lingo as lingo
import pycharmm.minimize as minimize
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.settings as settings
import pycharmm.write as write

from cphmd import TOPPAR_DIR

# Crystal type aliases
CrystalType = Literal[
    "CUBIC", "TETRAGONAL", "ORTHORHOMBIC", "MONOCLINIC",
    "TRICLINIC", "HEXAGONAL", "RHOMBOHEDRAL", "OCTAHEDRAL", "RHDO"
]

# Ion placement method aliases
IonMethod = Literal["AN", "SLTCAP"]


@dataclass
class SolvationConfig:
    """Configuration for solvation.

    Attributes:
        input_file: Path to input structure (without extension).
        output_dir: Output directory.
        crystal_type: Crystal box type.
        padding: Padding around molecule in Angstroms.
        salt_concentration: Salt concentration in M.
        positive_ion: Positive ion type (e.g., "POT", "SOD").
        negative_ion: Negative ion type (e.g., "CLA").
        temperature: Temperature in Kelvin.
        skip_ions: If True, skip ion placement.
        ion_method: Ion placement algorithm ("AN" or "SLTCAP").
        min_ion_distance: Minimum distance between ions in Angstroms.
        toppar_dir: Path to topology directory.
        topology_files: List of topology files to load.
    """

    input_file: str | Path
    output_dir: str | Path = "solvated"
    crystal_type: CrystalType = "OCTAHEDRAL"
    padding: float = 10.0
    salt_concentration: float = 0.10
    positive_ion: str = "POT"
    negative_ion: str = "CLA"
    temperature: float = 298.15
    skip_ions: bool = False
    ion_method: IonMethod = "SLTCAP"
    min_ion_distance: float = 5.0
    toppar_dir: Path | None = None
    topology_files: list[str] = field(default_factory=lambda: [
        "top_all36_prot.rtf",
        "par_all36m_prot.prm",
        "toppar_water_ions.str",
        "top_all36_na.rtf",
        "par_all36_na.prm",
        "top_all36_cgenff.rtf",
        "par_all36_cgenff.prm",
    ])
    extra_files: list[str | Path] = field(default_factory=list)
    """Additional topology/parameter files as absolute paths (for custom ligands)."""


def water_density(temp: float) -> float:
    """Calculate water density at a given temperature.

    Uses polynomial fit to experimental data.

    Args:
        temp: Temperature in Kelvin.

    Returns:
        Water density in g/cm³.
    """
    temp_c = temp - 273.15  # Convert to Celsius
    density = (
        999.8395
        + 6.7914e-2 * temp_c
        - 9.0894e-3 * temp_c**2
        + 1.0171e-4 * temp_c**3
        - 1.2846e-6 * temp_c**4
        + 1.1592e-8 * temp_c**5
        - 5.0125e-11 * temp_c**6
    )
    return density


def get_box_parameters(
    crystal_type: CrystalType, stats: dict, padding: float
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Calculate box parameters for a given crystal type.

    Args:
        crystal_type: Crystal box type.
        stats: Coordinate statistics dict with xmin, xmax, ymin, ymax, zmin, zmax.
        padding: Padding around molecule in Angstroms.

    Returns:
        Tuple of (A, B, C, Alpha, Beta, Gamma, BoxSizeX, BoxSizeY, BoxSizeZ).
    """
    xmax, xmin = stats["xmax"], stats["xmin"]
    ymax, ymin = stats["ymax"], stats["ymin"]
    zmax, zmin = stats["zmax"], stats["zmin"]

    Xinit = int((xmax - xmin) + 2 * padding) + 1
    Yinit = int((ymax - ymin) + 2 * padding) + 1
    Zinit = int((zmax - zmin) + 2 * padding) + 1

    crystal_type = crystal_type.upper()

    if crystal_type == "CUBIC":
        A = B = C = max(Xinit, Yinit, Zinit)
        Alpha = Beta = Gamma = 90.0
        BoxSizeX = BoxSizeY = BoxSizeZ = A * 1.1

    elif crystal_type == "TETRAGONAL":
        A = B = max(Xinit, Yinit)
        C = Zinit
        Alpha = Beta = Gamma = 90.0
        BoxSizeX = BoxSizeY = A * 1.1
        BoxSizeZ = C * 1.1

    elif crystal_type == "ORTHORHOMBIC":
        A, B, C = Xinit, Yinit, Zinit
        Alpha = Beta = Gamma = 90.0
        BoxSizeX, BoxSizeY, BoxSizeZ = A * 1.1, B * 1.1, C * 1.1

    elif crystal_type == "MONOCLINIC":
        A, B, C = Xinit, Yinit, Zinit
        Alpha = Gamma = 90.0
        Beta = 70.0
        BoxSizeX, BoxSizeY, BoxSizeZ = A * 1.1, B * 1.1, C * 1.1

    elif crystal_type == "TRICLINIC":
        A, B, C = Xinit, Yinit, Zinit
        Alpha, Beta, Gamma = 60.0, 70.0, 80.0
        BoxSizeX, BoxSizeY, BoxSizeZ = A * 1.1, B * 1.1, C * 1.1

    elif crystal_type == "HEXAGONAL":
        A = B = max(Xinit, Yinit)
        C = Zinit
        Alpha = Beta = 90.0
        Gamma = 120.0
        BoxSizeX = BoxSizeY = A * 1.1
        BoxSizeZ = C * 1.1

    elif crystal_type == "RHOMBOHEDRAL":
        A = B = C = max(Xinit, Yinit, Zinit)
        Alpha = Beta = Gamma = 67.0
        BoxSizeX = BoxSizeY = BoxSizeZ = A * 1.1

    elif crystal_type == "OCTAHEDRAL":
        A = B = C = max(Xinit, Yinit, Zinit)
        Alpha = Beta = Gamma = 109.4712206344907
        BoxSizeX = BoxSizeY = BoxSizeZ = A * 1.1

    elif crystal_type == "RHDO":
        A = B = C = max(Xinit, Yinit, Zinit)
        Alpha = Gamma = 60.0
        Beta = 90.0
        BoxSizeX = BoxSizeY = BoxSizeZ = A * 1.1

    else:
        raise ValueError(f"Invalid crystal_type: {crystal_type}")

    BoxSizeX = round(BoxSizeX, 3)
    BoxSizeY = round(BoxSizeY, 3)
    BoxSizeZ = round(BoxSizeZ, 3)

    return A, B, C, Alpha, Beta, Gamma, BoxSizeX, BoxSizeY, BoxSizeZ


def _read_topology_files(config: SolvationConfig, verbose: bool = True) -> None:
    """Load CHARMM topology and parameter files.

    Args:
        config: Solvation configuration.
        verbose: If False, suppress CHARMM output.
    """
    toppar_dir = config.toppar_dir or TOPPAR_DIR

    if not verbose:
        lingo.charmm_script("prnlev 0")

    # Suppress warnings during topology loading (e.g., duplicate definitions)
    settings.set_bomb_level(-1)

    prm_files = [f for f in config.topology_files if f.endswith(".prm")]
    rtf_files = [f for f in config.topology_files if f.endswith(".rtf")]
    str_files = [f for f in config.topology_files if f.endswith(".str")]

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

    # Load extra files (absolute paths for custom ligands)
    for extra_file in config.extra_files:
        extra_path = Path(extra_file)
        if extra_path.suffix == ".rtf":
            read.rtf(str(extra_path), append=True)
        elif extra_path.suffix == ".prm":
            read.prm(str(extra_path), flex=True, append=True)
        elif extra_path.suffix == ".str":
            lingo.charmm_script(f"stream {extra_path}")

    # Restore bomb level
    settings.set_bomb_level(0)

    if not verbose:
        lingo.charmm_script("prnlev 5")


def _check_gpu() -> bool:
    """Check if GPU is available via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def solvate_system(config: SolvationConfig) -> Path:
    """Solvate a molecular system in a water box.

    This function creates a solvated system with optional ions using
    pyCHARMM. The process includes:
    1. Reading the input structure
    2. Creating a water box with the specified crystal type
    3. Removing overlapping waters
    4. Placing ions (if enabled)
    5. Energy minimization
    6. Writing output files

    Args:
        config: Solvation configuration.

    Returns:
        Path to the output directory containing solvated files.

    Output files:
        - solvated.crd, solvated.psf, solvated.pdb: Full solvated system
        - waterbox.crd, waterbox.psf: Water box only
        - molecule.crd, molecule.psf: Molecule only
        - box.dat: Box parameters
    """
    # Setup
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = str(config.input_file)
    toppar_dir = config.toppar_dir or TOPPAR_DIR

    gpu_available = _check_gpu()

    # Load topology
    _read_topology_files(config)
    lingo.charmm_script("IOFOrmat EXTEnded")

    if gpu_available:
        lingo.charmm_script("blade on")

    # Read input structure
    read.psf_card(f"{input_file}.psf")
    if Path(f"{input_file}.crd").exists():
        read.coor_card(f"{input_file}.crd")
    else:
        read.pdb(f"{input_file}.pdb", resid=True)
        write.coor_card(f"{input_file}.crd")

    # Calculate molecule charge
    charge = sum(psf.get_charges())

    # Orient structure
    coor.orient()

    # Calculate box size
    stats = coor.stat()
    A, B, C, Alpha, Beta, Gamma, BoxSizeX, BoxSizeY, BoxSizeZ = get_box_parameters(
        config.crystal_type, stats, config.padding
    )
    print(f"Box size: {BoxSizeX} {BoxSizeY} {BoxSizeZ}")
    print(f"Molecule size: {A} {B} {C}")

    xcen = ycen = zcen = 0.0

    # Delete atoms for water box creation
    psf.delete_atoms(pycharmm.SelectAtoms().all_atoms())

    # Water box dimensions
    L = 18.8560  # TIP3P water box edge length
    Xnum = int(BoxSizeX / L) + 1
    Ynum = int(BoxSizeY / L) + 1
    Znum = int(BoxSizeZ / L) + 1

    # Create initial water segment
    lingo.charmm_script("read sequ TIP3 216")
    gen.new_segment(
        seg_name="W000",
        first_patch="NONE",
        last_patch="NONE",
        setup_ic=True,
        noangle=True,
        nodihedral=True,
    )
    read.coor_card(str(toppar_dir / "tip216.crd"))

    # Translate water box
    lingo.charmm_script(f"coor translate xdir 1 dist {L/2}")
    lingo.charmm_script(f"coor translate ydir 1 dist {L/2}")
    lingo.charmm_script(f"coor translate zdir 1 dist {L/2}")

    # Build XY plane water box
    for J2 in range(1, Ynum + 1):
        for J1 in range(1, Xnum + 1):
            wsegid = str((J2 - 1) * Xnum + J1)
            lingo.charmm_script("read sequ TIP3 216")
            gen.new_segment(
                seg_name=f"W{wsegid}",
                first_patch="NONE",
                last_patch="NONE",
                setup_ic=True,
                noangle=True,
                nodihedral=True,
            )
            lingo.charmm_script(f"coor duplicate sele segid W000 end sele segid W{wsegid} end")
            X = L * (J1 - 1)
            Y = L * (J2 - 1)
            pycharmm.charmm_script(f"coor translate xdir {X} ydir {Y} sele segid W{wsegid} end")
            if J1 == 1 and J2 == 1:
                lingo.charmm_script(f"RENAme segid SOLV sele segid W{wsegid} end")
            else:
                lingo.charmm_script(f"JOIN SOLV W{wsegid} RENUmber")

    if psf.get_natom == 0:
        raise RuntimeError("No waters in the box")

    psf.delete_atoms(pycharmm.SelectAtoms(seg_id="W000"))

    # Build Z direction
    N_water = psf.get_nres()
    for J3 in range(2, Znum + 1):
        Z = L * (J3 - 1)
        lingo.charmm_script(f"read sequ TIP3 {N_water}")
        gen.new_segment(
            seg_name=f"W{J3}",
            first_patch="NONE",
            last_patch="NONE",
            setup_ic=True,
            noangle=True,
            nodihedral=True,
        )
        lingo.charmm_script(f"coor duplicate sele segid SOLV .and. resid 1:{N_water} end sele segid W{J3} end")
        lingo.charmm_script(f"coor translate zdir {Z} sele segid W{J3} end")
        lingo.charmm_script(f"JOIN SOLV W{J3} RENUmber")

    # Trim Z direction
    lingo.charmm_script(f"""
        delete atom sele .byres. ( ( type OH2 ) .and. -
            ( prop Z .gt. {BoxSizeZ} ) ) end
    """)

    coor.orient(noro=True)

    # Shape the box
    lingo.charmm_script(f"coor convert symmetric aligned {BoxSizeX} {BoxSizeY} {BoxSizeZ} {Alpha} {Beta} {Gamma}")
    lingo.charmm_script("coor copy comp")

    lingo.charmm_script(f"crystal define {config.crystal_type} {BoxSizeX} {BoxSizeY} {BoxSizeZ} {Alpha} {Beta} {Gamma} {xcen} {ycen} {zcen}")
    crystal.build(2)

    image.setup_residue(0, 0, 0, "TIP3")

    lingo.charmm_script("nbonds ctonnb 2.0 ctofnb 3.0 cutnb 3.0 cutim 3.0 wmin 0.01 fswitch vswitch")
    lingo.charmm_script("crystal free")
    lingo.charmm_script("coor diff comp")

    lingo.charmm_script("""
        delete atom sele .byres. ( ( prop Xcomp .ne. 0 ) .or. -
            ( prop Ycomp .ne. 0 ) .or. -
            ( prop Zcomp .ne. 0 ) ) end
    """)

    # Save water box
    write.coor_card(str(output_dir / "waterbox.crd"))
    write.psf_card(str(output_dir / "waterbox.psf"))
    psf.delete_atoms(pycharmm.SelectAtoms().all_atoms())

    # Save box parameters
    with open(output_dir / "box.dat", "w", encoding="utf-8") as f:
        f.write(f"{config.crystal_type}\n")
        f.write(f"{BoxSizeX} {BoxSizeY} {BoxSizeZ}\n")
        f.write(f"{Alpha} {Beta} {Gamma}\n")

    # Combine molecule with water box
    read.psf_card(f"{input_file}.psf")
    read.coor_card(f"{input_file}.crd")
    coor.orient()
    N_molecule = psf.get_nres()
    write.psf_card(
        str(output_dir / "molecule.psf"),
        title="Molecule with Minimization (part with waterbox.*)",
        select="segid MOL end",
    )
    read.psf_card(str(output_dir / "waterbox.psf"), append=True)
    read.coor_card(str(output_dir / "waterbox.crd"), append=True)

    # Remove overlapping waters
    molecule = pycharmm.SelectAtoms().by_seg_id("SOLV").__invert__()
    water = pycharmm.SelectAtoms().by_seg_id("SOLV") & pycharmm.SelectAtoms().by_atom_type("OH2")
    water = (water & molecule.around(2.6)).whole_residues()
    psf.delete_atoms(water)

    N_water = psf.get_nres() - N_molecule

    # Ion placement
    if not config.skip_ions:
        _place_ions(config, N_water, charge, A, B, C, Alpha, Beta, Gamma, xcen, ycen, zcen)

    lingo.charmm_script("JOIN SOLV RENUmber")
    lingo.charmm_script("nbonds ctonnb 10 ctofnb 12 cutnb 14 cutim 14 wmin 1.0 fswitch vswitch")
    lingo.charmm_script(" define MOL sele .not. (segid SOLV .or. segid IONS) end")

    # Minimization
    energy.show()
    energy_ = energy.get_total()

    for force in [100, 50, 25, 5]:
        pycharmm.charmm_script(f"cons harm force {force} sele MOL .and. (.not. hydrogen) end")
        minimize.run_sd(nstep=50)
        minimize.run_abnr(nstep=100)
        lingo.charmm_script("cons harm clear")

    minimize.run_sd(nstep=1000)
    new_energy = energy.get_total()
    delta_energy = energy_ - new_energy

    print(f"Energy before minimization: {energy_} kcal/mol")
    print(f"Energy after minimization: {new_energy} kcal/mol")
    print(f"Energy change after minimization: {delta_energy} kcal/mol")

    # Write output files
    write.coor_card(str(output_dir / "solvated.crd"))
    write.psf_card(str(output_dir / "solvated.psf"))
    write.coor_pdb(str(output_dir / "solvated.pdb"))
    write.coor_card(
        str(output_dir / "molecule.crd"),
        title="Molecule with Minimization (part with waterbox.*)",
        select=".not. (segid SOLV .or. segid IONS) end",
    )
    # Suppress non-integer charge warnings during cleanup splits
    # (solute or waterbox alone may have fractional charge)
    settings.set_bomb_level(-1)
    psf.delete_atoms(molecule)
    write.coor_card(
        str(output_dir / "waterbox.crd"),
        title=f"{config.crystal_type} Waterbox with box size {BoxSizeX}:{BoxSizeY}:{BoxSizeZ}",
        select="segid SOLV .or. segid IONS end",
    )
    write.psf_card(
        str(output_dir / "waterbox.psf"),
        title=f"{config.crystal_type} Waterbox with box size {BoxSizeX}:{BoxSizeY}:{BoxSizeZ}",
        select="segid SOLV .or. segid IONS end",
    )
    settings.set_bomb_level(0)

    print("Solvation completed")
    return output_dir


def _place_ions(
    config: SolvationConfig,
    N_water: int,
    charge: float,
    A: float, B: float, C: float,
    Alpha: float, Beta: float, Gamma: float,
    xcen: float, ycen: float, zcen: float,
) -> None:
    """Place ions in the system.

    Args:
        config: Solvation configuration.
        N_water: Number of water molecules.
        charge: Total system charge.
        A, B, C: Box dimensions.
        Alpha, Beta, Gamma: Box angles.
        xcen, ycen, zcen: Box center coordinates.
    """
    print(f"Number of waters: {N_water}")
    print(f"Total of solute charge: {charge}")

    lingo.charmm_script("crystal free")
    lingo.charmm_script(f"crystal define {config.crystal_type} {A} {B} {C} {Alpha} {Beta} {Gamma} {xcen} {ycen} {zcen}")
    crystal.build(config.padding)

    # Calculate number of ions
    M_water = 18.01528  # g/mol
    Rho_water = water_density(config.temperature)
    print(f"Water density: {Rho_water} g/cm^3")

    N_0 = (N_water * M_water * config.salt_concentration) / Rho_water

    if config.ion_method == "AN":
        N_pos = round(N_0)
        N_neg = round(N_0 + charge)
    elif config.ion_method == "SLTCAP":
        factor = (1 + (charge / (2 * N_0))) ** 0.5
        N_pos = round(N_0 * factor - charge / 2)
        N_neg = round(N_0 * factor + charge / 2)
    else:
        raise ValueError(f"Unknown ion_method: {config.ion_method}")

    N_ion = N_pos + N_neg
    print(f"Ion placement algorithm: {config.ion_method}")
    print(f"Placing {N_pos} {config.positive_ion} and {N_neg} {config.negative_ion} ions")

    if N_ion == 0:
        return

    # Generate ions segment
    read.sequence_string(f"{config.positive_ion} " * N_pos + f"{config.negative_ion} " * N_neg)
    gen.new_segment(
        seg_name="IONS",
        first_patch="NONE",
        last_patch="NONE",
        setup_ic=True,
        noangle=True,
        nodihedral=True,
    )

    # Place ions
    for i in range(1, N_ion + 1):
        search = True
        while search:
            all_crds_df = coor.get_positions()

            sel_all_for_psf = pycharmm.SelectAtoms().all_atoms()
            if not sel_all_for_psf._atom_indexes:
                print("Warning: No atoms found in PSF for ion placement. Stopping ion placement.")
                return

            current_psf_df = pd.DataFrame({
                "atom_index": sel_all_for_psf._atom_indexes,
                "res_name": sel_all_for_psf._res_names,
                "res_id": sel_all_for_psf._res_ids,
                "atom_type": sel_all_for_psf._atom_types,
                "seg_id": sel_all_for_psf._seg_ids,
            })

            system_df = pd.merge(current_psf_df, all_crds_df, left_on="atom_index", right_index=True)

            water_oh2_for_resids = system_df[
                (system_df["seg_id"] == "SOLV")
                & (system_df["res_name"] == "TIP3")
                & (system_df["atom_type"] == "OH2")
            ]

            if water_oh2_for_resids.empty:
                print("Warning: No TIP3 OH2 atoms found in SOLV segment for ion placement.")
                return

            water_resids = water_oh2_for_resids["res_id"].unique()
            if not water_resids.size:
                print("Warning: No water residues available to replace with ions.")
                return

            # Molecule coordinates
            molecule_seg_ids = [
                sid for sid in system_df["seg_id"].unique() if sid not in ["SOLV", "IONS"]
            ]
            if molecule_seg_ids:
                molecule_sel_df = system_df[system_df["seg_id"].isin(molecule_seg_ids)]
                molecule_coords_np = molecule_sel_df[["x", "y", "z"]].values
            else:
                molecule_coords_np = np.empty((0, 3))

            # Existing ion coordinates
            ion_sel_df = system_df[system_df["seg_id"] == "IONS"]
            if not ion_sel_df.empty:
                other_ions_sel_df = ion_sel_df[ion_sel_df["res_id"] != i]
                if not other_ions_sel_df.empty:
                    ion_coords_np = other_ions_sel_df[["x", "y", "z"]].values
                else:
                    ion_coords_np = np.empty((0, 3))
            else:
                ion_coords_np = np.empty((0, 3))

            random_water_res_id = random.choice(water_resids)
            chosen_water_atoms_df = system_df[
                (system_df["seg_id"] == "SOLV") & (system_df["res_id"] == random_water_res_id)
            ]

            if chosen_water_atoms_df.empty:
                continue

            target_ion_pos_xyz = chosen_water_atoms_df[["x", "y", "z"]].mean().values

            # Distance checks
            if molecule_coords_np.shape[0] > 0:
                distances_to_mol = np.linalg.norm(molecule_coords_np - target_ion_pos_xyz, axis=1)
                if np.any(distances_to_mol <= 2.6):
                    continue

            if ion_coords_np.shape[0] > 0:
                distances_to_ions = np.linalg.norm(ion_coords_np - target_ion_pos_xyz, axis=1)
                if np.any(distances_to_ions < config.min_ion_distance):
                    continue

            # Place ion
            x_place, y_place, z_place = target_ion_pos_xyz
            lingo.charmm_script(
                f"coor set xdir {x_place:.4f} ydir {y_place:.4f} zdir {z_place:.4f} "
                f"sele segid IONS .and. resid {i} end"
            )

            # Delete replaced water
            lingo.charmm_script(f"dele atom sele resid {random_water_res_id} .and. segid SOLV end")

            search = False

        if search:
            print(f"Could not place ion {i}. Stopping further ion placement.")
            break
