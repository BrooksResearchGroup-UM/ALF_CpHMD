"""
Block and Restraint File Generator for CpHMD Production Runs.

This module generates the MSLD block commands and restraint files needed
to run production CpHMD simulations after ALF training is complete.

Usage:
    # Programmatic
    from cphmd.core.generate_block import BlockGeneratorConfig, generate_block_files
    result = generate_block_files(BlockGeneratorConfig(input_folder="my_system"))

    # CLI
    python -m cphmd.core.generate_block -i my_system --restrain-type SCAT
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class BlockGeneratorConfig:
    """Configuration for block/restraint file generation.

    Attributes:
        input_folder: Path to simulation folder with prep/patches.dat
        restrain_type: Restraint method: 'SCAT' or 'NOE'
        include_hydrogens: Include hydrogen atoms in restraints
        electrostatics: PME method: 'pmeex', 'pmeon', 'pmenn'
        temperature: Simulation temperature in Kelvin
        variables_dir: Directory containing var-{resname}.inp files
    """

    input_folder: str
    restrain_type: str = "SCAT"
    include_hydrogens: bool = False
    electrostatics: str = "pmeex"
    temperature: float = 298.15
    variables_dir: str = "variables"
    lambda_mass: float = 12.0    # Lambda mass in amu·Å² (BIMLAM)
    lambda_fbeta: float = 5.0    # Lambda Langevin friction in ps⁻¹ (BIBLAM)

    def __post_init__(self):
        self.input_folder = Path(self.input_folder)
        self.variables_dir = Path(self.variables_dir)

        # Validate patches.dat exists
        patches_file = self.input_folder / "prep" / "patches.dat"
        if not patches_file.exists():
            raise FileNotFoundError(f"patches.dat not found: {patches_file}")


@dataclass
class BlockGeneratorResult:
    """Results from block/restraint file generation.

    Attributes:
        block_file: Path to generated block.str
        restraint_file: Path to generated restrains.str
        n_sites: Number of titratable sites
        n_blocks: Total number of MSLD blocks
    """

    block_file: Path
    restraint_file: Path
    n_sites: int
    n_blocks: int


def _load_patches(input_folder: Path) -> pd.DataFrame:
    """Load and parse patches.dat file."""
    patches_file = input_folder / "prep" / "patches.dat"
    patch_info = pd.read_csv(patches_file, sep=",")

    # Extract site and sub from SELECT (e.g., s1s2 -> site=1, sub=2)
    patch_info[["site", "sub"]] = patch_info["SELECT"].str.extract(r"s(\d+)s(\d+)")
    patch_info["site"] = patch_info["site"].astype(int)
    patch_info["sub"] = patch_info["sub"].astype(int)

    return patch_info


def _load_variables(resname: str, variables_dir: Path) -> dict[str, float]:
    """Load variables from var-{resname}.inp or .txt file."""
    resname_lower = resname.lower()

    # Try .inp format first
    var_file = variables_dir / f"var-{resname_lower}.inp"
    if var_file.exists():
        variables = {}
        with open(var_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("set"):
                    # Parse: set varname = value
                    parts = line.replace("set", "").strip().split("=")
                    if len(parts) == 2:
                        var_name = parts[0].strip()
                        try:
                            var_value = float(parts[1].strip())
                        except ValueError:
                            var_value = 0.0
                        variables[var_name] = var_value
        return variables

    # Try .txt format
    var_file = variables_dir / f"var-{resname_lower}.txt"
    if var_file.exists():
        df = pd.read_csv(
            var_file, sep=",", header=None, names=["variable", "value"]
        )
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        # Filter to numeric values only (skip sysname, nnodes, etc.)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        return dict(zip(df["variable"], df["value"]))

    raise FileNotFoundError(f"No variable file found for {resname} in {variables_dir}")


def _generate_block_str(
    patch_info: pd.DataFrame,
    variables_dir: Path,
    temperature: float,
    electrostatics: str,
    fnex: float = 5.5,
    lambda_mass: float = 12.0,
    lambda_fbeta: float = 5.0,
) -> str:
    """Generate MSLD BLOCK command string."""
    from .bias_constants import derive_bias_constants
    constants = derive_bias_constants(fnex)
    lines = []

    # Initialize BLOCK components
    block_count = 1
    call_lines = []
    excl_lines = []
    ldin_lines = []
    msld_sites = []
    ldbv_lines = []

    # Initial LDIN for environment block
    ldin_lines.append(
        f"LDIN {1:<6} {1:<6} {0.0:<6} {lambda_mass:<7} {0.0:>8} {lambda_fbeta:>7} {'NONE':>12}"
    )

    site_counter = 0
    for site, group in patch_info.groupby("site", sort=False):
        site_counter += 1
        resname = group["PATCH"].iloc[0][:3]
        segid = group["SEGID"].iloc[0]
        resid = group["RESID"].iloc[0]

        # Load variables for this residue type
        try:
            var_df = _load_variables(resname, variables_dir)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            var_df = {}

        num_patches = len(group)
        site_start_block = block_count + 1

        # Add header comments
        ldin_lines.append(
            f"!---------------------------------------------------------------\n"
            f"! Lambda Initialization for {segid} {resid} {resname} | SITE {site}\n"
            f"!---------------------------------------------------------------"
        )

        call_lines.append(
            f"!---------------------------------------------------------------\n"
            f"! CALL selection for {segid} {resid} {resname} | SITE {site}\n"
            f"!---------------------------------------------------------------"
        )

        # Process each patch in the site
        for patch_idx, (idx, row) in enumerate(group.iterrows()):
            block_count += 1
            patch = row["PATCH"]
            utag = row.get("TAG", "NONE")

            # CALL statement
            call_lines.append(
                f"CALL {block_count:>4} SELEct segid {segid:>4} .and. "
                f"resid {resid:>4} .and. resname {patch:>4} end"
            )

            # MSLD site assignment
            msld_sites.append(str(site_counter))

            # LDIN statement
            patch_num = patch_idx + 1
            lam_key = f"lams1s{patch_num}"
            lam_bias = var_df.get(lam_key, 0.0)

            if patch_idx == 0:
                ldin_lines.append(
                    f"LDIN {block_count:<6} {round(1/num_patches, 2):<6} {0.0:<6} "
                    f"{lambda_mass:<7} {lam_bias:>8} {lambda_fbeta:>7} {'NONE':>12}"
                )
            else:
                ldin_lines.append(
                    f"LDIN {block_count:<6} {round(1/num_patches, 2):<6} {0.0:<6} "
                    f"{lambda_mass:<7} {lam_bias:>8} {lambda_fbeta:>7} {utag:>12}"
                )

        # Generate exclusions for this site
        site_blocks = list(range(site_start_block, block_count + 1))
        for b1, b2 in itertools.combinations(site_blocks, 2):
            excl_lines.append(f"adexcl {b1} {b2}")

        # Generate LDBV for this site
        ldbv_header = (
            f"!---------------------------------------------------------------\n"
            f"! Biasing Potential for {segid} {resid} {resname} | SITE {site}\n"
            f"!---------------------------------------------------------------"
        )
        ldbv_lines.append(ldbv_header)

        # Quadratic barriers (type 6) - combinations
        for b1, b2 in itertools.combinations(site_blocks, 2):
            s1 = b1 - site_start_block + 1
            s2 = b2 - site_start_block + 1
            c_key = f"cs1s{s1}s1s{s2}"
            c_val = var_df.get(c_key, 0.0)
            ldbv_lines.append(
                f"LDBV {{idx:<3}} {b1:>4} {b2:>4} {6:>4} {0.0:>8} {c_val:>10} {0:>5}"
            )

        ldbv_lines.append("")

        # Endpoint potentials (type 8) - permutations
        for b1, b2 in itertools.permutations(site_blocks, 2):
            s1 = b1 - site_start_block + 1
            s2 = b2 - site_start_block + 1
            s_key = f"ss1s{s1}s1s{s2}"
            s_val = var_df.get(s_key, 0.0)
            ldbv_lines.append(
                f"LDBV {{idx:<3}} {b1:>4} {b2:>4} {8:>4} {constants.chi_offset:>8.5f} {s_val:>10} {0:>5}"
            )

        ldbv_lines.append("")

        # Skew potentials (type 10) - permutations
        for b1, b2 in itertools.permutations(site_blocks, 2):
            s1 = b1 - site_start_block + 1
            s2 = b2 - site_start_block + 1
            x_key = f"xs1s{s1}s1s{s2}"
            x_val = var_df.get(x_key, 0.0)
            ldbv_lines.append(
                f"LDBV {{idx:<3}} {b1:>4} {b2:>4} {10:>4} {-constants.omega_decay:>8} {x_val:>10} {0:>5}"
            )

        ldbv_lines.append("")

    # Count LDBV entries and assign indices
    ldbv_count = sum(1 for line in ldbv_lines if line.startswith("LDBV"))
    ldbv_idx = 0
    indexed_ldbv = []
    for line in ldbv_lines:
        if line.startswith("LDBV"):
            ldbv_idx += 1
            indexed_ldbv.append(line.format(idx=ldbv_idx))
        else:
            indexed_ldbv.append(line)

    # Assemble the full BLOCK string
    output = []
    output.append("!---------------------------------------------------------------")
    output.append("! Set up l-dynamics by setting BLOCK parameters")
    output.append("!---------------------------------------------------------------\n")
    output.append(f"BLOCK {block_count} NREP @nrep\n")

    # CALL statements
    output.extend(call_lines)
    output.append("")

    # Exclusions
    output.append("!---------------------------------------------------------------")
    output.append("! l-exclusions")
    output.append("!---------------------------------------------------------------\n")
    output.extend(excl_lines)
    output.append("\n")

    # QLDM, LANG, PHMD, SOFT
    output.append("!------------------------------------------")
    output.append("!QLDM turns on lambda-dynamics option")
    output.append("!------------------------------------------\n")
    output.append("qldm theta\n")

    output.append("!------------------------------------------")
    output.append("!LANGEVIN turns on the langevin heatbath")
    output.append("!------------------------------------------\n")
    output.append(f"lang temp {temperature}\n")

    output.append("!------------------------------------------")
    output.append("!Setup initial pH from PHMD")
    output.append("!------------------------------------------\n")
    output.append("phmd pH 7\n")

    output.append("!------------------------------------------")
    output.append("!Soft Core Potential")
    output.append("!------------------------------------------\n")
    output.append("soft on\n")

    # LDIN statements
    output.append("!------------------------------------------")
    output.append("! lambda-dynamics energy constraints (from ALF)")
    output.append("!------------------------------------------\n")
    output.extend(ldin_lines)
    output.append("")

    # RMLA, MSLD, MSMA
    output.append("!------------------------------------------")
    output.append("! All bond/angle/dihe terms treated at full str (no scaling),")
    output.append("! prevent unphysical results")
    output.append("!------------------------------------------\n")
    output.append("rmla bond thet impr")

    output.append("!------------------------------------------")
    output.append("! Selects MSLD, the numbers assign each block to the specified site on the core")
    output.append("!------------------------------------------\n")

    # Format MSLD line with site assignments
    msld_line = "msld 0"
    for i, site_num in enumerate(msld_sites):
        msld_line += f" {site_num}"
        if (i + 1) % 10 == 0 and i < len(msld_sites) - 1:
            msld_line += " -\n"
    msld_line += " -\nfnex 5.5\n"
    output.append(msld_line)

    output.append("!------------------------------------------")
    output.append("! Constructs the interaction matrix and assigns lambda & theta values for each block")
    output.append("!------------------------------------------\n")
    output.append("msma\n")

    # PME electrostatics
    output.append("!------------------------------------------")
    output.append("! PME electrostatics")
    output.append("!------------------------------------------\n")
    if electrostatics in ("pmeon", "pme_on"):
        output.append("pmel on\n")
    elif electrostatics in ("pmenn", "pme_nn"):
        output.append("pmel nn\n")
    else:
        output.append("pmel ex\n")

    # LDBI and LDBV
    output.append("!------------------------------------------")
    output.append("! Enables bias potential on lambda variables")
    output.append("! INDEX, I,J(Bias between I & J)), CLASS, REF, CFORCE, NPOWER, Identity flag")
    output.append("! CLASS: Functional Form of bias, REF: Cut off for physical lambda states")
    output.append("! NPOWER: Power of functional form, CFORCE: kbias on Fvar, residue specific value")
    output.append("!------------------------------------------\n")
    output.append(f"LDBI {ldbv_count}")
    output.extend(indexed_ldbv)
    output.append("END")

    return "\n".join(output)


def _generate_restraints_str(
    patch_info: pd.DataFrame,
    restrain_type: str,
    include_hydrogens: bool,
    temperature: float,
) -> str:
    """Generate restraint commands string."""
    lines = []

    if restrain_type.upper() == "SCAT":
        lines.append("BLOCK")
        lines.append("scat on")
        lines.append(f"scat k {temperature}")

        for site in patch_info["site"].unique():
            site_data = patch_info[patch_info["site"] == site]

            # Get all unique atoms for this site
            atoms = set()
            for atoms_str in site_data["ATOMS"]:
                atoms.update(atoms_str.split())

            h_atoms = [a for a in atoms if a.startswith("H")]
            heavy_atoms = [a for a in atoms if not a.startswith("H")]

            # Build inline selection from SEGID, RESID, and PATCH columns
            segid = site_data["SEGID"].iloc[0]
            resid = site_data["RESID"].iloc[0]
            resnames = site_data["PATCH"].tolist()
            resname_clause = " .or. ".join(f"resname {r}" for r in resnames)

            for atom in heavy_atoms:
                lines.append(f"cats SELE type {atom} .and. segid {segid} .and. resid {resid} .and. ({resname_clause}) END")

            if include_hydrogens:
                for atom in h_atoms:
                    lines.append(f"cats SELE type {atom} .and. segid {segid} .and. resid {resid} .and. ({resname_clause}) END")

        lines.append("END")

    elif restrain_type.upper() == "NOE":
        lines.append("! Small minimization in case of atoms at same position")
        lines.append("cons fix sele .not. (resn %%%%) .or. resn TIP3 end")
        lines.append("mini sd nstep 2 step 0.005")
        lines.append("! NOE restraints")
        lines.append("NOE")

        group_idx = 1
        for site, group in patch_info.groupby("site", sort=False):
            segid = group["SEGID"].iloc[0]
            resid = group["RESID"].iloc[0]
            resname = group["PATCH"].iloc[0][:3]

            # Get repeated atoms
            atom_counts = group["ATOMS"].str.split().explode().value_counts()
            repeats = atom_counts[atom_counts > 1].index.tolist()

            if not repeats:
                continue

            lines.append(
                f"!---------------------------------------------------------------\n"
                f"! Restraints for {segid} {resname} {resid}, SITE {site}, GROUP {group_idx}\n"
                f"!---------------------------------------------------------------"
            )
            group_idx += 1

            for repeat_atom in repeats:
                if not include_hydrogens and repeat_atom.startswith("H"):
                    continue

                atom_patches = group[
                    group["ATOMS"].str.contains(repeat_atom, na=False)
                ]

                if len(atom_patches) < 2:
                    continue

                for i1, i2 in itertools.combinations(atom_patches.index, 2):
                    patch1 = atom_patches.loc[i1, "PATCH"]
                    patch2 = atom_patches.loc[i2, "PATCH"]

                    lines.append(
                        f"assign sele segid {segid} .and. resid {resid} .and. "
                        f"resn {patch1} .and. type {repeat_atom} end "
                        f"sele segid {segid} .and. resid {resid} .and. "
                        f"resn {patch2} .and. type {repeat_atom} end -"
                    )
                    lines.append(
                        "kmin 100.0 rmin 0.0 kmax 100.0 rmax 0.0 fmax 2.0 rswitch 99999 sexp 1.0"
                    )

        lines.append("END")
        lines.append("cons fix sele none end")

    return "\n".join(lines)


def generate_block_files(config: BlockGeneratorConfig) -> BlockGeneratorResult:
    """Generate block.str and restrains.str files.

    Args:
        config: BlockGeneratorConfig with generation parameters

    Returns:
        BlockGeneratorResult with file paths and statistics
    """
    patch_info = _load_patches(config.input_folder)

    n_sites = patch_info["site"].nunique()
    n_blocks = len(patch_info) + 1  # +1 for environment

    print(f"\nGenerating block files for {config.input_folder}")
    print("=" * 60)
    print(f"Sites: {n_sites}, Blocks: {n_blocks}")
    print(f"Restraint type: {config.restrain_type}")
    print(f"Include hydrogens: {config.include_hydrogens}")

    # Generate block.str
    block_str = _generate_block_str(
        patch_info,
        config.variables_dir,
        config.temperature,
        config.electrostatics,
        lambda_mass=config.lambda_mass,
        lambda_fbeta=config.lambda_fbeta,
    )

    block_file = config.input_folder / "prep" / "block.str"
    with open(block_file, "w") as f:
        f.write(block_str)
    print(f"\nBlock file: {block_file}")

    # Generate restrains.str
    restraint_str = _generate_restraints_str(
        patch_info,
        config.restrain_type,
        config.include_hydrogens,
        config.temperature,
    )

    restraint_file = config.input_folder / "prep" / "restrains.str"
    with open(restraint_file, "w") as f:
        f.write(restraint_str)
    print(f"Restraint file: {restraint_file}")

    return BlockGeneratorResult(
        block_file=block_file,
        restraint_file=restraint_file,
        n_sites=n_sites,
        n_blocks=n_blocks,
    )


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate MSLD block and restraint files from patches.dat"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input folder with prep/patches.dat"
    )
    parser.add_argument(
        "--restrain-type",
        choices=["SCAT", "NOE"],
        default="SCAT",
        help="Restraint type (default: SCAT)",
    )
    parser.add_argument(
        "-H", "--hydrogens", action="store_true", help="Include hydrogens in restraints"
    )
    parser.add_argument(
        "-e",
        "--electrostatics",
        default="pmeex",
        choices=["pmeon", "pmenn", "pmeex"],
        help="Electrostatics method (default: pmeex)",
    )
    parser.add_argument(
        "-t", "--temperature", type=float, default=298.15, help="Temperature (K)"
    )
    parser.add_argument(
        "-v", "--variables-dir", default="variables", help="Variables directory"
    )

    args = parser.parse_args()

    try:
        config = BlockGeneratorConfig(
            input_folder=args.input,
            restrain_type=args.restrain_type,
            include_hydrogens=args.hydrogens,
            electrostatics=args.electrostatics,
            temperature=args.temperature,
            variables_dir=args.variables_dir,
        )

        result = generate_block_files(config)
        print(f"\nGenerated {result.n_blocks} blocks for {result.n_sites} sites")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
