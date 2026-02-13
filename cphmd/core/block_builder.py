"""
BLOCK/MSLD command builder for lambda dynamics.

This module generates CHARMM BLOCK commands for Multi-Site Lambda Dynamics (MSLD)
simulations. BLOCK enables alchemical free energy calculations by defining
multiple chemical states that can interconvert during dynamics.

MSLD Setup Components:
1. CALL statements - Define atom selections for each alchemical block
2. ADEXCL statements - Exclude intra-site block interactions
3. QLDM THETA - Enable lambda dynamics with theta variable
4. LANG TEMP - Langevin thermostat for lambda particles
5. PHMD pH - CpHMD pH coupling (when pH is specified)
6. SOFT ON - Enable soft-core potentials for alchemical atoms
7. LDIN statements - Initialize lambda values and biases
8. MSLD/MSMA - Build the site/block interaction matrix
9. PMEL EX - Electrostatic PME exclusions
10. LDBI/LDBV - Variable lambda potentials (barriers, endpoints, skew)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BlockConfig:
    """Configuration for BLOCK command generation.

    Attributes:
        temperature: Simulation temperature in Kelvin
        pH: Target pH (None for non-CpHMD simulations)
        effective_pH: Computed effective pH for PHMD command
        delta_pKa: pH increment between replicas
        use_cphmd: Whether to include PHMD and TAG directives
    """
    temperature: float = 298.15
    pH: float | None = None
    effective_pH: float | None = None
    delta_pKa: float = 0.0
    use_cphmd: bool = True
    initial_lambdas: dict | None = None  # {site: [dirichlet_alphas]} biased sampling
    lambda_mass: float = 12.0    # Lambda mass in amu·Å² (BIMLAM)
    lambda_fbeta: float = 5.0    # Lambda Langevin friction in ps⁻¹ (BIBLAM)


def read_variable_file(var_file: Path) -> dict[str, float]:
    """Read CHARMM variable file and extract values.

    Args:
        var_file: Path to variables{N}.inp file

    Returns:
        Dictionary mapping variable names to values
    """
    variables = {}

    with open(var_file) as f:
        for line in f:
            if line.strip().startswith("set"):
                # Parse: set varname = value
                line = line.replace("set", "").strip()
                var_name, var_value = line.split("=")
                var_name = var_name.strip()
                var_value = var_value.strip()

                try:
                    val = float(var_value)
                except ValueError:
                    # String value (like system name)
                    variables[var_name] = var_value
                else:
                    # Fail-safe: reject NaN/Inf values that would crash CHARMM
                    if not np.isfinite(val):
                        raise ValueError(
                            f"Variable {var_name} has invalid value: {var_value}. "
                            "This indicates upstream NaN propagation - check WHAM output."
                        )
                    variables[var_name] = val

    return variables


def generate_block_header(n_blocks: int) -> str:
    """Generate BLOCK command header.

    Args:
        n_blocks: Total number of blocks (n_patches + 1 for environment)

    Returns:
        BLOCK header string
    """
    return f"BLOCK {n_blocks}\n\n"


def generate_call_statements(patch_info: pd.DataFrame) -> str:
    """Generate CALL statements for atom selections.

    Each titratable group gets a CALL that references its predefined selection.

    Args:
        patch_info: DataFrame from patches.dat

    Returns:
        CHARMM CALL statements
    """
    lines = [
        "!----------------------------------------",
        "! Set up l-dynamics by setting BLOCK parameters",
        "!----------------------------------------\n",
    ]

    for idx, row in patch_info.iterrows():
        lines.append(f"CALL {idx + 2} SELEct segid {row['SEGID']} .and. resid {row['RESID']} .and. resname {row['PATCH']} END")

    return "\n".join(lines) + "\n\n"


def generate_exclusions(patch_info: pd.DataFrame) -> str:
    """Generate ADEXCL statements for intra-site exclusions.

    Blocks within the same site must be excluded from each other
    to prevent double-counting of interactions.

    Args:
        patch_info: DataFrame from patches.dat

    Returns:
        CHARMM ADEXCL statements
    """
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


def generate_dynamics_setup(config: BlockConfig) -> str:
    """Generate QLDM, LANG, PHMD, and SOFT directives.

    Args:
        config: Block configuration

    Returns:
        CHARMM dynamics setup commands
    """
    lines = [
        "!----------------------------------------",
        "!QLDM turns on lambda-dynamics option",
        "!----------------------------------------\n",
        "QLDM THETa\n",
        "!----------------------------------------",
        "!LANGEVIN turns on the langevin heatbath",
        "!----------------------------------------\n",
        f"LANG TEMP {config.temperature:.2f}\n",
    ]

    # CpHMD pH coupling
    if config.use_cphmd:
        lines.extend([
            "!----------------------------------------",
            "!Setup CpHMD with replica-specific pH",
            "!----------------------------------------\n",
            f"PHMD pH {config.effective_pH:.3f}\n",
        ])
    elif config.pH is not None:
        lines.extend([
            "!----------------------------------------",
            "!CpHMD disabled (delta_pKa=0)",
            "!----------------------------------------\n",
        ])

    lines.extend([
        "!----------------------------------------",
        "!Soft-core potentials",
        "!----------------------------------------\n",
        "SOFT ON\n",
    ])

    return "\n".join(lines)


def generate_ldin_statements(
    patch_info: pd.DataFrame,
    variables: dict[str, float],
    config: BlockConfig,
) -> str:
    """Generate LDIN statements for lambda initialization.

    LDIN format: block# l0 vel mass bias fbeta [TAG]
    - block#: Block index (1=environment, 2+=titratable groups)
    - l0: Initial lambda value
    - vel: Initial velocity (usually 0)
    - mass: Lambda mass (default 12.0 amu·Å²)
    - bias: Fixed bias from ALF (from variables file)
    - fbeta: Lambda Langevin friction coefficient (default 5.0 ps⁻¹)
    - TAG: UPOS/UNEG/NONE with pKa value (for CpHMD)

    Args:
        patch_info: DataFrame from patches.dat
        variables: Variables from ALF variable file
        config: Block configuration

    Returns:
        CHARMM LDIN statements
    """
    lines = [
        "!----------------------------------------",
        "!lambda-dynamics energy constraints (from ALF) AKA fixed bias",
        "!----------------------------------------\n",
    ]

    # Initialize l0 values per site using Dirichlet distribution
    # initial_lambdas dict provides per-site alpha weights (default: all ones)
    patch_info = patch_info.copy()
    patch_info["l0"] = 0.0
    biased_alphas = config.initial_lambdas or {}

    for site in patch_info["site"].unique():
        site_mask = patch_info["site"] == site
        subsites = patch_info.loc[site_mask, "sub"].tolist()

        # Dirichlet alpha: uniform [1,1,...] or biased [1,1,10] for unsampled states
        alpha = np.array(biased_alphas.get(site, np.ones(len(subsites))), dtype=float)
        l0_values = np.random.dirichlet(alpha)
        l0_values = np.round(l0_values, 3)
        l0_values[-1] = np.round(1.0 - np.sum(l0_values[:-1]), 3)

        for sub, l0 in zip(subsites, l0_values):
            mask = (patch_info["site"] == site) & (patch_info["sub"] == sub)
            patch_info.loc[mask, "l0"] = l0

    # Generate LDIN statements
    use_tags = config.use_cphmd
    mass = config.lambda_mass
    fbeta = config.lambda_fbeta

    if use_tags:
        # Environment block
        lines.append(f"LDIN {1:<4} {1:<4} {0.0:<4} {mass:<4} {0.0:<2} {fbeta:<4} {'NONE':<5}")

        # Titratable groups with TAG
        for idx, row in patch_info.iterrows():
            l0 = row["l0"]
            var_key = f"lam{row['SELECT']}"
            bias = variables.get(var_key, 0.0)
            tag = row["TAG"]
            lines.append(f"LDIN {idx + 2:<4} {l0:<4} {0.0:<4} {mass:<4} {bias:<2} {fbeta:<4} {tag:<5}")
    else:
        # Without TAG values
        lines.append(f"LDIN {1:<4} {1:<4} {0.0:<4} {mass:<4} {0.0:<2} {fbeta:<4}")

        for idx, row in patch_info.iterrows():
            l0 = row["l0"]
            var_key = f"lam{row['SELECT']}"
            bias = variables.get(var_key, 0.0)
            lines.append(f"LDIN {idx + 2:<4} {l0:<4} {0.0:<4} {mass:<4} {bias:<2} {fbeta:<4}")

    return "\n".join(lines) + "\n\n"


def generate_rmla_msld(
    patch_info: pd.DataFrame,
    constraint_type: str = "fnex",
    fnex: float = 5.5,
    fpie_width: float = 1.0,
    fpie_force: float = 100.0,
) -> str:
    """Generate RMLA and MSLD/MSMA statements.

    RMLA removes lambda scaling from bonded terms (prevents unphysical results).
    MSLD assigns blocks to sites, MSMA builds the interaction matrix.

    Args:
        patch_info: DataFrame from patches.dat
        constraint_type: Implicit constraint type ("fnex" or "fpie")
        fnex: FNEX parameter value (when constraint_type="fnex")
        fpie_width: FPIE flat-bottom well width (when constraint_type="fpie")
        fpie_force: FPIE flat-bottom force constant (when constraint_type="fpie")

    Returns:
        CHARMM RMLA/MSLD/MSMA statements
    """
    lines = [
        "!------------------------------------------",
        "! All bond/angle/dihe terms treated at full strength (no scaling),",
        "! prevent unphysical results",
        "!------------------------------------------\n",
        "rmla bond theta impr\n",
        "!------------------------------------------",
        "! MSLD - numbers assign each block to the specified site",
        "!------------------------------------------\n",
        "MSLD 0 -",
    ]

    # Build site assignment string
    for i, select in enumerate(patch_info["SELECT"]):
        site_num = select.split("s")[1]
        if i < len(patch_info) - 1:
            lines.append(f"{site_num}  -")
        else:
            lines.append(f"{site_num} -")

    # Generate constraint specification based on type
    if constraint_type == "fpie":
        constraint_str = f"fpie widt {fpie_width} forc {fpie_force}"
    else:
        constraint_str = f"fnex {fnex}"

    lines.extend([
        f"{constraint_str} \n",
        "!------------------------------------------",
        "! Constructs the interaction matrix",
        "!------------------------------------------\n",
        "MSMA\n",
        "!------------------------------------------",
        "! PME for electrostatics",
        "!------------------------------------------\n",
        "PMEL EX\n",
    ])

    return "\n".join(lines)


def generate_ldbv_statements(
    patch_info: pd.DataFrame,
    variables: dict[str, float],
    fnex: float = 5.5,
    chi_offset: float | None = None,
    omega_decay: float | None = None,
) -> str:
    """Generate LDBI/LDBV statements for variable lambda potentials.

    Three types of potentials:
    1. Quadratic barriers (type 6) - prevent lambda from crossing site boundaries
    2. Endpoint potentials (type 8, REF=CHI_OFFSET) - bias toward physical endpoints
    3. Skew potentials (type 10, REF=-OMEGA_DECAY) - asymmetric bias between states

    The REF values for types 8 and 10 are derived from FNEX via bias_constants.

    Args:
        patch_info: DataFrame from patches.dat
        variables: Variables from ALF variable file
        fnex: FNEX softmax constraint parameter (default 5.5)
        chi_offset: Override s-term sigmoid offset (None = derive from fnex).
        omega_decay: Override x-term exponential decay (None = derive from fnex).

    Returns:
        CHARMM LDBI/LDBV statements
    """
    from .bias_constants import derive_bias_constants
    constants = derive_bias_constants(fnex, chi_offset=chi_offset, omega_decay=omega_decay)
    # Build all LDBV statements first to count them
    ldbv_lines = []
    idx = 0

    # Type 6: Quadratic Barriers
    barrier_lines = [
        "!------------------------------------------",
        "! Quadratic Barriers",
        "!------------------------------------------\n",
    ]

    for site in patch_info["site"].unique():
        site_data = patch_info[patch_info["site"] == site]

        for i1, row1 in site_data.iterrows():
            for i2, row2 in site_data.iterrows():
                if i2 > i1:
                    idx += 1
                    block1 = i1 + 2
                    block2 = i2 + 2
                    var_key = f"c{row1['SELECT']}{row2['SELECT']}"
                    c_val = variables.get(var_key, 0.0)
                    barrier_lines.append(
                        f"ldbv {idx:<3} {block1:<2} {block2:<2} {6:<4} {0.0:<8} {c_val:<6} {0:<1}"
                    )

    ldbv_lines.extend(barrier_lines)

    # Type 8: Endpoint Potentials
    endpoint_lines = [
        "!------------------------------------------",
        "!End point Potentials",
        "!------------------------------------------\n",
    ]

    for site in patch_info["site"].unique():
        site_data = patch_info[patch_info["site"] == site]

        for i1, row1 in site_data.iterrows():
            for i2, row2 in site_data.iterrows():
                if i2 != i1:
                    idx += 1
                    block1 = i1 + 2
                    block2 = i2 + 2
                    var_key = f"s{row1['SELECT']}{row2['SELECT']}"
                    s_val = variables.get(var_key, 0.0)
                    endpoint_lines.append(
                        f"ldbv {idx:<3} {block1:<2} {block2:<2} {8:<4} {constants.chi_offset:<8.5f} {s_val:<6} {0:<1}"
                    )

    ldbv_lines.extend(endpoint_lines)

    # Type 10: Skew Potentials
    skew_lines = [
        "!------------------------------------------",
        "! Skew Potentials",
        "!------------------------------------------\n",
    ]

    for site in patch_info["site"].unique():
        site_data = patch_info[patch_info["site"] == site]

        for i1, row1 in site_data.iterrows():
            for i2, row2 in site_data.iterrows():
                if i2 != i1:
                    idx += 1
                    block1 = i1 + 2
                    block2 = i2 + 2
                    var_key = f"x{row1['SELECT']}{row2['SELECT']}"
                    x_val = variables.get(var_key, 0.0)
                    skew_lines.append(
                        f"ldbv {idx:<3} {block1:<2} {block2:<2} {10:<4} {-constants.omega_decay:<8} {x_val:<6} {0:<1}"
                    )

    ldbv_lines.extend(skew_lines)

    # Prepend LDBI with total count
    header = [
        "!------------------------------------------",
        "! Variable lambda potentials",
        "!------------------------------------------\n",
        f"LDBI {idx}",
    ]

    return "\n".join(header + ldbv_lines) + "\n"


def build_block_command(
    patch_info: pd.DataFrame,
    variables: dict[str, float],
    config: BlockConfig,
    constraint_type: str = "fnex",
    fnex: float = 5.5,
    fpie_width: float = 1.0,
    fpie_force: float = 100.0,
    chi_offset: float | None = None,
    omega_decay: float | None = None,
) -> str:
    """Build complete BLOCK command string.

    Args:
        patch_info: DataFrame from patches.dat
        variables: Variables from ALF variable file
        config: Block configuration
        constraint_type: Implicit constraint type ("fnex" or "fpie")
        fnex: FNEX parameter value (when constraint_type="fnex")
        fpie_width: FPIE flat-bottom well width (when constraint_type="fpie")
        fpie_force: FPIE flat-bottom force constant (when constraint_type="fpie")
        chi_offset: Override s-term sigmoid offset (None = derive from fnex).
        omega_decay: Override x-term exponential decay (None = derive from fnex).

    Returns:
        Complete CHARMM BLOCK command
    """
    n_blocks = len(patch_info) + 1  # +1 for environment block

    parts = [
        generate_block_header(n_blocks),
        generate_call_statements(patch_info),
        generate_exclusions(patch_info),
        generate_dynamics_setup(config),
        generate_ldin_statements(patch_info, variables, config),
        generate_rmla_msld(
            patch_info,
            constraint_type=constraint_type,
            fnex=fnex,
            fpie_width=fpie_width,
            fpie_force=fpie_force,
        ),
        generate_ldbv_statements(patch_info, variables, fnex=fnex,
                                 chi_offset=chi_offset, omega_decay=omega_decay),
        "END",
    ]

    return "\n".join(parts)


def write_block_file(
    output_path: Path,
    patch_info: pd.DataFrame,
    variables: dict[str, float],
    config: BlockConfig,
    constraint_type: str = "fnex",
    fnex: float = 5.5,
    fpie_width: float = 1.0,
    fpie_force: float = 100.0,
    chi_offset: float | None = None,
    omega_decay: float | None = None,
) -> str:
    """Generate and write BLOCK command to file.

    Args:
        output_path: Path for output .str file
        patch_info: DataFrame from patches.dat
        variables: Variables from ALF variable file
        config: Block configuration
        constraint_type: Implicit constraint type ("fnex" or "fpie")
        fnex: FNEX parameter value (when constraint_type="fnex")
        fpie_width: FPIE flat-bottom well width (when constraint_type="fpie")
        fpie_force: FPIE flat-bottom force constant (when constraint_type="fpie")
        chi_offset: Override s-term sigmoid offset (None = derive from fnex).
        omega_decay: Override x-term exponential decay (None = derive from fnex).

    Returns:
        The generated BLOCK command string
    """
    block_cmd = build_block_command(
        patch_info,
        variables,
        config,
        constraint_type=constraint_type,
        fnex=fnex,
        fpie_width=fpie_width,
        fpie_force=fpie_force,
        chi_offset=chi_offset,
        omega_decay=omega_decay,
    )

    with open(output_path, "w") as f:
        f.write(block_cmd)

    return block_cmd
