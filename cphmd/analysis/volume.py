"""
Volume analysis for CpHMD simulations.

This module provides tools for analyzing system volume properties,
particularly useful for understanding hydration and cavity volumes
around titratable residues.

Key Features:
- Calculate molecular volume using CHARMM's COOR VOLU command
- Analyze protein cavity volumes
- Track volume changes during simulations
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cphmd.native import system


@dataclass
class VolumeConfig:
    """Configuration for volume analysis.

    Attributes:
        input_folder: Path to the system folder
        structure_file: Name of structure file (without extension)
        selection: CHARMM selection for volume calculation
        probe_radius: Probe radius for volume calculation (Angstroms)
    """

    input_folder: str | Path
    structure_file: str = "solvated"
    selection: str = "sele segid PROA end"
    probe_radius: float = 1.6

    def __post_init__(self):
        self.input_folder = Path(self.input_folder)


@dataclass
class VolumeResult:
    """Results from volume analysis.

    Attributes:
        total_volume: Total volume of selection (Angstrom^3)
        cavity_volume: Cavity/void volume (Angstrom^3)
        surface_area: Surface area (Angstrom^2)
    """

    total_volume: float | None = None
    cavity_volume: float | None = None
    surface_area: float | None = None


def calculate_volume(config: VolumeConfig) -> VolumeResult:
    """Calculate molecular volume using pyCHARMM.

    This function uses CHARMM's COOR VOLU command to compute
    volume properties of a molecular selection.

    Args:
        config: Configuration for volume calculation

    Returns:
        VolumeResult with computed volumes

    Note:
        Requires pyCHARMM to be properly initialized with
        topology files loaded.
    """
    prep_dir = config.input_folder / "prep"
    psf_file = prep_dir / f"{config.structure_file}.psf"
    crd_file = prep_dir / f"{config.structure_file}.crd"

    if not psf_file.exists():
        raise FileNotFoundError(f"PSF file not found: {psf_file}")

    # Read structure
    system.read_psf(psf_file)
    system.read_coor(crd_file)

    # Calculate volume using CHARMM's method
    # First get bounding box
    volume_script = f"""
    ! Calculate bounding box for space parameter
    calc XSIZE = INT(?XMAX - ?XMIN) + 1
    calc YSIZE = INT(?YMAX - ?YMIN) + 1
    calc ZSIZE = INT(?ZMAX - ?ZMIN) + 1
    calc SPACE = @XSIZE * @YSIZE * @ZSIZE * 1000000

    ! Set probe radius
    SCALar WMAIn = RADIus
    SCALar WMAIn ADD {config.probe_radius}

    ! Calculate volume
    coor volu SPACE @SPACE hole {config.selection}
    """
    with tempfile.TemporaryDirectory(prefix="cphmd_volume_") as tmp:
        script_path = Path(tmp) / "volume.inp"
        script_path.write_text(volume_script)
        system.stream_file(script_path)

    # Extract results from CHARMM variables
    # Note: Actual implementation would parse CHARMM output
    # This is a placeholder that demonstrates the interface

    return VolumeResult(
        total_volume=None,  # Would be extracted from CHARMM
        cavity_volume=None,
        surface_area=None,
    )


def analyze_trajectory_volumes(
    config: VolumeConfig, dcd_file: str | Path, stride: int = 10
) -> dict[str, np.ndarray]:
    """Analyze volume changes over a trajectory.

    Args:
        config: Configuration for volume analysis
        dcd_file: Path to DCD trajectory file
        stride: Frame stride for analysis

    Returns:
        Dictionary with time series of volume properties
    """
    dcd_path = Path(dcd_file)
    if not dcd_path.exists():
        raise FileNotFoundError(f"DCD file not found: {dcd_path}")

    # This would iterate over trajectory frames
    # Placeholder implementation
    volumes = []
    times = []

    # TODO: Implement trajectory iteration with volume calculation

    return {
        "time": np.array(times),
        "volume": np.array(volumes),
    }


# CLI entry point
def main():
    """Command-line interface for volume analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze molecular volumes")
    parser.add_argument("-i", "--input", required=True, help="Input folder")
    parser.add_argument("-f", "--file", default="solvated", help="Structure file name")
    parser.add_argument("-s", "--selection", default="sele segid PROA end", help="CHARMM selection")
    parser.add_argument("-r", "--radius", type=float, default=1.6, help="Probe radius (Angstroms)")

    args = parser.parse_args()

    config = VolumeConfig(
        input_folder=args.input,
        structure_file=args.file,
        selection=args.selection,
        probe_radius=args.radius,
    )

    print(f"Analyzing volumes in {config.input_folder}")
    print("[Note: Volume analysis requires pyCHARMM environment]")

    try:
        result = calculate_volume(config)
        print("\nResults:")
        print(f"  Total volume: {result.total_volume}")
        print(f"  Cavity volume: {result.cavity_volume}")
    except ImportError as e:
        print(f"Error: {e}")
        print("Volume analysis requires pyCHARMM to be installed and configured.")


if __name__ == "__main__":
    main()
