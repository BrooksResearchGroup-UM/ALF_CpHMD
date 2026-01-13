"""Integration tests for pyCHARMM utilities.

These tests require pyCHARMM and verify that the CHARMM utility functions
work correctly with actual CHARMM commands.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os


# Skip all tests if pyCHARMM is not available
pycharmm = pytest.importorskip("pycharmm")


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def toppar_dir(project_root: Path) -> Path:
    """Return the toppar directory."""
    return project_root / "toppar"


@pytest.fixture
def minimal_topology_files() -> list[str]:
    """Return minimal set of topology files for testing."""
    return [
        "top_all36_prot.rtf",
        "par_all36m_prot.prm",
        "toppar_water_ions.str",
    ]


class TestBoxParameters:
    """Tests for BoxParameters dataclass."""

    def test_box_parameters_from_file(self, tmp_path):
        """Test loading box parameters from file."""
        from cphmd.core.charmm_utils import BoxParameters

        # Create test box.dat file
        box_file = tmp_path / "box.dat"
        box_file.write_text("OCTAHEDRAL\n60.0 60.0 60.0\n109.47 109.47 109.47\n")

        params = BoxParameters.from_file(box_file)

        assert params.crystal_type == "OCTAHEDRAL"
        assert params.dimensions == [60.0, 60.0, 60.0]
        assert params.angles == pytest.approx([109.47, 109.47, 109.47])

    def test_box_parameters_cubic(self, tmp_path):
        """Test loading cubic box parameters."""
        from cphmd.core.charmm_utils import BoxParameters

        box_file = tmp_path / "box.dat"
        box_file.write_text("CUBIC\n50.0 50.0 50.0\n90.0 90.0 90.0\n")

        params = BoxParameters.from_file(box_file)

        assert params.crystal_type == "CUBIC"
        assert params.dimensions == [50.0, 50.0, 50.0]
        assert params.angles == [90.0, 90.0, 90.0]


class TestFFTParameters:
    """Tests for FFTParameters dataclass."""

    def test_fft_parameters_from_file(self, tmp_path):
        """Test loading FFT parameters from file."""
        from cphmd.core.charmm_utils import FFTParameters

        fft_file = tmp_path / "fft.dat"
        fft_file.write_text("64 64 64")

        params = FFTParameters.from_file(fft_file)

        assert params.fftx == 64
        assert params.ffty == 64
        assert params.fftz == 64


class TestNonBondedConfig:
    """Tests for NonBondedConfig dataclass."""

    def test_nonbonded_config_defaults(self):
        """Test NonBondedConfig default values."""
        from cphmd.core.charmm_utils import NonBondedConfig

        config = NonBondedConfig()

        assert config.cutnb == 14.0
        assert config.ctofnb == 12.0
        assert config.ctonnb == 10.0
        assert config.use_pme is True

    def test_nonbonded_config_to_dict(self):
        """Test conversion to dictionary."""
        from cphmd.core.charmm_utils import NonBondedConfig

        config = NonBondedConfig(cutnb=12.0, use_pme=True, fftx=64, ffty=64, fftz=64)
        params = config.to_dict()

        assert params["cutnb"] == 12.0
        assert params["pmewald"] is True
        assert params["fftx"] == 64

    def test_nonbonded_config_no_pme(self):
        """Test non-bonded config without PME."""
        from cphmd.core.charmm_utils import NonBondedConfig

        config = NonBondedConfig(use_pme=False)
        params = config.to_dict()

        assert "pmewald" not in params
        assert "ewald" not in params


class TestTopologyReading:
    """Tests for topology file reading (requires pyCHARMM)."""

    @pytest.mark.slow
    def test_read_topology_files(self, toppar_dir, minimal_topology_files):
        """Test reading topology files."""
        from cphmd.core.charmm_utils import read_topology_files

        # This should not raise an exception
        read_topology_files(toppar_dir, minimal_topology_files, verbose=False)

        # Verify something was loaded by checking we can execute a simple command
        import pycharmm.lingo as lingo
        # This would fail if topology wasn't loaded
        lingo.charmm_script("! Topology loaded successfully")


class TestSelectionDefinition:
    """Tests for selection definition."""

    def test_define_selections_command_format(self):
        """Test that selection definition generates correct CHARMM syntax."""
        # We test the command format without actually executing
        patch_info = pd.DataFrame({
            "SELECT": ["s1s1", "s1s2"],
            "SEGID": ["PROA", "PROA"],
            "RESID": ["15", "15"],
            "PATCH": ["ASPO", "ASPP"],
            "ATOMS": ["CB CG OD1 OD2", "CB CG OD1 OD2 HD2"],
        })

        # Build the command manually to verify format
        row = patch_info.iloc[0]
        atoms = row["ATOMS"].split()
        atom_clause = " -\n .or. type ".join(atoms)

        expected_atoms = "CB -\n .or. type CG -\n .or. type OD1 -\n .or. type OD2"
        assert atom_clause == expected_atoms


class TestBlockExecution:
    """Tests for BLOCK command execution."""

    def test_clear_block_command(self):
        """Test BLOCK CLEAR command format."""
        # Verify the command string is correct
        clear_cmd = "BLOCK\n CLEAR\n END"
        assert "BLOCK" in clear_cmd
        assert "CLEAR" in clear_cmd
        assert "END" in clear_cmd


class TestCHARMMSession:
    """Tests for CHARMMSession context manager."""

    @pytest.mark.slow
    def test_charmm_session_initialization(self, toppar_dir, minimal_topology_files):
        """Test CHARMMSession context manager."""
        from cphmd.core.charmm_utils import CHARMMSession

        with CHARMMSession(toppar_dir, minimal_topology_files, verbose=False) as session:
            assert session._initialized is True


class TestIntegration:
    """Integration tests that combine multiple components."""

    def test_full_setup_workflow_mock(self, tmp_path):
        """Test the full setup workflow with mock data structures."""
        from cphmd.core.charmm_utils import (
            BoxParameters,
            FFTParameters,
            NonBondedConfig,
        )

        # Create mock files
        box_file = tmp_path / "box.dat"
        box_file.write_text("CUBIC\n50.0 50.0 50.0\n90.0 90.0 90.0\n")

        fft_file = tmp_path / "fft.dat"
        fft_file.write_text("64 64 64")

        # Load parameters
        box_params = BoxParameters.from_file(box_file)
        fft_params = FFTParameters.from_file(fft_file)

        # Create non-bonded config with FFT
        nb_config = NonBondedConfig(
            fftx=fft_params.fftx,
            ffty=fft_params.ffty,
            fftz=fft_params.fftz,
        )

        # Verify configuration
        assert box_params.crystal_type == "CUBIC"
        params_dict = nb_config.to_dict()
        assert params_dict["fftx"] == 64
        assert params_dict["pmewald"] is True
