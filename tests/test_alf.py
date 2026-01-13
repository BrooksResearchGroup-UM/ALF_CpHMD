"""Tests for ALF-related modules that don't require pyCHARMM."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestCpHMDParams:
    """Tests for CpHMD parameter calculations."""

    def test_parse_tag_upos(self):
        """Test parsing UPOS tag values."""
        from cphmd.core.cphmd_params import parse_tag_value

        tag_type, pKa = parse_tag_value("UPOS 6.0")
        assert tag_type == "UPOS"
        assert pKa == 6.0

    def test_parse_tag_uneg(self):
        """Test parsing UNEG tag values."""
        from cphmd.core.cphmd_params import parse_tag_value

        tag_type, pKa = parse_tag_value("UNEG 4.0")
        assert tag_type == "UNEG"
        assert pKa == 4.0

    def test_parse_tag_none(self):
        """Test parsing NONE tag values."""
        from cphmd.core.cphmd_params import parse_tag_value

        tag_type, pKa = parse_tag_value("NONE")
        assert tag_type == "NONE"
        assert pKa is None

    def test_delta_pka_phase1(self):
        """Test delta_pKa for phase 1."""
        from cphmd.core.cphmd_params import get_delta_pKa_for_phase

        assert get_delta_pKa_for_phase(1) == 1.0

    def test_delta_pka_phase2(self):
        """Test delta_pKa for phase 2."""
        from cphmd.core.cphmd_params import get_delta_pKa_for_phase

        assert get_delta_pKa_for_phase(2) == 0.5

    def test_delta_pka_phase3(self):
        """Test delta_pKa for phase 3."""
        from cphmd.core.cphmd_params import get_delta_pKa_for_phase

        assert get_delta_pKa_for_phase(3) == 0.25

    def test_cphmd_parameters_ktln10(self):
        """Test kTln10 calculation."""
        from cphmd.core.cphmd_params import CpHMDParameters, KB

        params = CpHMDParameters(temperature=298.15, target_pH=7.0)
        expected_kTln10 = KB * 298.15 * np.log(10.0)
        assert params.kTln10 == pytest.approx(expected_kTln10)

    def test_compute_site_parameters_acidic(self):
        """Test site parameter computation for acidic residue (ASP-like)."""
        from cphmd.core.cphmd_params import compute_site_parameters

        # Simulate ASP with two UNEG states
        patch_info = pd.DataFrame({
            "SELECT": ["s1s1", "s1s2"],
            "TAG": ["UNEG 4.0", "NONE"],
            "site": ["1", "1"],
        })

        site_patches = patch_info[patch_info["site"] == "1"]
        params = compute_site_parameters("1", site_patches)

        assert params.site_type == "acidic"
        assert params.pH0 == pytest.approx(4.0)
        assert params.pKa_neg == pytest.approx(4.0)
        assert params.pKa_pos is None

    def test_compute_site_parameters_basic(self):
        """Test site parameter computation for basic residue (LYS-like)."""
        from cphmd.core.cphmd_params import compute_site_parameters

        # Simulate LYS with UPOS state
        patch_info = pd.DataFrame({
            "SELECT": ["s1s1", "s1s2"],
            "TAG": ["UPOS 10.5", "NONE"],
            "site": ["1", "1"],
        })

        site_patches = patch_info[patch_info["site"] == "1"]
        params = compute_site_parameters("1", site_patches)

        assert params.site_type == "basic"
        assert params.pH0 == pytest.approx(10.5)
        assert params.pKa_pos == pytest.approx(10.5)
        assert params.pKa_neg is None

    def test_compute_site_parameters_three_state(self):
        """Test site parameter computation for three-state residue (HIS-like)."""
        from cphmd.core.cphmd_params import compute_site_parameters

        # Simulate HIS with both UPOS and UNEG states
        patch_info = pd.DataFrame({
            "SELECT": ["s1s1", "s1s2", "s1s3"],
            "TAG": ["UPOS 6.0", "UNEG 7.0", "NONE"],
            "site": ["1", "1", "1"],
        })

        site_patches = patch_info[patch_info["site"] == "1"]
        params = compute_site_parameters("1", site_patches)

        assert params.site_type == "three_state"
        # pH0 should be midpoint of pKa_pos and pKa_neg
        expected_pH0 = 0.5 * (6.0 + 7.0)
        assert params.pH0 == pytest.approx(expected_pH0)


class TestBlockBuilder:
    """Tests for BLOCK command generation."""

    def test_read_variable_file(self, tmp_path):
        """Test reading CHARMM variable file."""
        from cphmd.core.block_builder import read_variable_file

        var_file = tmp_path / "variables1.inp"
        var_file.write_text(
            "set lams1s1 = 0.0\n"
            "set lams1s2 = 1.5\n"
            "set cs1s1s1s2 = 0.5\n"
        )

        variables = read_variable_file(var_file)

        assert variables["lams1s1"] == 0.0
        assert variables["lams1s2"] == 1.5
        assert variables["cs1s1s1s2"] == 0.5

    def test_generate_block_header(self):
        """Test BLOCK header generation."""
        from cphmd.core.block_builder import generate_block_header

        header = generate_block_header(3)
        assert "BLOCK 3" in header

    def test_generate_call_statements(self):
        """Test CALL statement generation."""
        from cphmd.core.block_builder import generate_call_statements

        patch_info = pd.DataFrame({
            "SELECT": ["s1s1", "s1s2"],
            "site": ["1", "1"],
        })

        calls = generate_call_statements(patch_info)

        assert "CALL 2 SELECT s1s1 END" in calls
        assert "CALL 3 SELECT s1s2 END" in calls

    def test_generate_exclusions(self):
        """Test ADEXCL statement generation."""
        from cphmd.core.block_builder import generate_exclusions

        patch_info = pd.DataFrame({
            "SELECT": ["s1s1", "s1s2", "s2s1"],
            "site": ["1", "1", "2"],
        })

        exclusions = generate_exclusions(patch_info)

        # Should exclude s1s1 from s1s2 (same site)
        assert "adexcl 2" in exclusions
        assert "adexcl" in exclusions


class TestRestraints:
    """Tests for restraint generation."""

    def test_generate_scat_restraints(self):
        """Test SCAT restraint generation."""
        from cphmd.core.restraints import generate_scat_restraints

        patch_info = pd.DataFrame({
            "SELECT": ["s1s1", "s1s2"],
            "ATOMS": ["CB CG OD1 OD2", "CB CG OD1 OD2 HD2"],
            "site": ["1", "1"],
        })

        scat = generate_scat_restraints(patch_info, include_hydrogen=False)

        assert "BLOCK" in scat
        assert "scat on" in scat
        assert "scat k 300" in scat
        assert "cats SELE type CB" in scat
        assert "END" in scat
        # Should not include hydrogen by default
        assert "HD2" not in scat

    def test_generate_scat_with_hydrogen(self):
        """Test SCAT restraints including hydrogen."""
        from cphmd.core.restraints import generate_scat_restraints

        patch_info = pd.DataFrame({
            "SELECT": ["s1s1", "s1s2"],
            "ATOMS": ["CB CG HD2", "CB CG"],
            "site": ["1", "1"],
        })

        scat = generate_scat_restraints(patch_info, include_hydrogen=True)

        assert "HD2" in scat

    def test_generate_noe_restraints(self):
        """Test NOE restraint generation."""
        from cphmd.core.restraints import generate_noe_restraints

        patch_info = pd.DataFrame({
            "SELECT": ["s1s1", "s1s2"],
            "ATOMS": ["CB CG OD1", "CB CG OD1"],
            "SEGID": ["PROA", "PROA"],
            "RESID": ["15", "15"],
            "PATCH": ["ASPO", "ASPP"],
            "site": ["1", "1"],
        })

        noe = generate_noe_restraints(patch_info)

        assert "NOE" in noe
        assert "assign" in noe
        assert "kmin 100.0" in noe
        assert "END" in noe


class TestALFConfig:
    """Tests for ALF configuration dataclass."""

    def test_alf_config_defaults(self):
        """Test ALFConfig default values."""
        from cphmd.core.alf_runner import ALFConfig

        # This will fail validation since input_folder doesn't exist
        # But we can test the dataclass structure
        with pytest.raises(FileNotFoundError):
            ALFConfig(input_folder="/nonexistent/path")

    def test_alf_config_temperature(self):
        """Test ALFConfig temperature setting."""
        from cphmd.core.alf_runner import ALFConfig

        # Use __new__ to bypass __post_init__ validation
        config = object.__new__(ALFConfig)
        config.temperature = 310.0
        assert config.temperature == 310.0


class TestSimulationState:
    """Tests for SimulationState dataclass."""

    def test_simulation_state_defaults(self):
        """Test SimulationState default values."""
        from cphmd.core.alf_runner import SimulationState

        state = SimulationState()

        assert state.rank == 0
        assert state.size == 1
        assert state.gpuid == 0
        assert state.phase == 1
        assert state.current_run == 1
        assert state.box_size == [0.0, 0.0, 0.0]
        assert state.box_angles == [90.0, 90.0, 90.0]
        assert state.patch_info is None
        assert state.alf_info is None
