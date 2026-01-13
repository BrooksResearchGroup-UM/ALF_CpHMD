"""Tests for utility functions that don't require pyCHARMM."""

import pytest
from pathlib import Path


class TestSolvateUtils:
    """Tests for solvate utility functions."""

    def test_water_density_room_temp(self):
        """Test water density at room temperature (25°C = 298.15K)."""
        from cphmd.setup.solvate import water_density

        density = water_density(298.15)
        # Water density at 25°C is approximately 997 g/cm³
        assert 995 < density < 1000

    def test_water_density_body_temp(self):
        """Test water density at body temperature (37°C = 310.15K)."""
        from cphmd.setup.solvate import water_density

        density = water_density(310.15)
        # Water density at 37°C is approximately 993 g/cm³
        assert 991 < density < 996

    def test_water_density_cold(self):
        """Test water density at cold temperature (4°C = 277.15K)."""
        from cphmd.setup.solvate import water_density

        density = water_density(277.15)
        # Water density at 4°C is approximately 1000 g/cm³ (maximum)
        assert 999 < density < 1001

    def test_get_box_parameters_cubic(self):
        """Test box parameter calculation for cubic box."""
        from cphmd.setup.solvate import get_box_parameters

        stats = {"xmax": 25, "xmin": -25, "ymax": 20, "ymin": -20, "zmax": 15, "zmin": -15}
        pad = 10.0

        A, B, C, Alpha, Beta, Gamma, BoxX, BoxY, BoxZ = get_box_parameters("CUBIC", stats, pad)

        # For CUBIC, all dimensions should be equal
        assert A == B == C
        assert Alpha == Beta == Gamma == 90.0
        # Max dimension is 50 (xmax - xmin), + 2*pad = 70, + 1 = 71
        assert A == 71
        assert BoxX == BoxY == BoxZ == pytest.approx(71 * 1.1, rel=0.01)

    def test_get_box_parameters_octahedral(self):
        """Test box parameter calculation for octahedral box."""
        from cphmd.setup.solvate import get_box_parameters

        stats = {"xmax": 20, "xmin": -20, "ymax": 20, "ymin": -20, "zmax": 20, "zmin": -20}
        pad = 10.0

        A, B, C, Alpha, Beta, Gamma, BoxX, BoxY, BoxZ = get_box_parameters("OCTAHEDRAL", stats, pad)

        # For OCTAHEDRAL, all dimensions equal and angles are ~109.47°
        assert A == B == C
        assert Alpha == Beta == Gamma == pytest.approx(109.4712206344907)

    def test_get_box_parameters_invalid(self):
        """Test that invalid crystal type raises error."""
        from cphmd.setup.solvate import get_box_parameters

        stats = {"xmax": 10, "xmin": -10, "ymax": 10, "ymin": -10, "zmax": 10, "zmin": -10}

        with pytest.raises(ValueError, match="Invalid crystal_type"):
            get_box_parameters("INVALID", stats, 10.0)


class TestPatchingUtils:
    """Tests for patching utility functions."""

    def test_fft_number_small(self):
        """Test FFT number calculation for small values."""
        from cphmd.core.patching import _fft_number

        # 64 = 2^6 is a valid FFT number
        assert _fft_number(60) == 64
        assert _fft_number(64) == 64

    def test_fft_number_medium(self):
        """Test FFT number calculation for medium values."""
        from cphmd.core.patching import _fft_number

        # 96 = 2^5 * 3 is a valid FFT number
        assert _fft_number(90) == 96
        # 128 = 2^7 is a valid FFT number
        assert _fft_number(100) == 108  # 108 = 2^2 * 3^3

    def test_fft_number_large(self):
        """Test FFT number calculation for larger values."""
        from cphmd.core.patching import _fft_number

        result = _fft_number(200)
        # Result should be >= 200 and only have factors of 2, 3, 5
        assert result >= 200
        # Check it's a valid FFT number (even and factors of 2,3,5)
        temp = result
        for factor in [2, 3, 5]:
            while temp % factor == 0:
                temp //= factor
        assert temp == 1

    def test_parse_selection_criteria_resname(self):
        """Test parsing residue name selections."""
        from cphmd.core.patching import _parse_selection_criteria

        result = _parse_selection_criteria(["ASP", "glu", "HSP"])

        assert len(result) == 3
        assert result[0] == ("resname", "ASP")
        assert result[1] == ("resname", "GLU")
        assert result[2] == ("resname", "HSP")

    def test_parse_selection_criteria_resid(self):
        """Test parsing residue ID selections."""
        from cphmd.core.patching import _parse_selection_criteria

        result = _parse_selection_criteria(["15", "42", "100"])

        assert len(result) == 3
        assert result[0] == ("resid", "15")
        assert result[1] == ("resid", "42")
        assert result[2] == ("resid", "100")

    def test_parse_selection_criteria_segid_resid(self):
        """Test parsing segid:resid selections."""
        from cphmd.core.patching import _parse_selection_criteria

        result = _parse_selection_criteria(["PROA:15", "proa:42"])

        assert len(result) == 2
        assert result[0] == ("segid_resid", ("PROA", "15"))
        assert result[1] == ("segid_resid", ("PROA", "42"))

    def test_parse_selection_criteria_mixed(self):
        """Test parsing mixed selection criteria."""
        from cphmd.core.patching import _parse_selection_criteria

        result = _parse_selection_criteria(["ASP", "15", "PROA:42"])

        assert len(result) == 3
        assert result[0] == ("resname", "ASP")
        assert result[1] == ("resid", "15")
        assert result[2] == ("segid_resid", ("PROA", "42"))

    def test_should_patch_residue_no_criteria(self):
        """Test that all residues are patched when no criteria given."""
        from cphmd.core.patching import _should_patch_residue

        assert _should_patch_residue("PROA", 15, "ASP", []) is True
        assert _should_patch_residue("PROA", 42, "GLU", []) is True

    def test_should_patch_residue_resname_match(self):
        """Test residue name matching."""
        from cphmd.core.patching import _should_patch_residue

        criteria = [("resname", "ASP"), ("resname", "GLU")]

        assert _should_patch_residue("PROA", 15, "ASP", criteria) is True
        assert _should_patch_residue("PROA", 42, "GLU", criteria) is True
        assert _should_patch_residue("PROA", 50, "LYS", criteria) is False

    def test_should_patch_residue_resid_match(self):
        """Test residue ID matching."""
        from cphmd.core.patching import _should_patch_residue

        criteria = [("resid", "15"), ("resid", "42")]

        assert _should_patch_residue("PROA", 15, "ASP", criteria) is True
        assert _should_patch_residue("PROA", 42, "GLU", criteria) is True
        assert _should_patch_residue("PROA", 50, "ASP", criteria) is False

    def test_should_patch_residue_segid_resid_match(self):
        """Test segid:resid matching."""
        from cphmd.core.patching import _should_patch_residue

        criteria = [("segid_resid", ("PROA", "15"))]

        assert _should_patch_residue("PROA", 15, "ASP", criteria) is True
        assert _should_patch_residue("PROB", 15, "ASP", criteria) is False
        assert _should_patch_residue("PROA", 42, "ASP", criteria) is False


class TestDataclasses:
    """Tests for configuration dataclasses."""

    def test_solvation_config_defaults(self):
        """Test SolvationConfig default values."""
        from cphmd.setup.solvate import SolvationConfig

        config = SolvationConfig(input_file="test.pdb")

        assert config.output_dir == "solvated"
        assert config.crystal_type == "OCTAHEDRAL"
        assert config.padding == 10.0
        assert config.salt_concentration == 0.10
        assert config.positive_ion == "POT"
        assert config.negative_ion == "CLA"
        assert config.temperature == 298.15
        assert config.skip_ions is False
        assert config.ion_method == "SLTCAP"
        assert config.min_ion_distance == 5.0

    def test_solvation_config_custom(self):
        """Test SolvationConfig with custom values."""
        from cphmd.setup.solvate import SolvationConfig

        config = SolvationConfig(
            input_file="protein.pdb",
            output_dir="output",
            crystal_type="CUBIC",
            padding=15.0,
            salt_concentration=0.15,
            temperature=310.0,
        )

        assert config.input_file == "protein.pdb"
        assert config.output_dir == "output"
        assert config.crystal_type == "CUBIC"
        assert config.padding == 15.0
        assert config.salt_concentration == 0.15
        assert config.temperature == 310.0

    def test_patch_config_defaults(self):
        """Test PatchConfig default values."""
        from cphmd.core.patching import PatchConfig

        config = PatchConfig(input_folder="test_folder")

        assert config.structure_file == "solvated"
        assert config.hmr is True
        assert config.hmr_waters is False
        assert config.selected_residues == []

    def test_patch_config_custom(self):
        """Test PatchConfig with custom values."""
        from cphmd.core.patching import PatchConfig

        config = PatchConfig(
            input_folder="my_folder",
            structure_file="my_structure",
            hmr=False,
            hmr_waters=True,
            selected_residues=["ASP", "GLU"],
        )

        assert config.input_folder == "my_folder"
        assert config.structure_file == "my_structure"
        assert config.hmr is False
        assert config.hmr_waters is True
        assert config.selected_residues == ["ASP", "GLU"]
