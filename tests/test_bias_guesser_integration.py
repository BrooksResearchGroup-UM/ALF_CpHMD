"""Integration tests for bias guessing with pyCHARMM.

These tests require pyCHARMM and the example systems.
Run with: pytest tests/test_bias_guesser_integration.py -v -m integration
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

# Skip entire module if pyCHARMM not available
pycharmm = pytest.importorskip("pycharmm", reason="pyCHARMM required")


@pytest.mark.integration
class TestBiasGuesserGLU:
    """Integration test: bias guessing on GLU water example."""

    @pytest.fixture(autouse=True)
    def setup_charmm(self):
        """Set up CHARMM session with GLU example topology."""
        import pycharmm.read as read
        import pycharmm.settings as settings

        from cphmd.core.charmm_utils import (
            BoxParameters,
            FFTParameters,
            NonBondedConfig,
            define_selections,
            read_topology_files,
            setup_crystal,
            setup_nonbonded,
        )

        example_dir = EXAMPLES_DIR / "00_glu_water" / "solvated"
        prep_dir = example_dir / "prep"

        if not prep_dir.exists():
            pytest.skip("GLU example not found")

        # Load topology
        toppar_dir = EXAMPLES_DIR.parent / "toppar"
        topology_files = [
            "top_all36_prot.rtf",
            "par_all36m_prot.prm",
            "toppar_water_ions.str",
            "my_files/titratable_residues.str",
        ]
        read_topology_files(toppar_dir, topology_files)

        # Load structure
        settings.set_bomb_level(-1)
        read.psf_card(str(prep_dir / "system.psf"))
        settings.set_bomb_level(0)
        read.coor_card(str(prep_dir / "system_min.crd"))

        # Load patches.dat and derive site/sub columns from SELECT
        import pandas as pd

        self.patch_info = pd.read_csv(prep_dir / "patches.dat")
        self.patch_info[["site", "sub"]] = self.patch_info["SELECT"].str.extract(
            r"s(\d+)s(\d+)"
        )
        self.patch_info["site"] = self.patch_info["site"].astype(int)
        self.patch_info["sub"] = self.patch_info["sub"].astype(int)

        # Setup crystal and nonbonded
        box_params = BoxParameters.from_file(prep_dir / "box.dat")
        fft_params = FFTParameters.from_file(prep_dir / "fft.dat")
        nb_config = NonBondedConfig(
            fftx=fft_params.fftx,
            ffty=fft_params.ffty,
            fftz=fft_params.fftz,
        )
        setup_crystal(box_params, nb_config)
        setup_nonbonded(nb_config)

        # Define selections
        define_selections(self.patch_info)

        self.nsubs = [3]  # GLU: 3 substates

        yield

        # Cleanup
        from cphmd.core.charmm_utils import clear_block

        clear_block()

    def test_endpoint_energies_differ(self):
        """Endpoint energies for GLU's 3 states should be different."""
        from cphmd.core.bias_guesser import (
            _build_energy_block_command,
            _evaluate_energy,
            generate_lambda_configs,
        )
        from cphmd.core.charmm_utils import clear_block, execute_block_command

        configs = generate_lambda_configs(self.nsubs)
        energies = []

        for lam in configs[0]["endpoints"]:
            clear_block()
            cmd = _build_energy_block_command(
                self.patch_info,
                {0: lam},
                self.nsubs,
            )
            execute_block_command(cmd)
            energies.append(_evaluate_energy())

        # Energies should not all be identical
        assert not np.allclose(energies[0], energies[1], atol=0.1)
        assert not np.allclose(energies[0], energies[2], atol=0.1)

    def test_guessed_biases_reasonable(self):
        """Guessed biases should have correct shape and non-zero values."""
        from cphmd.core.bias_guesser import guess_initial_biases

        b, c = guess_initial_biases(self.patch_info, self.nsubs)

        # Shape checks
        assert b.shape == (1, 3)
        assert c.shape == (3, 3)

        # b[0] should be 0 (reference convention)
        assert b[0, 0] == 0.0

        # b should not be all zeros (states have different energies)
        assert np.any(np.abs(b) > 0.1), f"b too small: {b}"

        # c should be symmetric
        np.testing.assert_allclose(c, c.T)

        # c diagonal should be zero
        np.testing.assert_allclose(np.diag(c), 0.0)

    def test_guessed_b_reference_convention(self):
        """b[0] should be zero (reference) and other b's should be non-trivial."""
        from cphmd.core.bias_guesser import guess_initial_biases

        b_guess, c_guess = guess_initial_biases(self.patch_info, self.nsubs)

        # First substate is reference → b[0] = 0
        assert b_guess[0, 0] == 0.0

        # Other substates should have non-zero biases
        assert np.any(np.abs(b_guess[0, 1:]) > 1.0), f"b too small: {b_guess}"

        # c barriers should have correct sign (negative = barrier)
        # Extract upper-triangle values from c matrix to compare with preset list
        from cphmd.presets.biases import get_bias_params_only

        try:
            _, c_preset, _, _ = get_bias_params_only("GLU", "pme_ex_vswitch")
        except (KeyError, ValueError):
            pytest.skip("GLU preset not available for this config")

        # c_preset is flat upper-triangle: [c01, c02, c12]
        # c_guess is (3,3) matrix: extract same elements
        c_upper = [c_guess[0, 1], c_guess[0, 2], c_guess[1, 2]]
        for i, (cg, cp) in enumerate(zip(c_upper, c_preset)):
            if abs(cp) > 1.0:
                assert np.sign(cg) == np.sign(cp), (
                    f"c sign mismatch at pair {i}: guess={cg:.2f}, preset={cp:.2f}"
                )
