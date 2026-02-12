"""Tests for initial bias guessing from single-point energies."""

from __future__ import annotations

import numpy as np


class TestComputeBFromEndpoints:
    """Test linear bias (b) computation from endpoint energies."""

    def test_two_state_symmetric(self):
        """Two states with equal energy → zero biases."""
        from cphmd.core.bias_guesser import compute_b_from_endpoints

        # site 0 has 2 substates, energies are equal
        endpoint_energies = {0: np.array([100.0, 100.0])}
        nsubs = [2]
        b = compute_b_from_endpoints(endpoint_energies, nsubs)
        assert b.shape == (1, 2)
        np.testing.assert_allclose(b, 0.0, atol=1e-10)

    def test_two_state_asymmetric(self):
        """Two states: E=[0, 10] → b[0]=0 (ref), b[1]=-(10-0)=-10."""
        from cphmd.core.bias_guesser import compute_b_from_endpoints

        endpoint_energies = {0: np.array([0.0, 10.0])}
        nsubs = [2]
        b = compute_b_from_endpoints(endpoint_energies, nsubs)
        assert b.shape == (1, 2)
        # b[i] = -(E[i] - E[0]); b[0]=0, b[1]=-(10-0)=-10
        np.testing.assert_allclose(b[0], [0.0, -10.0])

    def test_three_state_glu(self):
        """Three-state GLU: b[0]=0, others relative to reference."""
        from cphmd.core.bias_guesser import compute_b_from_endpoints

        # Simulating GLU: deprotonated (state 0), protonated1, protonated2
        endpoint_energies = {0: np.array([-100.0, -90.0, -85.0])}
        nsubs = [3]
        b = compute_b_from_endpoints(endpoint_energies, nsubs)
        assert b.shape == (1, 3)
        # b[i] = -(E[i] - E[0]); E[0]=-100
        # b[0]=0, b[1]=-(-90-(-100))=-10, b[2]=-(-85-(-100))=-15
        np.testing.assert_allclose(b[0], [0.0, -10.0, -15.0])
        # First substate is always zero
        assert b[0, 0] == 0.0

    def test_multisite_independent(self):
        """Two sites: biases computed independently."""
        from cphmd.core.bias_guesser import compute_b_from_endpoints

        endpoint_energies = {
            0: np.array([0.0, 10.0]),  # site 0: 2 substates
            1: np.array([5.0, 15.0, 25.0]),  # site 1: 3 substates
        }
        nsubs = [2, 3]
        b = compute_b_from_endpoints(endpoint_energies, nsubs)
        assert b.shape == (1, 5)  # 2 + 3 = 5 blocks

        # Site 0: b[0]=0 (ref), b[1]=-(10-0)=-10
        np.testing.assert_allclose(b[0, 0:2], [0.0, -10.0])
        # Site 1: b[0]=0 (ref), b[1]=-(15-5)=-10, b[2]=-(25-5)=-20
        np.testing.assert_allclose(b[0, 2:5], [0.0, -10.0, -20.0])


class TestComputeCFromMidpoints:
    """Test quadratic barrier (c) computation from midpoint energies."""

    def test_two_state_no_barrier(self):
        """Linear interpolation → zero barrier."""
        from cphmd.core.bias_guesser import compute_c_from_midpoints

        endpoint_energies = {0: np.array([0.0, 10.0])}
        # Midpoint energy = exact linear interpolation = 5.0
        midpoint_energies = {0: {(0, 1): 5.0}}
        nsubs = [2]
        c = compute_c_from_midpoints(midpoint_energies, endpoint_energies, nsubs)
        assert c.shape == (2, 2)
        np.testing.assert_allclose(c, 0.0, atol=1e-10)

    def test_two_state_positive_barrier(self):
        """Midpoint energy higher than interpolation → negative c (barrier)."""
        from cphmd.core.bias_guesser import compute_c_from_midpoints

        endpoint_energies = {0: np.array([0.0, 10.0])}
        # Midpoint at 8.0 instead of 5.0 → excess of 3.0
        midpoint_energies = {0: {(0, 1): 8.0}}
        nsubs = [2]
        c = compute_c_from_midpoints(midpoint_energies, endpoint_energies, nsubs)
        # c[i][j] = -(E_mid - 0.5*(E_i + E_j)) = -(8 - 5) = -3
        np.testing.assert_allclose(c[0, 1], -3.0)
        np.testing.assert_allclose(c[1, 0], -3.0)  # Symmetric

    def test_three_state_all_pairs(self):
        """Three substates: all 3 pairs computed."""
        from cphmd.core.bias_guesser import compute_c_from_midpoints

        endpoint_energies = {0: np.array([0.0, 10.0, 20.0])}
        midpoint_energies = {
            0: {
                (0, 1): 7.0,  # excess = 7 - 5 = 2
                (0, 2): 15.0,  # excess = 15 - 10 = 5
                (1, 2): 18.0,  # excess = 18 - 15 = 3
            }
        }
        nsubs = [3]
        c = compute_c_from_midpoints(midpoint_energies, endpoint_energies, nsubs)
        assert c.shape == (3, 3)
        np.testing.assert_allclose(c[0, 1], -2.0)
        np.testing.assert_allclose(c[0, 2], -5.0)
        np.testing.assert_allclose(c[1, 2], -3.0)
        # Symmetric
        np.testing.assert_allclose(c[1, 0], -2.0)
        np.testing.assert_allclose(c[2, 0], -5.0)
        np.testing.assert_allclose(c[2, 1], -3.0)
        # Diagonal is zero
        np.testing.assert_allclose(np.diag(c), 0.0)

    def test_multisite_independent(self):
        """Two sites: c is block-diagonal (no cross-site coupling)."""
        from cphmd.core.bias_guesser import compute_c_from_midpoints

        endpoint_energies = {
            0: np.array([0.0, 10.0]),
            1: np.array([5.0, 15.0, 25.0]),
        }
        midpoint_energies = {
            0: {(0, 1): 8.0},  # excess 3
            1: {
                (0, 1): 14.0,  # excess 4
                (0, 2): 22.0,  # excess 7
                (1, 2): 25.0,  # excess 5
            },
        }
        nsubs = [2, 3]
        c = compute_c_from_midpoints(midpoint_energies, endpoint_energies, nsubs)
        assert c.shape == (5, 5)
        # Site 0 block
        np.testing.assert_allclose(c[0, 1], -3.0)
        # Site 1 block
        np.testing.assert_allclose(c[2, 3], -4.0)
        np.testing.assert_allclose(c[2, 4], -7.0)
        np.testing.assert_allclose(c[3, 4], -5.0)
        # Cross-site should be zero
        np.testing.assert_allclose(c[0, 2], 0.0)
        np.testing.assert_allclose(c[1, 3], 0.0)


class TestGenerateLambdaConfigs:
    """Test lambda configuration generation for energy evaluations."""

    def test_two_state_endpoints(self):
        """2-state site: 2 endpoint configs."""
        from cphmd.core.bias_guesser import generate_lambda_configs

        configs = generate_lambda_configs(nsubs=[2])
        endpoints = configs[0]["endpoints"]
        assert len(endpoints) == 2
        np.testing.assert_allclose(endpoints[0], [1.0, 0.0])
        np.testing.assert_allclose(endpoints[1], [0.0, 1.0])

    def test_two_state_midpoints(self):
        """2-state site: 1 midpoint config."""
        from cphmd.core.bias_guesser import generate_lambda_configs

        configs = generate_lambda_configs(nsubs=[2])
        midpoints = configs[0]["midpoints"]
        assert len(midpoints) == 1
        pair, lam = midpoints[0]
        assert pair == (0, 1)
        np.testing.assert_allclose(lam, [0.5, 0.5])

    def test_three_state_endpoints(self):
        """3-state site: 3 endpoint configs."""
        from cphmd.core.bias_guesser import generate_lambda_configs

        configs = generate_lambda_configs(nsubs=[3])
        endpoints = configs[0]["endpoints"]
        assert len(endpoints) == 3
        np.testing.assert_allclose(endpoints[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(endpoints[1], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(endpoints[2], [0.0, 0.0, 1.0])

    def test_three_state_midpoints(self):
        """3-state site: 3 midpoint configs (all pairs)."""
        from cphmd.core.bias_guesser import generate_lambda_configs

        configs = generate_lambda_configs(nsubs=[3])
        midpoints = configs[0]["midpoints"]
        assert len(midpoints) == 3
        pairs = [m[0] for m in midpoints]
        assert (0, 1) in pairs
        assert (0, 2) in pairs
        assert (1, 2) in pairs

    def test_multisite_count(self):
        """Two sites: each gets independent configs."""
        from cphmd.core.bias_guesser import generate_lambda_configs

        configs = generate_lambda_configs(nsubs=[2, 3])
        assert len(configs) == 2
        assert len(configs[0]["endpoints"]) == 2
        assert len(configs[1]["endpoints"]) == 3
        assert len(configs[0]["midpoints"]) == 1
        assert len(configs[1]["midpoints"]) == 3


class TestInitVarsIntegration:
    """Test init_vars accepts b_init and c_init."""

    def test_init_vars_with_guessed_biases(self, tmp_path):
        """init_vars writes guessed b to b_prev.dat when b_init is provided."""
        from cphmd.core.alf_utils import init_vars

        alf_info = {
            "name": "test",
            "nsubs": np.array([3]),
            "nblocks": 3,
            "nreps": 1,
            "ncentral": 0,
            "nnodes": 1,
            "temp": 298.15,
            "engine": "charmm",
            "ntersite": [0, 0],
            "fnex": 5.5,
            "g_imp_bins": 20,
            "cutlsum": 0.5,
        }

        b_init = np.array([[0.0, -5.0, -10.0]])
        c_init = np.array(
            [
                [0.0, -3.0, -2.0],
                [-3.0, 0.0, -1.0],
                [-2.0, -1.0, 0.0],
            ]
        )

        analysis0 = init_vars(tmp_path, alf_info, b_init=b_init, c_init=c_init)

        b_prev = np.loadtxt(analysis0 / "b_prev.dat")
        c_prev = np.loadtxt(analysis0 / "c_prev.dat")
        np.testing.assert_allclose(b_prev, b_init.flatten(), atol=1e-5)
        np.testing.assert_allclose(c_prev, c_init, atol=1e-5)

        # Current cycle b.dat should be zero (delta for this cycle)
        b_curr = np.loadtxt(analysis0 / "b.dat")
        np.testing.assert_allclose(b_curr, 0.0, atol=1e-10)

    def test_init_vars_without_guesses_gives_zeros(self, tmp_path):
        """init_vars with no b_init/c_init produces zero biases (legacy behavior)."""
        from cphmd.core.alf_utils import init_vars

        alf_info = {
            "name": "test",
            "nsubs": np.array([2]),
            "nblocks": 2,
            "nreps": 1,
            "ncentral": 0,
            "nnodes": 1,
            "temp": 298.15,
            "engine": "charmm",
            "ntersite": [0, 0],
            "fnex": 5.5,
            "g_imp_bins": 20,
            "cutlsum": 0.5,
        }

        analysis0 = init_vars(tmp_path, alf_info)

        b_prev = np.loadtxt(analysis0 / "b_prev.dat")
        c_prev = np.loadtxt(analysis0 / "c_prev.dat")
        np.testing.assert_allclose(b_prev, 0.0, atol=1e-10)
        np.testing.assert_allclose(c_prev, 0.0, atol=1e-10)
