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
        """Two states: E=[0, 10] → b should favor higher-energy state."""
        from cphmd.core.bias_guesser import compute_b_from_endpoints

        endpoint_energies = {0: np.array([0.0, 10.0])}
        nsubs = [2]
        b = compute_b_from_endpoints(endpoint_energies, nsubs)
        assert b.shape == (1, 2)
        # b[i] = -(E[i] - E_mean); E_mean=5
        # b[0] = -(0 - 5) = 5; b[1] = -(10 - 5) = -5
        np.testing.assert_allclose(b[0], [5.0, -5.0])

    def test_three_state_glu(self):
        """Three-state GLU: different energies for each state."""
        from cphmd.core.bias_guesser import compute_b_from_endpoints

        # Simulating GLU: deprotonated (state 0), protonated1, protonated2
        endpoint_energies = {0: np.array([-100.0, -90.0, -85.0])}
        nsubs = [3]
        b = compute_b_from_endpoints(endpoint_energies, nsubs)
        assert b.shape == (1, 3)
        # E_mean = (-100 + -90 + -85) / 3 = -91.667
        e_mean = np.mean([-100.0, -90.0, -85.0])
        expected = -(np.array([-100.0, -90.0, -85.0]) - e_mean)
        np.testing.assert_allclose(b[0], expected)
        # Sum should be zero (centered)
        np.testing.assert_allclose(b.sum(), 0.0, atol=1e-10)

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

        # Site 0: E_mean=5, b = [5, -5]
        np.testing.assert_allclose(b[0, 0:2], [5.0, -5.0])
        # Site 1: E_mean=15, b = [10, 0, -10]
        np.testing.assert_allclose(b[0, 2:5], [10.0, 0.0, -10.0])


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
