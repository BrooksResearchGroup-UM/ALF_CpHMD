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
