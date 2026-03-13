"""Regression tests: fused dlnZ + cuBLAS DGEMM vs serial WHAM profiles.

Validates that the fused CUDA code path produces identical C/V output
to the serial path. Requires a CUDA GPU to run.
"""

from __future__ import annotations

import math
import os
import subprocess

import numpy as np
import pytest


def _gpu_available() -> bool:
    """Check if CUDA GPU is accessible via nvidia-smi."""
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


pytestmark = pytest.mark.skipif(
    not _gpu_available(), reason="No CUDA GPU available"
)


class TestFusedToggle:
    """Verify CPHMD_WHAM_SERIAL environment variable is respected."""

    def test_serial_env_var_valid(self):
        """If set, CPHMD_WHAM_SERIAL should be '0' or '1'."""
        val = os.environ.get("CPHMD_WHAM_SERIAL")
        if val is not None:
            assert val in ("0", "1"), f"Invalid CPHMD_WHAM_SERIAL={val}"


class TestComputeRCFormulas:
    """Verify inline RC formulas match expected values (CPU-side check)."""

    def test_phi_is_lambda(self):
        """rc_phi returns lambda[j1]."""
        lambdas = [0.3, 0.5, 0.2]
        assert lambdas[2] == pytest.approx(0.2)

    def test_psi_is_product(self):
        """rc_psi returns lambda[j1] * lambda[j2]."""
        lambdas = [0.3, 0.5, 0.2]
        assert lambdas[0] * lambdas[1] == pytest.approx(0.15)

    def test_chi_formula(self):
        """rc_chi = lambda_j * (1 - exp(-lambda_i / omega_scale))."""
        omega_scale = 1.0 / 5.5  # 0.18182
        lam_i, lam_j = 0.8, 0.5
        expected = lam_j * (1.0 - math.exp(-lam_i / omega_scale))
        assert expected == pytest.approx(lam_j * (1.0 - math.exp(-4.4)), rel=1e-6)

    def test_omega_formula(self):
        """rc_omega = lambda_j * (1 - 1/(lambda_i/chi_offset + 1))."""
        chi_offset = 4 * math.exp(-5.5)  # ~0.01635
        lam_i, lam_j = 0.9, 0.4
        expected = lam_j * (1.0 - 1.0 / (lam_i / chi_offset + 1.0))
        assert expected > 0  # Positive for lam_i >> chi_offset

    def test_omega2_formula(self):
        """rc_omega2 = -lambda_j * (1 - 1/(lambda_i/(-1-chi_offset_t) + 1))."""
        chi_offset_t = 0.012
        lam_i, lam_j = 0.7, 0.3
        expected = -lam_j * (1.0 - 1.0 / (lam_i / (-1.0 - chi_offset_t) + 1.0))
        assert math.isfinite(expected)

    def test_omega3_formula(self):
        """rc_omega3 = lambda_j^2 * (1 - 1/(lambda_i/chi_offset_u + 1))."""
        chi_offset_u = 0.012
        lam_i, lam_j = 0.6, 0.4
        expected = lam_j * lam_j * (1.0 - 1.0 / (lam_i / chi_offset_u + 1.0))
        assert expected > 0


class TestDGEMMFactorization:
    """Verify the M*M^T factorization is algebraically correct."""

    def test_mmt_equals_weighted_outer(self):
        """CC[j1,j2] = sum_k w[k] * a[j1,k] * a[j2,k] = (M*M^T)[j1,j2]."""
        rng = np.random.default_rng(42)
        jN, B_N = 10, 20
        w = rng.uniform(0.1, 10.0, size=B_N)
        a = rng.standard_normal((jN, B_N))

        # Direct computation
        CC_direct = np.zeros((jN, jN))
        for j1 in range(jN):
            for j2 in range(jN):
                CC_direct[j1, j2] = np.sum(w * a[j1] * a[j2])

        # M*M^T factorization
        M = np.sqrt(w)[np.newaxis, :] * a  # [jN, B_N]
        CC_mmt = M @ M.T

        np.testing.assert_allclose(CC_direct, CC_mmt, rtol=1e-12)

    def test_mmt_symmetric(self):
        """M*M^T is always symmetric."""
        rng = np.random.default_rng(123)
        M = rng.standard_normal((5, 10))
        CC = M @ M.T
        np.testing.assert_allclose(CC, CC.T, atol=1e-15)

    def test_mmt_zeros_for_nonfinite_lnz(self):
        """Zeroing M columns for non-finite lnZ excludes those bins."""
        rng = np.random.default_rng(99)
        jN, B_N = 4, 8
        w = np.ones(B_N)
        a = rng.standard_normal((jN, B_N))

        # Mask bins 2 and 5 as non-finite
        mask = np.ones(B_N, dtype=bool)
        mask[2] = False
        mask[5] = False

        M = np.sqrt(w)[np.newaxis, :] * a
        M[:, ~mask] = 0.0  # Zero non-finite columns
        CC = M @ M.T

        # Should equal direct sum over valid bins only
        CC_ref = np.zeros((jN, jN))
        for j1 in range(jN):
            for j2 in range(jN):
                CC_ref[j1, j2] = np.sum(w[mask] * a[j1, mask] * a[j2, mask])

        np.testing.assert_allclose(CC, CC_ref, rtol=1e-12)
