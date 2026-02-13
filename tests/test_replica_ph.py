"""Tests for pH-replica differentiation (cphmd_params)."""

import numpy as np
import pandas as pd
import pytest

from cphmd.core.cphmd_params import (
    adjust_tags_for_effective_ph,
    compute_all_site_parameters,
    compute_per_unit_shift,
    get_delta_pKa_for_phase,
    replica_pH,
)

KB = 0.0019872041  # kcal·mol⁻¹·K⁻¹


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lys_patch_info():
    """Single LYS site with 2 substates (protonated=NONE, deprotonated=UPOS)."""
    return pd.DataFrame({
        "site": [1, 1],
        "sub": [1, 2],
        "SEGID": ["PROA", "PROA"],
        "RESID": [3, 3],
        "PATCH": ["LYS", "LYSD"],
        "SELECT": ["s1s1", "s1s2"],
        "TAG": ["NONE", "UPOS 10.5"],
    })


@pytest.fixture
def asp_patch_info():
    """Single ASP site with 3 substates (protonated1=NONE, protonated2=NONE, deprotonated=UNEG)."""
    return pd.DataFrame({
        "site": [1, 1, 1],
        "sub": [1, 2, 3],
        "SEGID": ["PROA", "PROA", "PROA"],
        "RESID": [5, 5, 5],
        "PATCH": ["ASPP1", "ASPP2", "ASP"],
        "SELECT": ["s1s1", "s1s2", "s1s3"],
        "TAG": ["NONE", "NONE", "UNEG 3.9"],
    })


# ---------------------------------------------------------------------------
# replica_pH
# ---------------------------------------------------------------------------


class TestReplicaPH:
    """Test per-replica pH calculation."""

    def test_central_replica_gets_reference_pH(self):
        assert replica_pH(10.5, 1.0, replica_idx=2, ncentral=2) == 10.5

    def test_symmetric_fan_out(self):
        """5 replicas, ncentral=2, delta_pKa=1.0 → pH from 8.5 to 12.5."""
        pHs = [replica_pH(10.5, 1.0, k, ncentral=2) for k in range(5)]
        assert pHs == pytest.approx([8.5, 9.5, 10.5, 11.5, 12.5])

    def test_2_replicas_ncentral_1(self):
        """2 replicas, ncentral=1 → replica 0 at pH-1, replica 1 at pH."""
        pHs = [replica_pH(10.5, 1.0, k, ncentral=1) for k in range(2)]
        assert pHs == pytest.approx([9.5, 10.5])

    def test_phase2_smaller_spacing(self):
        """Phase 2 uses delta_pKa=0.5."""
        pHs = [replica_pH(10.5, 0.5, k, ncentral=2) for k in range(5)]
        assert pHs == pytest.approx([9.5, 10.0, 10.5, 11.0, 11.5])


# ---------------------------------------------------------------------------
# compute_per_unit_shift
# ---------------------------------------------------------------------------


class TestPerUnitShift:
    """Test per-unit pH shift computation."""

    def test_lys_upos_sign(self, lys_patch_info):
        """UPOS tag → positive shift; NONE → zero."""
        cphmd = compute_all_site_parameters(lys_patch_info, 298.15)
        b_shift, _ = compute_per_unit_shift(cphmd, lys_patch_info, delta_pKa=1.0)

        kTln10 = KB * 298.15 * np.log(10)
        assert b_shift[0] == pytest.approx(0.0)          # NONE: no shift
        assert b_shift[1] == pytest.approx(+kTln10)       # UPOS: +kTln10 × delta_pKa

    def test_asp_uneg_sign(self, asp_patch_info):
        """UNEG tag → negative shift."""
        cphmd = compute_all_site_parameters(asp_patch_info, 298.15)
        b_shift, _ = compute_per_unit_shift(cphmd, asp_patch_info, delta_pKa=1.0)

        kTln10 = KB * 298.15 * np.log(10)
        assert b_shift[0] == pytest.approx(0.0)          # NONE
        assert b_shift[1] == pytest.approx(0.0)          # NONE
        assert b_shift[2] == pytest.approx(-kTln10)       # UNEG: -kTln10 × delta_pKa

    def test_scales_with_delta_pka(self, lys_patch_info):
        """b_shift ∝ delta_pKa."""
        cphmd = compute_all_site_parameters(lys_patch_info, 298.15)
        b1, _ = compute_per_unit_shift(cphmd, lys_patch_info, delta_pKa=1.0)
        b2, _ = compute_per_unit_shift(cphmd, lys_patch_info, delta_pKa=0.5)
        assert b2[1] == pytest.approx(b1[1] * 0.5)

    def test_fix_shift_is_pka_correction(self, lys_patch_info):
        """b_fix_shift reflects subsite pKa shift, not replica shift."""
        cphmd = compute_all_site_parameters(lys_patch_info, 298.15)
        _, b_fix = compute_per_unit_shift(cphmd, lys_patch_info, delta_pKa=1.0)

        # For LYS: pH₀ = pKa = 10.5, so subsite shift = pH₀ - pKa = 0.0
        assert b_fix[0] == pytest.approx(0.0)  # NONE
        assert b_fix[1] == pytest.approx(0.0)  # UPOS with pKa_shift=0


# ---------------------------------------------------------------------------
# Analysis accounting: b_shift × (k - ncentral) reconstructs total replica shift
# ---------------------------------------------------------------------------


class TestAnalysisAccounting:
    """Verify that per-unit b_shift × (k - ncentral) gives the correct total shift."""

    def test_total_shift_matches_ph_offset(self, lys_patch_info):
        """Total shift for each replica matches ±kTln10 × pH offset."""
        cphmd = compute_all_site_parameters(lys_patch_info, 298.15)
        kTln10 = cphmd.kTln10
        delta_pKa = 1.0
        ncentral = 2
        nreps = 5

        b_shift, _ = compute_per_unit_shift(cphmd, lys_patch_info, delta_pKa)

        for k in range(nreps):
            pH_offset = delta_pKa * (k - ncentral)
            # UPOS block (index 1): expected total = +kTln10 × pH_offset
            total = b_shift[1] * (k - ncentral)
            expected = +kTln10 * pH_offset
            assert total == pytest.approx(expected), f"replica {k}"


# ---------------------------------------------------------------------------
# get_delta_pKa_for_phase
# ---------------------------------------------------------------------------


class TestDeltaPKaForPhase:
    def test_phase_values(self):
        assert get_delta_pKa_for_phase(1) == 1.0
        assert get_delta_pKa_for_phase(2) == 0.5
        assert get_delta_pKa_for_phase(3) == 0.25


# ---------------------------------------------------------------------------
# adjust_tags_for_effective_ph
# ---------------------------------------------------------------------------


@pytest.fixture
def multisite_patch_info():
    """Two sites: LYS (pKa=10.5) + ASP (pKa=3.9) → effective_pH=0.0."""
    return pd.DataFrame({
        "site": [1, 1, 2, 2, 2],
        "sub": [1, 2, 1, 2, 3],
        "SEGID": ["PROA"] * 5,
        "RESID": [3, 3, 5, 5, 5],
        "PATCH": ["LYS", "LYSD", "ASPP1", "ASPP2", "ASP"],
        "SELECT": ["s1s1", "s1s2", "s2s1", "s2s2", "s2s3"],
        "TAG": ["NONE", "UPOS 10.5", "NONE", "NONE", "UNEG 3.9"],
    })


class TestAdjustTagsForEffectivePH:
    """Test TAG pKa adjustment for multi-site systems."""

    def test_single_site_is_noop(self, lys_patch_info):
        """Single site: effective_pH == pH₀ → no adjustment."""
        cphmd = compute_all_site_parameters(lys_patch_info, 298.15)
        adjusted = adjust_tags_for_effective_ph(lys_patch_info, cphmd)

        # TAG values should be unchanged
        assert adjusted.at[0, "TAG"] == "NONE"
        assert adjusted.at[1, "TAG"] == "UPOS 10.50"

    def test_multisite_adjusts_tags(self, multisite_patch_info):
        """Multi-site: effective_pH=0.0, TAGs shifted to be relative to pH=0."""
        cphmd = compute_all_site_parameters(multisite_patch_info, 298.15)
        assert cphmd.effective_pH == pytest.approx(0.0)

        adjusted = adjust_tags_for_effective_ph(multisite_patch_info, cphmd)

        # LYS UPOS: new_pKa = 10.5 + (0.0 - 10.5) = 0.0
        assert adjusted.at[1, "TAG"] == "UPOS 0.00"
        # ASP UNEG: new_pKa = 3.9 + (0.0 - 3.9) = 0.0
        assert adjusted.at[4, "TAG"] == "UNEG 0.00"
        # NONE tags untouched
        assert adjusted.at[0, "TAG"] == "NONE"
        assert adjusted.at[2, "TAG"] == "NONE"

    def test_does_not_mutate_original(self, multisite_patch_info):
        """Original DataFrame must not be modified."""
        cphmd = compute_all_site_parameters(multisite_patch_info, 298.15)
        original_tags = multisite_patch_info["TAG"].tolist()

        adjust_tags_for_effective_ph(multisite_patch_info, cphmd)

        assert multisite_patch_info["TAG"].tolist() == original_tags

    def test_asp_single_site_noop(self, asp_patch_info):
        """Single ASP site: effective_pH == pKa → no shift."""
        cphmd = compute_all_site_parameters(asp_patch_info, 298.15)
        adjusted = adjust_tags_for_effective_ph(asp_patch_info, cphmd)

        assert adjusted.at[2, "TAG"] == "UNEG 3.90"
