"""Tests for EWBS (Energy-Weighted Bias Stability) convergence metric.

Covers the EWBSState dataclass, compute_rms_changes, update_ewbs_state,
ewbs_bottleneck_type, and integration with check_stop_criteria and
check_phase_transition.
"""

from __future__ import annotations

import json
import math

import numpy as np

from cphmd.core.phase_switcher import (
    EWBSState,
    PhaseTransitionConfig,
    StopCriteriaConfig,
    check_phase_transition,
    check_stop_criteria,
    compute_rms_changes,
    ewbs_bottleneck_type,
    update_ewbs_state,
)
from tests.generators import generate_balanced_lambda

# ---------------------------------------------------------------------------
# compute_rms_changes
# ---------------------------------------------------------------------------


class TestComputeRmsChanges:
    """Tests for per-type RMS magnitude computation."""

    def test_known_values(self):
        """Hand-computable RMS: [3, 4] -> sqrt((9+16)/2) = 3.536."""
        b = np.array([3.0, 4.0])
        c = np.array([1.0])
        x = np.array([0.0])  # all-zero → 0.0
        s = np.array([5.0])

        rms_b, rms_c, rms_x, rms_s = compute_rms_changes(b, c, x, s)

        assert rms_x == 0.0, "All-zero should return 0.0"
        np.testing.assert_allclose(rms_b, np.sqrt((9 + 16) / 2), atol=1e-10)
        np.testing.assert_allclose(rms_c, 1.0, atol=1e-10)
        np.testing.assert_allclose(rms_s, 5.0, atol=1e-10)

    def test_all_zeros(self):
        """Phase 1 common case: x/s disabled → all zeros."""
        z = np.zeros(5)
        rms_b, rms_c, rms_x, rms_s = compute_rms_changes(z, z, z, z)
        assert (rms_b, rms_c, rms_x, rms_s) == (0.0, 0.0, 0.0, 0.0)

    def test_mixed_zero_nonzero(self):
        """Zeros are disabled params — RMS should only use nonzero entries."""
        b = np.array([0.0, 3.0, 0.0, 4.0])
        c = np.zeros(4)
        x = np.zeros(4)
        s = np.zeros(4)

        rms_b, _, _, _ = compute_rms_changes(b, c, x, s)
        # RMS over [3, 4] = sqrt((9+16)/2)
        np.testing.assert_allclose(rms_b, np.sqrt((9 + 16) / 2), atol=1e-10)

    def test_2d_matrix(self):
        """Coupling matrices (c, x, s) are 2D — verify ravel works."""
        c = np.array([[0.0, 0.5], [0.5, 0.0]])
        rms_b, rms_c, _, _ = compute_rms_changes(
            np.zeros(2), c, np.zeros((2, 2)), np.zeros((2, 2))
        )
        # nonzero entries: [0.5, 0.5], RMS = 0.5
        np.testing.assert_allclose(rms_c, 0.5, atol=1e-10)

    def test_nan_values_filtered(self):
        """NaN in bias data must be filtered out, not propagate."""
        b = np.array([1.0, float("nan"), 2.0])
        c = np.array([float("nan")])
        x = np.zeros(3)
        s = np.zeros(3)

        rms_b, rms_c, _, _ = compute_rms_changes(b, c, x, s)
        # b: finite nonzero = [1, 2], RMS = sqrt((1+4)/2) = sqrt(2.5)
        np.testing.assert_allclose(rms_b, np.sqrt(2.5), atol=1e-10)
        # c: only NaN → no valid entries → 0.0
        assert rms_c == 0.0

    def test_inf_values_filtered(self):
        """Inf in bias data must be filtered out."""
        b = np.array([float("inf"), 3.0, float("-inf")])
        rms_b, _, _, _ = compute_rms_changes(
            b, np.zeros(1), np.zeros(1), np.zeros(1)
        )
        # Only finite nonzero: [3.0], RMS = 3.0
        np.testing.assert_allclose(rms_b, 3.0, atol=1e-10)

    def test_single_element(self):
        """Single-element array: RMS = absolute value."""
        rms_b, _, _, _ = compute_rms_changes(
            np.array([-7.0]), np.zeros(1), np.zeros(1), np.zeros(1)
        )
        np.testing.assert_allclose(rms_b, 7.0, atol=1e-10)


# ---------------------------------------------------------------------------
# update_ewbs_state
# ---------------------------------------------------------------------------


class TestUpdateEwbsState:
    """Tests for EWMA-based EWBS state updates."""

    def test_first_update_initializes(self):
        """First update should set EMA directly (no smoothing)."""
        state = EWBSState()
        b = np.array([3.0, 4.0])
        c = np.array([1.0])
        x = np.zeros(4)
        s = np.zeros(4)

        ewbs = update_ewbs_state(state, b, c, x, s)

        expected_b = np.sqrt((9 + 16) / 2)
        np.testing.assert_allclose(state.ema_b, expected_b, atol=1e-10)
        np.testing.assert_allclose(state.ema_c, 1.0, atol=1e-10)
        assert state.ema_x == 0.0
        assert state.ema_s == 0.0
        assert ewbs == max(state.ema_b, state.ema_c, state.ema_x, state.ema_s)
        assert len(state.history) == 1

    def test_ewma_smoothing(self):
        """Second update applies alpha * new + (1-alpha) * old."""
        state = EWBSState(alpha=0.3)

        # First update
        update_ewbs_state(state, np.array([10.0]), np.zeros(1), np.zeros(1), np.zeros(1))
        assert state.ema_b == 10.0

        # Second update with smaller value
        update_ewbs_state(state, np.array([4.0]), np.zeros(1), np.zeros(1), np.zeros(1))
        expected = 0.3 * 4.0 + 0.7 * 10.0  # = 8.2
        np.testing.assert_allclose(state.ema_b, expected, atol=1e-10)

    def test_ewbs_is_max(self):
        """EWBS = max of all four EMA types."""
        state = EWBSState()
        # b=1, c=5, x=3, s=2
        update_ewbs_state(
            state,
            np.array([1.0]),
            np.array([5.0]),
            np.array([3.0]),
            np.array([2.0]),
        )
        assert state.ewbs == 5.0  # max(1, 5, 3, 2)

    def test_history_accumulates(self):
        """History grows by 1 per update, recording EWBS."""
        state = EWBSState()
        values = []
        for i in range(5):
            v = update_ewbs_state(
                state,
                np.array([float(i + 1)]),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
            )
            values.append(v)

        assert len(state.history) == 5
        np.testing.assert_allclose(state.history, values)

    def test_custom_alpha(self):
        """Custom alpha=0.5 changes smoothing behavior."""
        state = EWBSState(alpha=0.5)
        update_ewbs_state(state, np.array([10.0]), np.zeros(1), np.zeros(1), np.zeros(1))
        update_ewbs_state(state, np.array([2.0]), np.zeros(1), np.zeros(1), np.zeros(1))
        expected = 0.5 * 2.0 + 0.5 * 10.0  # = 6.0
        np.testing.assert_allclose(state.ema_b, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# ewbs_bottleneck_type
# ---------------------------------------------------------------------------


class TestEwbsBottleneckType:
    """Tests for bottleneck type identification."""

    def test_identifies_worst(self):
        """Should return the type with highest EMA."""
        state = EWBSState(ema_b=0.01, ema_c=0.02, ema_x=0.05, ema_s=0.03)
        assert ewbs_bottleneck_type(state) == "x"

    def test_b_dominant(self):
        state = EWBSState(ema_b=0.1, ema_c=0.01, ema_x=0.01, ema_s=0.01)
        assert ewbs_bottleneck_type(state) == "b"


# ---------------------------------------------------------------------------
# EWBSState save/load
# ---------------------------------------------------------------------------


class TestEwbsStatePersistence:
    """Tests for JSON save/load roundtrip."""

    def test_roundtrip(self, tmp_path):
        """Save then load should recover identical state."""
        state = EWBSState(
            ema_b=0.123, ema_c=0.456, ema_x=0.789, ema_s=0.012,
            ewbs=0.789, history=[0.5, 0.4, 0.3], alpha=0.25,
        )
        path = tmp_path / "ewbs_state.json"
        state.save(path)
        loaded = EWBSState.load(path)

        assert loaded.ema_b == state.ema_b
        assert loaded.ema_c == state.ema_c
        assert loaded.ema_x == state.ema_x
        assert loaded.ema_s == state.ema_s
        assert loaded.ewbs == state.ewbs
        assert loaded.history == state.history
        assert loaded.alpha == state.alpha

    def test_save_with_inf_default(self, tmp_path):
        """Default EWBSState (ewbs=inf) should save without crashing."""
        state = EWBSState()  # ewbs=float("inf")
        path = tmp_path / "ewbs_state.json"
        state.save(path)  # Should NOT raise ValueError

        # Verify JSON is valid
        with open(path) as f:
            data = json.load(f)
        assert data["ewbs"] is None  # inf serialized as null

    def test_load_restores_inf_from_null(self, tmp_path):
        """Loading null ewbs should restore float('inf')."""
        path = tmp_path / "ewbs_state.json"
        with open(path, "w") as f:
            json.dump({"ewbs": None, "history": [None, 0.5]}, f)

        loaded = EWBSState.load(path)
        assert math.isinf(loaded.ewbs)
        assert math.isinf(loaded.history[0])
        assert loaded.history[1] == 0.5

    def test_load_missing_keys(self, tmp_path):
        """Loading from JSON with missing keys should use defaults."""
        path = tmp_path / "ewbs_state.json"
        with open(path, "w") as f:
            json.dump({"ema_b": 0.5}, f)

        loaded = EWBSState.load(path)
        assert loaded.ema_b == 0.5
        assert loaded.ema_c == 0.0
        assert math.isinf(loaded.ewbs)
        assert loaded.alpha == 0.3


# ---------------------------------------------------------------------------
# check_stop_criteria with EWBS
# ---------------------------------------------------------------------------


class TestStopCriteriaEwbs:
    """Tests for EWBS integration in check_stop_criteria."""

    def _make_converged_ewbs(self, n=15, value=0.01):
        """Create an EWBSState that looks converged."""
        state = EWBSState(
            ema_b=value, ema_c=value, ema_x=value, ema_s=value,
            ewbs=value, history=[value] * n,
        )
        return state

    def _make_unconverged_ewbs(self, n=15, value=0.5):
        """Create an EWBSState with high EWBS."""
        state = EWBSState(
            ema_b=value, ema_c=0.01, ema_x=0.01, ema_s=0.01,
            ewbs=value, history=[value] * n,
        )
        return state

    def test_ewbs_blocks_stop(self, two_state_nsubs):
        """High EWBS should prevent stop even with perfect populations."""
        data = generate_balanced_lambda(two_state_nsubs, n_frames=200000, seed=50)
        config = StopCriteriaConfig(
            max_frac_diff=0.05, min_total_samples=10000, max_ewbs=0.03,
        )
        ewbs_state = self._make_unconverged_ewbs()

        result = check_stop_criteria(
            data, config, nsubs=two_state_nsubs, ewbs_state=ewbs_state,
        )
        assert result.should_stop is False
        assert any("ewbs" in r.lower() for r in result.reasons)

    def test_ewbs_passes_with_low_values(self, two_state_nsubs):
        """Low EWBS + balanced data + enough samples → should stop."""
        data = generate_balanced_lambda(two_state_nsubs, n_frames=200000, seed=51)
        config = StopCriteriaConfig(
            max_frac_diff=0.05, min_total_samples=10000, max_ewbs=0.03,
        )
        ewbs_state = self._make_converged_ewbs()

        result = check_stop_criteria(
            data, config, nsubs=two_state_nsubs, ewbs_state=ewbs_state,
        )
        assert result.should_stop is True, f"Should stop: {result.reasons}"

    def test_ewbs_none_backward_compat(self, two_state_nsubs):
        """Without ewbs_state, stop criteria should work normally."""
        data = generate_balanced_lambda(two_state_nsubs, n_frames=200000, seed=52)
        config = StopCriteriaConfig(
            max_frac_diff=0.05, min_total_samples=10000,
        )

        result = check_stop_criteria(data, config, nsubs=two_state_nsubs)
        assert result.should_stop is True
        assert result.ewbs == float("inf")
        assert result.ewbs_bottleneck == ""

    def test_ewbs_empty_history_blocks(self, two_state_nsubs):
        """Fresh EWBSState (empty history) should block stop — no data = not converged."""
        data = generate_balanced_lambda(two_state_nsubs, n_frames=200000, seed=53)
        config = StopCriteriaConfig(
            max_frac_diff=0.05, min_total_samples=10000, max_ewbs=0.03,
        )
        ewbs_state = EWBSState()  # empty history

        result = check_stop_criteria(
            data, config, nsubs=two_state_nsubs, ewbs_state=ewbs_state,
        )
        # Empty history → EWBS blocks stop (no data = unconverged)
        assert result.should_stop is False
        assert any("ewbs" in r.lower() for r in result.reasons)


# ---------------------------------------------------------------------------
# check_phase_transition with EWBS (2→3)
# ---------------------------------------------------------------------------


class TestPhaseTransitionEwbs:
    """Tests for EWBS gate in Phase 2→3 transition."""

    def test_ewbs_blocks_2to3(self, two_state_nsubs):
        """High EWBS should block Phase 2→3 even with perfect data."""
        data = generate_balanced_lambda(two_state_nsubs, n_frames=100000, seed=60)
        config = PhaseTransitionConfig(
            spread_2to3=0.2, min_hits_2to3=1000,
            ewbs_2to3=0.10, ewbs_2to3_window=5,
            min_phase2_runs=0,
        )
        ewbs_state = EWBSState(
            ema_b=0.5, ema_c=0.01, ema_x=0.01, ema_s=0.01,
            ewbs=0.5, history=[0.5] * 10,
        )

        new_phase, reason = check_phase_transition(
            current_phase=2, lambda_data=data, config=config,
            nsubs=two_state_nsubs, ewbs_state=ewbs_state,
            phase2_run_count=30,
        )
        assert new_phase == 2, f"EWBS should block: {reason}"
        assert "ewbs" in reason.lower()

    def test_ewbs_passes_2to3(self, two_state_nsubs):
        """Low EWBS with enough history should allow Phase 2→3."""
        data = generate_balanced_lambda(two_state_nsubs, n_frames=100000, seed=61)
        config = PhaseTransitionConfig(
            spread_2to3=0.2, min_hits_2to3=1000,
            ewbs_2to3=0.10, ewbs_2to3_window=5,
            min_phase2_runs=0,
        )
        ewbs_state = EWBSState(
            ema_b=0.01, ema_c=0.01, ema_x=0.01, ema_s=0.01,
            ewbs=0.01, history=[0.01] * 10,
        )

        new_phase, reason = check_phase_transition(
            current_phase=2, lambda_data=data, config=config,
            nsubs=two_state_nsubs, ewbs_state=ewbs_state,
            phase2_run_count=30,
        )
        assert new_phase == 3, f"Should advance: {reason}"

    def test_insufficient_history_blocks_2to3(self, two_state_nsubs):
        """Too few EWBS history entries should block 2→3."""
        data = generate_balanced_lambda(two_state_nsubs, n_frames=100000, seed=62)
        config = PhaseTransitionConfig(
            spread_2to3=0.2, min_hits_2to3=1000,
            ewbs_2to3=0.10, ewbs_2to3_window=5,
            min_phase2_runs=0,
        )
        # Only 3 history entries, but window requires 5
        ewbs_state = EWBSState(
            ema_b=0.01, ema_c=0.01, ema_x=0.01, ema_s=0.01,
            ewbs=0.01, history=[0.01] * 3,
        )

        new_phase, reason = check_phase_transition(
            current_phase=2, lambda_data=data, config=config,
            nsubs=two_state_nsubs, ewbs_state=ewbs_state,
            phase2_run_count=30,
        )
        assert new_phase == 2, f"Insufficient history should block: {reason}"
        assert "ewbs_history" in reason.lower()

    def test_empty_history_blocks_2to3(self, two_state_nsubs):
        """Empty EWBS history should block Phase 2→3 — no data = not stable."""
        data = generate_balanced_lambda(two_state_nsubs, n_frames=100000, seed=64)
        config = PhaseTransitionConfig(
            spread_2to3=0.2, min_hits_2to3=1000,
            ewbs_2to3=0.10, ewbs_2to3_window=5,
            min_phase2_runs=0,
        )
        ewbs_state = EWBSState()  # empty history

        new_phase, reason = check_phase_transition(
            current_phase=2, lambda_data=data, config=config,
            nsubs=two_state_nsubs, ewbs_state=ewbs_state,
            phase2_run_count=30,
        )
        assert new_phase == 2, f"Empty EWBS history should block: {reason}"
        assert "ewbs" in reason.lower()

    def test_ewbs_none_backward_compat_2to3(self, two_state_nsubs):
        """Without ewbs_state, Phase 2→3 should work normally."""
        data = generate_balanced_lambda(two_state_nsubs, n_frames=100000, seed=63)
        config = PhaseTransitionConfig(
            spread_2to3=0.2, min_hits_2to3=1000,
            min_phase2_runs=0,
        )

        new_phase, reason = check_phase_transition(
            current_phase=2, lambda_data=data, config=config,
            nsubs=two_state_nsubs, phase2_run_count=30,
        )
        assert new_phase == 3, f"Should advance without EWBS: {reason}"
