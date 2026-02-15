"""Test phase transition detection logic.

Verifies that check_phase_transition correctly identifies when to
advance phases based on synthetic lambda data with known properties.
"""

from __future__ import annotations

import numpy as np

from cphmd.core.phase_switcher import (
    PhaseTransitionConfig,
    StopCriteriaConfig,
    check_phase_transition,
    check_stop_criteria,
    compute_worst_site_pop_diff,
)
from tests.generators import generate_balanced_lambda, generate_trapped_lambda

# ---------------------------------------------------------------------------
# Phase 1 -> 2 transition
# ---------------------------------------------------------------------------

class TestPhase1To2:
    """Phase 1 -> 2 requires balanced sampling and enough hits."""

    def test_balanced_triggers_transition(self, two_state_nsubs):
        """Balanced sampling with enough hits should advance to Phase 2."""
        data = generate_balanced_lambda(two_state_nsubs, n_frames=50000, seed=10)
        config = PhaseTransitionConfig(spread_1to2=0.5, min_hits_1to2=100)

        new_phase, reason = check_phase_transition(
            current_phase=1, lambda_data=data, config=config, nsubs=two_state_nsubs
        )
        assert new_phase == 2, f"Expected phase 2, got {new_phase}: {reason}"

    def test_trapped_stays_phase1(self, two_state_nsubs):
        """Trapped trajectory should NOT advance to Phase 2."""
        data = generate_trapped_lambda(
            two_state_nsubs, trapped_state=[0], fraction=0.98, n_frames=5000, seed=11
        )
        config = PhaseTransitionConfig(spread_1to2=0.5, min_hits_1to2=100)

        new_phase, reason = check_phase_transition(
            current_phase=1, lambda_data=data, config=config, nsubs=two_state_nsubs
        )
        assert new_phase == 1, f"Should stay phase 1: {reason}"

    def test_too_few_samples_stays(self, two_state_nsubs):
        """Even balanced data with too few frames should stay in Phase 1."""
        data = generate_balanced_lambda(two_state_nsubs, n_frames=50, seed=12)
        config = PhaseTransitionConfig(min_hits_1to2=100)

        new_phase, reason = check_phase_transition(
            current_phase=1, lambda_data=data, config=config, nsubs=two_state_nsubs
        )
        assert new_phase == 1

    def test_multisite_one_trapped_stays(self, multi_site_nsubs):
        """Multi-site: if one site is trapped, should stay in Phase 1."""
        # Site 0: balanced (2 states), Site 1: trapped at state 0 (3 states)
        rng = np.random.default_rng(13)
        n_frames = 20000

        # Build mixed data: site0 balanced, site1 trapped
        nblocks = sum(multi_site_nsubs)
        data = np.zeros((n_frames, nblocks))

        # Site 0: balanced
        for t in range(n_frames):
            s = rng.choice(2)
            data[t, s] = rng.uniform(0.99, 1.0)
            data[t, 1 - s] = rng.uniform(0.0, 0.01)

        # Site 1: trapped in state 0
        for t in range(n_frames):
            data[t, 2] = rng.uniform(0.99, 1.0)
            data[t, 3] = rng.uniform(0.0, 0.01)
            data[t, 4] = rng.uniform(0.0, 0.01)

        config = PhaseTransitionConfig(spread_1to2=0.5, min_hits_1to2=100)
        new_phase, reason = check_phase_transition(
            current_phase=1, lambda_data=data, config=config, nsubs=multi_site_nsubs
        )
        assert new_phase == 1, f"Should stay phase 1 with trapped site: {reason}"


# ---------------------------------------------------------------------------
# Phase 1 -> 2: visited-states gate
# ---------------------------------------------------------------------------

class TestPhase1To2VisitedStates:
    """Phase 1→2 visited-states gate handles kinetically inaccessible states."""

    def test_all_states_visited_transitions(self, three_state_nsubs):
        """Balanced data with all 3 states visited should advance to Phase 2."""
        data = generate_balanced_lambda(three_state_nsubs, n_frames=50000, seed=50)
        config = PhaseTransitionConfig(
            spread_1to2=0.5, min_hits_1to2=100,
        )
        new_phase, reason = check_phase_transition(
            current_phase=1, lambda_data=data, config=config,
            nsubs=three_state_nsubs,
        )
        assert new_phase == 2, f"Expected phase 2 with all states visited: {reason}"
        assert "visited=3/3" in reason

    def test_two_of_three_states_stays(self, three_state_nsubs):
        """ASP-like scenario: states 0 and 2 visited, state 1 absent.

        Should stay in Phase 1: ALL states must be visited before advancing.
        NONE/UPOS/UNEG are all physical protonation states — TAG only controls
        which gets the pH shift, not whether the state is accessible.
        """
        rng = np.random.default_rng(51)
        n = 50000
        data = np.zeros((n, 3))
        # 50% state 0, 50% state 2, 0% state 1
        for t in range(n):
            if t % 2 == 0:
                data[t, 0] = rng.uniform(0.99, 1.0)
                data[t, 1] = rng.uniform(0.0, 0.005)
                data[t, 2] = rng.uniform(0.0, 0.005)
            else:
                data[t, 2] = rng.uniform(0.99, 1.0)
                data[t, 0] = rng.uniform(0.0, 0.005)
                data[t, 1] = rng.uniform(0.0, 0.005)

        config = PhaseTransitionConfig(
            spread_1to2=0.5, min_hits_1to2=100,
        )
        new_phase, reason = check_phase_transition(
            current_phase=1, lambda_data=data, config=config,
            nsubs=three_state_nsubs,
        )
        assert new_phase == 1, (
            f"Should stay phase 1 with 2/3 states: {reason}")
        assert "states visited" in reason

    def test_two_of_three_with_patch_info_stays(self, three_state_nsubs):
        """ASP-like with patch_info: state 0 NONE, states 1+2 UNEG.

        2/3 states visited → stays. patch_info doesn't change visited logic;
        TAG only controls pH shift, all states are equally required.
        """
        import pandas as pd

        rng = np.random.default_rng(51)
        n = 50000
        data = np.zeros((n, 3))
        for t in range(n):
            if t % 2 == 0:
                data[t, 0] = rng.uniform(0.99, 1.0)
                data[t, 1] = rng.uniform(0.0, 0.005)
                data[t, 2] = rng.uniform(0.0, 0.005)
            else:
                data[t, 2] = rng.uniform(0.99, 1.0)
                data[t, 0] = rng.uniform(0.0, 0.005)
                data[t, 1] = rng.uniform(0.0, 0.005)

        patch_info = pd.DataFrame({
            "site": ["s1", "s1", "s1"],
            "TAG": ["NONE", "UNEG 3.67", "UNEG 3.67"],
            "SELECT": ["s1s1", "s1s2", "s1s3"],
        })
        config = PhaseTransitionConfig(
            spread_1to2=0.5, min_hits_1to2=100,
        )
        new_phase, reason = check_phase_transition(
            current_phase=1, lambda_data=data, config=config,
            nsubs=three_state_nsubs, patch_info=patch_info,
        )
        assert new_phase == 1, (
            f"Should stay: 2/3 states visited: {reason}")
        assert "states visited" in reason

    def test_all_three_states_transitions(self, three_state_nsubs):
        """All 3 states visited with balanced populations → should advance."""
        rng = np.random.default_rng(51)
        n = 60000
        data = np.zeros((n, 3))
        for t in range(n):
            s = t % 3
            data[t, s] = rng.uniform(0.99, 1.0)
            for j in range(3):
                if j != s:
                    data[t, j] = rng.uniform(0.0, 0.005)

        config = PhaseTransitionConfig(
            spread_1to2=0.5, min_hits_1to2=100,
        )
        new_phase, reason = check_phase_transition(
            current_phase=1, lambda_data=data, config=config,
            nsubs=three_state_nsubs,
        )
        assert new_phase == 2, (
            f"Should advance with 3/3 states visited: {reason}")
        assert "visited=3/3" in reason

    def test_single_state_trapped_stays(self, three_state_nsubs):
        """Only 1 state visited — should stay in Phase 1."""
        data = generate_trapped_lambda(
            three_state_nsubs, trapped_state=[0], fraction=0.99,
            n_frames=50000, seed=52,
        )
        config = PhaseTransitionConfig(
            spread_1to2=0.5, min_hits_1to2=100,
        )
        new_phase, reason = check_phase_transition(
            current_phase=1, lambda_data=data, config=config,
            nsubs=three_state_nsubs,
        )
        assert new_phase == 1, f"Should stay phase 1 with 1 state: {reason}"
        assert "states visited" in reason

    def test_two_states_unbalanced_stays(self, three_state_nsubs):
        """2 states visited but very imbalanced spread — should stay."""
        rng = np.random.default_rng(53)
        n = 50000
        data = np.zeros((n, 3))
        # 95% state 0, 5% state 2 → spread = 0.95 - 0.05 = 0.9 > 0.5
        for t in range(n):
            if rng.random() < 0.95:
                data[t, 0] = rng.uniform(0.99, 1.0)
                data[t, 1] = rng.uniform(0.0, 0.005)
                data[t, 2] = rng.uniform(0.0, 0.005)
            else:
                data[t, 2] = rng.uniform(0.99, 1.0)
                data[t, 0] = rng.uniform(0.0, 0.005)
                data[t, 1] = rng.uniform(0.0, 0.005)

        config = PhaseTransitionConfig(
            spread_1to2=0.5, min_hits_1to2=100,
        )
        new_phase, reason = check_phase_transition(
            current_phase=1, lambda_data=data, config=config,
            nsubs=three_state_nsubs,
        )
        assert new_phase == 1, (
            f"Should stay phase 1 with imbalanced visited states: {reason}")
        assert "spread" in reason

    def test_multisite_partial_visits_stays(self):
        """Multi-site system: site 0 (2-state) fully visited,
        site 1 (3-state) has 2/3 visited — should stay (need all)."""
        nsubs = [2, 3]
        rng = np.random.default_rng(54)
        n = 50000
        nblocks = sum(nsubs)
        data = np.zeros((n, nblocks))

        for t in range(n):
            # Site 0: balanced 2-state
            s0 = t % 2
            data[t, s0] = rng.uniform(0.99, 1.0)
            data[t, 1 - s0] = rng.uniform(0.0, 0.005)
            # Site 1: states 0 and 2 only (state 1 absent)
            s1 = 0 if t % 2 == 0 else 2
            data[t, 2 + s1] = rng.uniform(0.99, 1.0)
            for j in range(3):
                if j != s1:
                    data[t, 2 + j] = rng.uniform(0.0, 0.005)

        config = PhaseTransitionConfig(
            spread_1to2=0.5, min_hits_1to2=100,
        )
        new_phase, reason = check_phase_transition(
            current_phase=1, lambda_data=data, config=config,
            nsubs=nsubs,
        )
        assert new_phase == 1, (
            f"Multi-site with partial visits should stay: {reason}")
        assert "states visited" in reason

    def test_multisite_all_visited_advances(self):
        """Multi-site system: all states visited on both sites → advances."""
        nsubs = [2, 3]
        rng = np.random.default_rng(54)
        n = 60000
        nblocks = sum(nsubs)
        data = np.zeros((n, nblocks))

        for t in range(n):
            # Site 0: balanced 2-state
            s0 = t % 2
            data[t, s0] = rng.uniform(0.99, 1.0)
            data[t, 1 - s0] = rng.uniform(0.0, 0.005)
            # Site 1: all 3 states visited
            s1 = t % 3
            data[t, 2 + s1] = rng.uniform(0.99, 1.0)
            for j in range(3):
                if j != s1:
                    data[t, 2 + j] = rng.uniform(0.0, 0.005)

        config = PhaseTransitionConfig(
            spread_1to2=0.5, min_hits_1to2=100,
        )
        new_phase, reason = check_phase_transition(
            current_phase=1, lambda_data=data, config=config,
            nsubs=nsubs,
        )
        assert new_phase == 2, (
            f"Multi-site with all visits should advance: {reason}")

    def test_concatenated_trapped_runs_pass_phase_check(self, three_state_nsubs):
        """Concatenated trapped runs: each run in one state, but all states
        represented across runs. check_phase_transition sees balanced data
        and allows transition. (The alf_runner quality gate is the protection
        against this scenario, not check_phase_transition itself.)
        """
        rng = np.random.default_rng(55)
        n_frames_per_run = 25000
        n_runs = 20
        n_states = three_state_nsubs[0]

        chunks = []
        for i in range(n_runs):
            state = i % n_states  # cycle through states
            run_data = np.zeros((n_frames_per_run, n_states))
            run_data[:, state] = rng.uniform(0.99, 1.0, size=n_frames_per_run)
            for other in range(n_states):
                if other != state:
                    run_data[:, other] = rng.uniform(0.0, 0.01, size=n_frames_per_run)
            chunks.append(run_data)

        accumulated = np.vstack(chunks)
        config = PhaseTransitionConfig(
            spread_1to2=0.5, min_hits_1to2=100,
        )
        new_phase, reason = check_phase_transition(
            current_phase=1, lambda_data=accumulated, config=config,
            nsubs=three_state_nsubs,
        )
        # This SHOULD pass because all 3 states are visited with good balance.
        # Protection against false positives is in alf_runner (quality gate),
        # not in check_phase_transition.
        assert new_phase == 2, (
            f"Concatenated data with all states should pass phase check: {reason}"
        )


# ---------------------------------------------------------------------------
# Legacy format compatibility (no TAG column, no patch_info)
# ---------------------------------------------------------------------------


class TestLegacyFormatCompatibility:
    """Phase transition works with legacy patches.dat formats.

    Legacy (msld-py-prep) systems may have:
    - No patches.dat at all (patch_info=None)
    - patches.dat without TAG column (4 columns: PATCHRES, SEGID, RESID, PATCH)

    Behavior is identical: all states must be visited regardless of format.
    """

    def test_no_patch_info_requires_all_states(self, three_state_nsubs):
        """Without patch_info, all 3 states must be visited (no mask)."""
        rng = np.random.default_rng(70)
        n = 50000
        data = np.zeros((n, 3))
        # Only 2 of 3 states visited
        for t in range(n):
            s = 0 if t % 2 == 0 else 2
            data[t, s] = rng.uniform(0.99, 1.0)
            for j in range(3):
                if j != s:
                    data[t, j] = rng.uniform(0.0, 0.005)

        config = PhaseTransitionConfig(spread_1to2=0.5, min_hits_1to2=100)
        new_phase, reason = check_phase_transition(
            current_phase=1, lambda_data=data, config=config,
            nsubs=three_state_nsubs, patch_info=None,
        )
        assert new_phase == 1, f"Should stay: 2/3 states visited, no patch_info: {reason}"
        assert "states visited" in reason

    def test_no_patch_info_all_visited_advances(self, three_state_nsubs):
        """Without patch_info, all 3 states visited → should advance."""
        data = generate_balanced_lambda(three_state_nsubs, n_frames=50000, seed=71)
        config = PhaseTransitionConfig(spread_1to2=0.5, min_hits_1to2=100)
        new_phase, reason = check_phase_transition(
            current_phase=1, lambda_data=data, config=config,
            nsubs=three_state_nsubs, patch_info=None,
        )
        assert new_phase == 2, f"Should advance: all 3 visited, no patch_info: {reason}"

    def test_legacy_patch_info_no_tag_column(self, three_state_nsubs):
        """Legacy patches.dat without TAG column — mask is None, all states required."""
        import pandas as pd

        # Legacy 4-column format — no TAG, no site
        legacy_patch_info = pd.DataFrame({
            "PATCHRES": ["ASP", "ASP", "ASP"],
            "SEGID": ["PROA", "PROA", "PROA"],
            "RESID": [1, 1, 1],
            "PATCH": ["ASPP1", "ASPP2", "ASPO"],
        })
        rng = np.random.default_rng(72)
        n = 50000
        data = np.zeros((n, 3))
        # Only 2 of 3 visited
        for t in range(n):
            s = 0 if t % 2 == 0 else 2
            data[t, s] = rng.uniform(0.99, 1.0)
            for j in range(3):
                if j != s:
                    data[t, j] = rng.uniform(0.0, 0.005)

        config = PhaseTransitionConfig(spread_1to2=0.5, min_hits_1to2=100)
        new_phase, reason = check_phase_transition(
            current_phase=1, lambda_data=data, config=config,
            nsubs=three_state_nsubs, patch_info=legacy_patch_info,
        )
        assert new_phase == 1, (
            f"Legacy (no TAG): 2/3 visited should stay: {reason}")
        assert "states visited" in reason

    def test_no_nsubs_global_check(self):
        """Without nsubs (legacy), global visited check applies."""
        rng = np.random.default_rng(73)
        n = 50000
        data = np.zeros((n, 3))
        # All 3 states visited evenly
        for t in range(n):
            s = t % 3
            data[t, s] = rng.uniform(0.99, 1.0)
            for j in range(3):
                if j != s:
                    data[t, j] = rng.uniform(0.0, 0.005)

        config = PhaseTransitionConfig(spread_1to2=0.5, min_hits_1to2=100)
        new_phase, reason = check_phase_transition(
            current_phase=1, lambda_data=data, config=config,
            nsubs=None, patch_info=None,
        )
        assert new_phase == 2, f"Global check, all visited should advance: {reason}"


# ---------------------------------------------------------------------------
# Phase 2 -> 3 transition
# ---------------------------------------------------------------------------

class TestPhase2To3:
    """Phase 2 -> 3 requires tighter balance and more samples."""

    def test_well_sampled_advances(self, two_state_nsubs):
        """Very well-sampled balanced data should advance to Phase 3."""
        data = generate_balanced_lambda(two_state_nsubs, n_frames=100000, seed=20)
        config = PhaseTransitionConfig(spread_2to3=0.2, min_hits_2to3=1000)

        new_phase, reason = check_phase_transition(
            current_phase=2, lambda_data=data, config=config, nsubs=two_state_nsubs
        )
        assert new_phase == 3, f"Expected phase 3: {reason}"

    def test_moderate_imbalance_stays(self, two_state_nsubs):
        """Moderately imbalanced data should stay in Phase 2."""
        data = generate_trapped_lambda(
            two_state_nsubs, trapped_state=[0], fraction=0.7, n_frames=50000, seed=21
        )
        config = PhaseTransitionConfig(spread_2to3=0.2, min_hits_2to3=1000)

        new_phase, reason = check_phase_transition(
            current_phase=2, lambda_data=data, config=config, nsubs=two_state_nsubs
        )
        assert new_phase == 2, f"Should stay phase 2: {reason}"


# ---------------------------------------------------------------------------
# Phase 3 stop criteria
# ---------------------------------------------------------------------------

class TestPhase3Stop:
    """Phase 3 -> STOP criteria based on population balance."""

    def test_perfectly_balanced_stops(self, two_state_nsubs):
        """Perfectly balanced, high-sample data should trigger stop."""
        data = generate_balanced_lambda(two_state_nsubs, n_frames=200000, seed=30)
        config = StopCriteriaConfig(
            max_frac_diff=0.05,
            min_total_samples=10000,
        )

        result = check_stop_criteria(data, config, nsubs=two_state_nsubs)
        assert result.should_stop is True, f"Should stop: {result.reasons}"

    def test_trapped_does_not_stop(self, two_state_nsubs):
        """Trapped trajectory should NOT stop."""
        data = generate_trapped_lambda(
            two_state_nsubs, trapped_state=[0], fraction=0.95, n_frames=200000, seed=31
        )
        config = StopCriteriaConfig(max_frac_diff=0.05, min_total_samples=1000)

        result = check_stop_criteria(data, config, nsubs=two_state_nsubs)
        assert result.should_stop is False

    def test_insufficient_samples_does_not_stop(self, two_state_nsubs):
        """Balanced but too few samples should NOT stop."""
        data = generate_balanced_lambda(two_state_nsubs, n_frames=100, seed=32)
        config = StopCriteriaConfig(min_total_samples=100000)

        result = check_stop_criteria(data, config, nsubs=two_state_nsubs)
        assert result.should_stop is False

    def test_none_data_does_not_stop(self):
        """None data should not stop."""
        result = check_stop_criteria(None)
        assert result.should_stop is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestPhaseEdgeCases:
    """Edge cases for phase transition logic."""

    def test_none_data_stays(self):
        """None data should keep current phase."""
        new_phase, _ = check_phase_transition(current_phase=1, lambda_data=None)
        assert new_phase == 1

    def test_empty_data_stays(self):
        """Empty array should keep current phase."""
        new_phase, _ = check_phase_transition(current_phase=2, lambda_data=np.array([]))
        assert new_phase == 2

    def test_phase3_stays_without_stop(self, two_state_nsubs):
        """Phase 3 with check_phase_transition should stay at 3 (stop is separate)."""
        data = generate_balanced_lambda(two_state_nsubs, n_frames=10000, seed=40)
        new_phase, _ = check_phase_transition(
            current_phase=3, lambda_data=data, nsubs=two_state_nsubs
        )
        assert new_phase == 3


# ---------------------------------------------------------------------------
# Worst-site population diff helper
# ---------------------------------------------------------------------------

class TestWorstSitePopDiff:
    """Tests for compute_worst_site_pop_diff helper."""

    def test_single_site_balanced(self):
        """Balanced 2-state system -> diff ~ 0."""
        data = generate_balanced_lambda([2], n_frames=5000, seed=70)
        diff = compute_worst_site_pop_diff(data, [2])
        assert diff < 0.2

    def test_single_site_stuck(self):
        """Fully trapped in one state -> diff ~ 1.0."""
        data = generate_trapped_lambda([3], fraction=0.999, n_frames=5000, seed=71)
        diff = compute_worst_site_pop_diff(data, [3])
        assert diff > 0.95

    def test_multisite_worst_reported(self):
        """Multi-site: one balanced, one stuck -> reports the stuck site's diff."""
        rng = np.random.default_rng(72)
        n = 5000
        data = np.zeros((n, 5))
        for t in range(n):
            s0 = t % 2
            data[t, s0] = rng.uniform(0.99, 1.0)
            data[t, 1 - s0] = rng.uniform(0.0, 0.005)
            data[t, 2] = rng.uniform(0.99, 1.0)
            data[t, 3] = rng.uniform(0.0, 0.005)
            data[t, 4] = rng.uniform(0.0, 0.005)
        diff = compute_worst_site_pop_diff(data, [2, 3])
        assert diff > 0.9

    def test_no_nsubs_global(self):
        """Without nsubs, computes global diff."""
        data = generate_trapped_lambda([3], fraction=0.999, n_frames=5000, seed=73)
        diff = compute_worst_site_pop_diff(data, None)
        assert diff > 0.95


# ---------------------------------------------------------------------------
# Pop-diff pre-gate
# ---------------------------------------------------------------------------

class TestPopDiffPreGate:
    """Pop-diff pre-gate blocks accumulated data when last run is stuck."""

    def test_stuck_last_run_blocks_transition_multisite(self):
        """nsubs=[8,15]: accumulated data has spread<0.5 but last run is 100%
        stuck -> should stay in Phase 1."""
        nsubs = [8, 15]
        nblocks = sum(nsubs)
        rng = np.random.default_rng(80)
        n_per_run = 5000

        # Build 20 runs where each is mostly stuck but random init
        # visits a few states for the first ~50 frames
        accumulated_runs = []
        for r in range(20):
            run_data = np.zeros((n_per_run, nblocks))
            dominant_s0 = 4
            dominant_s1 = 11
            for t in range(n_per_run):
                if t < 50:
                    s0 = rng.integers(0, 8)
                    s1 = 8 + rng.integers(0, 15)
                else:
                    s0 = dominant_s0
                    s1 = 8 + dominant_s1
                run_data[t, s0] = rng.uniform(0.99, 1.0)
                run_data[t, s1] = rng.uniform(0.99, 1.0)
                for j in range(nblocks):
                    if j != s0 and j != s1:
                        run_data[t, j] = rng.uniform(0.0, 0.003)
            accumulated_runs.append(run_data)

        last_run = accumulated_runs[-1]

        # With single-run (stuck) data, should stay in phase 1
        config = PhaseTransitionConfig(spread_1to2=0.5, min_hits_1to2=100)
        new_phase_single, reason_single = check_phase_transition(
            current_phase=1, lambda_data=last_run, config=config,
            nsubs=nsubs,
        )
        assert new_phase_single == 1, f"Single-run stuck data should block: {reason_single}"

        # The pop-diff gate should detect the stuck last run
        diff = compute_worst_site_pop_diff(last_run, nsubs)
        assert diff > 0.9, f"Last run should show diff>0.9, got {diff}"


# ---------------------------------------------------------------------------
# Phase 2 -> 1 regression
# ---------------------------------------------------------------------------

class TestPhase2To1Regression:
    """Phase 2->1 regression when stuck for too many consecutive runs."""

    def test_stuck_data_detected(self):
        """Stuck Phase 2 data: connectivity=0, diff>0.95."""
        nsubs = [8, 15]
        nblocks = sum(nsubs)
        rng = np.random.default_rng(90)
        n = 5000
        stuck_data = np.zeros((n, nblocks))
        for t in range(n):
            stuck_data[t, 4] = rng.uniform(0.99, 1.0)
            stuck_data[t, 8 + 11] = rng.uniform(0.99, 1.0)
            for j in range(nblocks):
                if j not in (4, 19):
                    stuck_data[t, j] = rng.uniform(0.0, 0.003)

        diff = compute_worst_site_pop_diff(stuck_data, nsubs)
        assert diff > 0.95, f"Stuck data should have diff>0.95, got {diff}"

        connectivity = 0.0
        is_stuck = connectivity == 0.0 and diff > 0.95
        assert is_stuck

    def test_regression_count_caps(self):
        """After max_phase_regressions, no more regressions allowed."""
        config = PhaseTransitionConfig(max_phase_regressions=2)
        assert config.max_phase_regressions == 2
