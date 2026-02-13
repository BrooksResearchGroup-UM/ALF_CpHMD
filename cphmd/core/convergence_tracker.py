"""Convergence tracking for ALF simulations.

Manages EWBS state, population monitoring, phase transitions,
stop criteria, and diagnostic plot generation.
"""

from pathlib import Path

import numpy as np

from cphmd.core.phase_switcher import (
    EWBSState,
    StopCriteriaConfig,
    _per_site_ranges,
    calculate_populations,
    check_phase3_stop,
    check_phase_transition,
    ewbs_bottleneck_type,
    load_lambda_data,
    update_ewbs_state,
    write_populations_file,
)


class ConvergenceTracker:
    """Tracks convergence via EWBS, populations, and phase transitions.

    Receives ``config`` and ``state`` by reference — writes directly to
    ``state.phase``, ``state.converged``, ``state.stop_reason``, etc.

    Args:
        config: ALFConfig instance
        state: SimulationState instance (mutated in-place)
    """

    def __init__(self, config, state):
        self.config = config
        self.state = state

    # ------------------------------------------------------------------
    # EWBS
    # ------------------------------------------------------------------

    def update_ewbs(self) -> None:
        """Update EWBS (Energy-Weighted Bias Stability) from b/c/x/s.dat."""
        if self.state.ewbs_state is None:
            ewbs_path = Path("..") / "ewbs_state.json"
            if ewbs_path.exists():
                self.state.ewbs_state = EWBSState.load(ewbs_path)
            else:
                self.state.ewbs_state = EWBSState()

        try:
            b_dat = np.loadtxt("b.dat")
            c_dat = np.loadtxt("c.dat")
            x_dat = np.loadtxt("x.dat")
            s_dat = np.loadtxt("s.dat")
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"Warning: Cannot read bias files for EWBS: {e}")
            return

        ewbs_val = update_ewbs_state(
            self.state.ewbs_state, b_dat, c_dat, x_dat, s_dat
        )
        btn = ewbs_bottleneck_type(self.state.ewbs_state)
        st = self.state.ewbs_state
        print(f"EWBS: {ewbs_val:.4f} (b={st.ema_b:.4f} c={st.ema_c:.4f} "
              f"x={st.ema_x:.4f} s={st.ema_s:.4f}, bottleneck={btn})")
        try:
            self.state.ewbs_state.save(Path("..") / "ewbs_state.json")
        except OSError as e:
            print(f"Warning: Could not save EWBS state: {e}")

    # ------------------------------------------------------------------
    # Population monitoring
    # ------------------------------------------------------------------

    def compute_populations(self, nsubs) -> tuple:
        """Load lambda data and compute populations.

        Returns:
            (lambda_data, pop_data, pop_strict) — any may be None
        """
        data_dir = Path("data")
        lambda_data, _ = load_lambda_data(data_dir)

        if lambda_data is None:
            return None, None, None

        pop_data = calculate_populations(
            lambda_data, thresholds=(0.8, 0.985), nsubs=nsubs,
        )
        write_populations_file(Path("populations.dat"), pop_data)

        pop_strict = pop_data.get("pop_strict_norm", []) if pop_data else []

        if len(pop_strict) > 0:
            if nsubs is not None and len(nsubs) > 1:
                print("Populations (λ>0.985):")
                site_diffs = []
                for si, (start, end) in enumerate(_per_site_ranges(nsubs)):
                    s = pop_strict[start:end]
                    site_diffs.append((max(s) - min(s)) * 100)
                    site_str = ", ".join(f"{p:.1%}" for p in s)
                    print(f"  Site {si+1} ({nsubs[si]} subs): [{site_str}] "
                          f"diff={site_diffs[-1]:.1f}%")
                print(f"  Worst-site diff={max(site_diffs):.1f}%")
            else:
                pop_str = ", ".join(f"{p:.1%}" for p in pop_strict)
                frac_diff = (max(pop_strict) - min(pop_strict)) * 100
                print(f"Populations (λ>0.985): [{pop_str}] diff={frac_diff:.1f}%")

        return lambda_data, pop_data, pop_strict

    # ------------------------------------------------------------------
    # Forced initial lambdas
    # ------------------------------------------------------------------

    def detect_forced_lambdas(self, run_idx: int, pop_strict, nsubs,
                              trans_matrices, input_folder: Path) -> None:
        """Detect unsampled states and set biased Dirichlet initial lambdas."""
        nsubs_eff = nsubs if nsubs is not None else [len(pop_strict)]
        if self.state.patch_info is not None:
            sites = list(self.state.patch_info["site"].unique())
        else:
            sites = list(range(1, len(nsubs_eff) + 1))

        # Collect population history from last 5 analysis dirs
        pop_history = []
        for prev_r in range(max(run_idx - 4, 1), run_idx + 1):
            prev_pop_file = input_folder / f"analysis{prev_r}" / "pop_strict.dat"
            if prev_pop_file.exists():
                try:
                    prev_pops = np.loadtxt(prev_pop_file)
                    if len(prev_pops) == len(pop_strict):
                        pop_history.append(prev_pops)
                except Exception:
                    pass
        pop_history.append(np.array(pop_strict))

        avg_pops = np.mean(pop_history, axis=0)

        biased_alphas = {}
        for site_idx, (start, end) in enumerate(_per_site_ranges(nsubs_eff)):
            site_pops = avg_pops[start:end]
            low_mask = site_pops < 0.05
            if low_mask.any() and not low_mask.all():
                low_indices = np.where(low_mask)[0]
                worst = low_indices[np.argmin(site_pops[low_indices])]
                alpha = np.ones(len(site_pops))
                alpha_val = 10.0 if self.state.phase >= 2 else 3.0
                alpha[worst] = alpha_val
                biased_alphas[sites[site_idx]] = alpha.tolist()
                print(f"  Biased Dirichlet site {sites[site_idx]}: "
                      f"alpha[{worst}]={alpha_val:.0f} (avg pop "
                      f"{site_pops[worst]:.1%} over {len(pop_history)} runs)")

        if not biased_alphas and trans_matrices is not None:
            from cphmd.core.transitions import find_weakest_transitions
            weak_transitions = find_weakest_transitions(
                trans_matrices, nsubs_eff, threshold_count=5
            )
            for site_idx, (si, sj) in weak_transitions.items():
                alpha = np.ones(nsubs_eff[site_idx])
                alpha[si] = 2.0
                alpha[sj] = 2.0
                biased_alphas[sites[site_idx]] = alpha.tolist()
                print(f"  Transition Dirichlet site {sites[site_idx]}: "
                      f"alpha[{si}]=alpha[{sj}]=2 "
                      f"(weak {si}<->{sj} transition)")

        self.state.forced_initial_lambdas = biased_alphas or None

    # ------------------------------------------------------------------
    # Phase transition
    # ------------------------------------------------------------------

    def check_and_update_phase(
        self,
        run_idx: int,
        lambda_data,
        nsubs,
        cut_params: dict,
        trans_matrices,
        regenerate_g_imp_fn,
    ) -> None:
        """Check and handle phase transitions (RMSD or population-based).

        Args:
            run_idx: Current run number
            lambda_data: Lambda data array
            nsubs: Subsite counts
            cut_params: Cutoff params (for connectivity)
            trans_matrices: Transition matrices
            regenerate_g_imp_fn: Callable(old_phase, new_phase) for G_imp regen
        """
        if self.config.convergence_mode == "rmsd" and self.config.auto_phase_switch:
            self._check_rmsd_phase(run_idx, nsubs, regenerate_g_imp_fn)
        elif self.config.auto_phase_switch and lambda_data is not None:
            self._check_population_phase(
                run_idx, lambda_data, nsubs, cut_params, regenerate_g_imp_fn,
            )

    def _check_rmsd_phase(self, run_idx, nsubs, regenerate_g_imp_fn):
        """Handle RMSD-based convergence checking."""
        from .rmsd_convergence import (
            RMSDConvergenceConfig,
            RMSDState,
            check_rmsd_phase_transition,
            compute_site_rmsd,
        )
        if self.state.rmsd_state is None:
            self.state.rmsd_state = RMSDState()

        rmsd_cfg = RMSDConvergenceConfig()

        ref_run = run_idx - rmsd_cfg.lag
        ref_dir = Path(f"analysis{ref_run}")

        if not (ref_dir.is_dir() and (ref_dir / "multisite").is_dir()):
            print(f"RMSD: reference analysis{ref_run} not available yet (need {rmsd_cfg.lag} runs)")
            return

        analysis_dir = Path(f"analysis{run_idx}")
        site_rmsds, site_coverages = compute_site_rmsd(
            analysis_dir, ref_dir, list(nsubs),
        )
        self.state.rmsd_state.rmsd_history.append(site_rmsds)
        self.state.rmsd_state.coverage_history.append(site_coverages)
        self.state.rmsd_state.run_indices.append(run_idx)

        rmsd_label = ", ".join(
            f"s{i+1}={r:.2f}({c:.0%})"
            for i, (r, c) in enumerate(zip(site_rmsds, site_coverages))
        )
        print(f"RMSD (vs run {ref_run}): {rmsd_label}")

        new_phase, reason = check_rmsd_phase_transition(
            self.state.phase, self.state.rmsd_state, rmsd_cfg,
        )
        if new_phase != self.state.phase:
            print(f"PHASE TRANSITION: {self.state.phase} → {new_phase}")
            print(f"  {reason}")
            if new_phase == 2 and self.state.phase2_start_run is None:
                self.state.phase2_start_run = run_idx
            old_phase = self.state.phase
            self.state.phase = new_phase
            regenerate_g_imp_fn(old_phase, new_phase)
        else:
            print(f"Phase check: {reason}")

        self.state.rmsd_state.save(Path("..") / "rmsd_state.json")

    def _check_population_phase(self, run_idx, lambda_data, nsubs, cut_params,
                                regenerate_g_imp_fn):
        """Handle population-based phase transitions."""
        from .cphmd_params import get_delta_pKa_for_phase
        delta_pKa = get_delta_pKa_for_phase(self.state.phase)

        cphmd_kwargs = {}
        if (self.config.pH is not None and self.config.nreps > 3
                and self.state.patch_info is not None):
            from .cphmd_params import compute_all_site_parameters
            cphmd_params = compute_all_site_parameters(
                self.state.patch_info,
                self.config.temperature,
                self.config.pH,
            )
            cphmd_kwargs = {
                "data_dir": Path("data"),
                "patch_info": self.state.patch_info,
                "effective_pH": cphmd_params.effective_pH,
                "delta_pKa": delta_pKa,
                "nreps": self.config.nreps,
            }

        # Accumulate lambda data for Phase 1→2 check
        lambda_data_for_check = lambda_data
        min_phase1_runs = 20
        if self.state.phase == 1 and run_idx >= min_phase1_runs:
            window = 20
            accumulated = []
            for prev_r in range(max(1, run_idx - window + 1), run_idx + 1):
                prev_data_dir = Path("..") / f"analysis{prev_r}" / "data"
                if prev_data_dir.exists():
                    prev_data, _ = load_lambda_data(prev_data_dir)
                    if prev_data is not None:
                        accumulated.append(prev_data)
            if len(accumulated) >= 2:
                lam_thresh = self.config.phase_transition.lambda_threshold
                min_ms = self.config.phase_transition.min_multistate_runs_1to2
                multistate_count = 0
                for run_data in accumulated:
                    run_mask = run_data > lam_thresh
                    states_visited = int((run_mask.sum(axis=0) > 0).sum())
                    if states_visited >= 2:
                        multistate_count += 1
                if multistate_count >= min_ms:
                    lambda_data_for_check = np.vstack(accumulated)
                    print(f"  Phase 1 check: accumulated {len(accumulated)} runs "
                          f"({lambda_data_for_check.shape[0]} frames), "
                          f"{multistate_count} multi-state")
                else:
                    print(f"  Phase 1 check: only {multistate_count}/{len(accumulated)} "
                          f"runs show multi-state behavior (need {min_ms}) "
                          f"— using single-run data")

        p2_run_count = None
        if (self.state.phase == 2
                and self.state.phase2_start_run is not None):
            p2_run_count = run_idx - self.state.phase2_start_run

        new_phase, reason = check_phase_transition(
            self.state.phase,
            lambda_data_for_check,
            config=self.config.phase_transition,
            **cphmd_kwargs,
            nsubs=nsubs,
            connectivity=cut_params.get("connectivity"),
            phase2_run_count=p2_run_count,
            ewbs_state=self.state.ewbs_state,
        )

        if new_phase != self.state.phase:
            print(f"PHASE TRANSITION: {self.state.phase} → {new_phase}")
            print(f"  Reason: {reason}")
            if new_phase == 2 and self.state.phase2_start_run is None:
                self.state.phase2_start_run = run_idx
            old_phase = self.state.phase
            self.state.phase = new_phase
            regenerate_g_imp_fn(old_phase, new_phase)
        else:
            print(f"Phase check: {reason}")

    # ------------------------------------------------------------------
    # Stop criteria
    # ------------------------------------------------------------------

    def check_stop(self, run_idx: int, lambda_data, nsubs,
                   confirmation: bool) -> None:
        """Check convergence stop criteria (RMSD or population-based)."""
        if self.config.convergence_mode == "rmsd":
            self._check_rmsd_stop(run_idx, confirmation)
        elif (self.config.auto_stop and self.state.phase == 3
              and lambda_data is not None
              and self.config.convergence_mode == "population"):
            self._check_population_stop(run_idx, lambda_data, nsubs, confirmation)

    def _check_rmsd_stop(self, run_idx: int, confirmation: bool) -> None:
        """Check RMSD-based auto-stop in Phase 3."""
        if not (self.config.auto_stop and self.state.phase == 3):
            return
        if self.state.rmsd_state is None:
            return

        from .rmsd_convergence import RMSDConvergenceConfig, check_rmsd_stop
        rmsd_cfg = RMSDConvergenceConfig()

        should_stop, stop_reason = check_rmsd_stop(
            self.state.rmsd_state, rmsd_cfg,
        )
        if confirmation:
            if should_stop:
                self.state.converged = True
                self.state.stop_reason = stop_reason
                self.state.needs_confirmation = False
                print(f"\n{'='*60}")
                print(f"RMSD CONVERGENCE CONFIRMED at run {run_idx}")
                print(f"  {stop_reason}")
                print(f"{'='*60}\n")
                with open("CONVERGED", "w") as f:
                    f.write(f"RMSD converged at run {run_idx}\n")
                    f.write(f"{stop_reason}\n")
            else:
                self.state.needs_confirmation = False
                print(f"RMSD convergence NOT confirmed: {stop_reason}")
        elif should_stop:
            self.state.needs_confirmation = True
            print(f"\n{'='*60}")
            print(f"RMSD CONVERGENCE CANDIDATE at run {run_idx}")
            print(f"  {stop_reason}")
            print("  Triggering confirmation repeat...")
            print(f"{'='*60}\n")
        else:
            print(f"RMSD stop check: {stop_reason}")

    def _check_population_stop(self, run_idx: int, lambda_data, nsubs,
                               confirmation: bool) -> None:
        """Check population-based stop criteria in Phase 3."""
        timestep_fs = 4.0 if self.config.hmr else 2.0
        stop_config = StopCriteriaConfig(
            timestep_fs=timestep_fs,
            max_frac_diff=0.02,
        )

        # Build bias history
        bias_history = None
        bias_rows = []
        for prev_r in range(max(1, run_idx - stop_config.bias_window + 1),
                            run_idx + 1):
            bsum_file = Path("..") / f"analysis{prev_r}" / "b_sum.dat"
            if bsum_file.exists():
                try:
                    row = np.loadtxt(bsum_file).ravel()
                    bias_rows.append(row)
                except Exception:
                    pass
        if len(bias_rows) >= 2:
            bias_history = np.array(bias_rows)

        should_stop, stop_reason, stop_result = check_phase3_stop(
            lambda_data, stop_config,
            bias_history=bias_history, nsubs=nsubs,
            ewbs_state=self.state.ewbs_state,
        )

        if confirmation:
            if should_stop:
                self.state.converged = True
                self.state.stop_reason = stop_reason
                self.state.needs_confirmation = False
                print(f"\n{'='*60}")
                print(f"CONVERGENCE CONFIRMED at run {run_idx}")
                print(f"  {stop_reason}")
                print(f"  Fractions (λ>{stop_config.threshold_strict}): {stop_result.fractions}")
                print(f"  Fraction diff: {stop_result.frac_diff_pct:.2f}%")
                print(f"  Entropy (norm): {stop_result.entropy_normalized:.2f}")
                print(f"  Block variance: {stop_result.block_variance:.4f}")
                if stop_result.ewbs < float("inf"):
                    print(f"  EWBS: {stop_result.ewbs:.4f} "
                          f"(bottleneck={stop_result.ewbs_bottleneck})")
                print(f"  Bias stable: {stop_result.bias_stable} "
                      f"(std={stop_result.bias_rolling_std:.3f})")
                print(f"  Score: {stop_result.score:.4f}")
                print(f"{'='*60}\n")

                with open("CONVERGED", "w") as f:
                    f.write(f"Converged at run {run_idx}\n")
                    f.write(f"{stop_reason}\n")
                    f.write(f"Fractions: {stop_result.fractions}\n")
                    f.write(f"Entropy (norm): {stop_result.entropy_normalized:.2f}\n")
                    f.write(f"Block variance: {stop_result.block_variance:.4f}\n")
                    if stop_result.ewbs < float("inf"):
                        f.write(f"EWBS: {stop_result.ewbs:.4f} "
                                f"({stop_result.ewbs_bottleneck})\n")
                    f.write(f"Bias stable: {stop_result.bias_stable}\n")
                    f.write(f"Score: {stop_result.score:.4f}\n")
            else:
                self.state.needs_confirmation = False
                print(f"\n{'='*60}")
                print(f"CONVERGENCE NOT CONFIRMED at run {run_idx}")
                print(f"  With additional data: {stop_reason}")
                print("  Continuing optimization...")
                print(f"{'='*60}\n")
        else:
            if should_stop:
                self.state.needs_confirmation = True
                print(f"\n{'='*60}")
                print(f"CONVERGENCE CANDIDATE at run {run_idx}")
                print(f"  {stop_reason}")
                print(f"  Fractions: {stop_result.fractions}")
                print(f"  Fraction diff: {stop_result.frac_diff_pct:.2f}%")
                print(f"  Entropy (norm): {stop_result.entropy_normalized:.2f}")
                print(f"  Block variance: {stop_result.block_variance:.4f}")
                if stop_result.ewbs < float("inf"):
                    print(f"  EWBS: {stop_result.ewbs:.4f} "
                          f"(bottleneck={stop_result.ewbs_bottleneck})")
                print("  Triggering confirmation repeat...")
                print(f"{'='*60}\n")
            else:
                diag = (f"entropy={stop_result.entropy_normalized:.2f}, "
                        f"block_var={stop_result.block_variance:.4f}")
                if stop_result.ewbs < float("inf"):
                    diag += (f", ewbs={stop_result.ewbs:.4f}"
                             f"({stop_result.ewbs_bottleneck})")
                if stop_result.bias_rolling_std > 0:
                    diag += f", bias_std={stop_result.bias_rolling_std:.3f}"
                print(f"Stop check: {stop_reason} [{diag}]")

    # ------------------------------------------------------------------
    # Analysis plots
    # ------------------------------------------------------------------

    def generate_plots(self, run_idx: int, nsubs, msprof: int,
                       input_folder: Path) -> None:
        """Generate convergence and diagnostic plots after analysis."""
        plots_dir = input_folder / "plots"

        from cphmd.analysis.population_convergence import generate_population_plots
        generate_population_plots(
            input_folder=Path(".."),
            max_run=run_idx,
            output_dir=plots_dir,
            nsubs=nsubs,
        )

        try:
            if (Path("..") / f"analysis{run_idx}" / "multisite").is_dir():
                from cphmd.analysis.rmsd_convergence_plot import (
                    generate_rmsd_convergence_plots,
                )
                generate_rmsd_convergence_plots(
                    input_folder=Path(".."),
                    max_run=run_idx,
                    output_dir=plots_dir,
                    nsubs=list(nsubs),
                    rmsd_state=getattr(self.state, "rmsd_state", None),
                )
        except Exception as e:
            print(f"Warning: RMSD convergence plots failed: {e}")

        try:
            from cphmd.analysis.population_convergence import (
                _read_phases_from_runs,
            )
            from cphmd.analysis.rmsd_convergence_plot import (
                _collect_b_biases_from_dirs,
                plot_b_bias_convergence,
            )
            b_runs, b_values = _collect_b_biases_from_dirs(
                Path(".."), run_idx, list(nsubs),
            )
            if len(b_runs) >= 1:
                b_phases = _read_phases_from_runs(Path(".."), b_runs)
                plot_b_bias_convergence(
                    runs=b_runs,
                    b_values=b_values,
                    nsubs=list(nsubs),
                    output_path=plots_dir / "b_bias_convergence.png",
                    phases=b_phases,
                )
        except Exception as e:
            print(f"Warning: b-bias convergence plot failed: {e}")

        try:
            from cphmd.analysis.energy_profiles import plot_1d_profiles
            plot_1d_profiles(analysis_dir=Path.cwd(), nsubs=list(nsubs))
        except Exception as e:
            print(f"Warning: 1D energy profile plots failed: {e}")

        try:
            from cphmd.analysis.wham_profiles import plot_wham_profiles
            plot_wham_profiles(
                analysis_dir=Path.cwd(), nsubs=list(nsubs), msprof=msprof,
            )
        except Exception as e:
            print(f"Warning: WHAM profile plots failed: {e}")
