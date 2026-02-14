"""Bias analysis via WHAM/LMALF for ALF simulations.

Handles GPU WHAM, LMALF, and hybrid analysis with retry logic,
cutoff computation, and output validation.
"""

import contextlib
import shutil
from pathlib import Path
from typing import Literal

import numpy as np

from cphmd.core.free_energy import get_free_energy5

# Type aliases
AnalysisMethod = Literal["wham", "lmalf", "hybrid", "nonlinear"]
PhaseType = Literal[1, 2, 3]


class BiasAnalyzer:
    """Runs WHAM/LMALF bias analysis and computes adaptive cutoffs.

    Receives ``alf_info`` per-call (not at construction) to avoid
    implicit mutation coupling with the main orchestrator.

    Args:
        config: ALFConfig instance (read-only access to settings)
    """

    def __init__(self, config):
        self.config = config
        # Transient per-run data (set by prepare_data)
        self._wham_lambda = None
        self._wham_energy = None
        self._wham_gshift = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        alf_info: dict,
        run_idx: int,
        phase: PhaseType,
    ) -> int:
        """Compute WHAM inputs in memory from lambda/energy data.

        Sets self._wham_lambda, self._wham_energy, self._wham_gshift.

        Args:
            alf_info: ALF info dict
            run_idx: Current run number
            phase: Current simulation phase

        Returns:
            nf: Number of frames for WHAM
        """
        if phase == 1:
            im5 = max(run_idx - 15, 1)
            skipE = 10
        else:
            im5 = max(run_idx - 5, 1)
            skipE = 100

        from cphmd.core.alf_utils import compute_wham_inputs
        (
            self._wham_lambda,
            self._wham_energy,
            self._wham_gshift,
            nf,
        ) = compute_wham_inputs(alf_info, im5, run_idx, skipE=skipE)
        return nf

    # ------------------------------------------------------------------
    # Cutoff computation
    # ------------------------------------------------------------------

    def compute_cutoffs(
        self,
        phase: PhaseType,
        run_idx: int,
        coupling_scale: float,
        phase2_start_run: int | None,
        alf_info: dict,
        input_folder: Path,
    ) -> dict:
        """Compute cutoff parameters based on phase and run index.

        Args:
            phase: Current simulation phase
            run_idx: Current run number
            coupling_scale: sqrt(2/max_nsubs) scaling factor
            phase2_start_run: Run index when Phase 2 started
            alf_info: ALF info dict (for nsubs)
            input_folder: Path to input folder (for pop_strict files)

        Returns:
            dict of cutoff/calc parameters for get_free_energy5
        """
        if phase == 1:
            return self._phase1_cutoffs(run_idx)
        elif phase == 2:
            return self._phase2_cutoffs(run_idx, coupling_scale, phase2_start_run)
        else:
            return self._phase3_cutoffs(
                run_idx, coupling_scale, alf_info, input_folder
            )

    def _phase1_cutoffs(self, run_idx: int) -> dict:
        """Compute fixed staged cutoffs for Phase 1."""
        if run_idx < 20:
            cutb, cutc = 5.0, 20.0
        else:
            cutb, cutc = 2.5, 10.0
        cutx, cuts = 0.0, 0.0
        cutt, cutu = 0.0, 0.0
        print(f"  Phase 1 cutoffs: cutb={cutb:.1f} cutc={cutc:.1f} (fixed, run {run_idx})")

        cut_params = {
            "cutb": cutb, "cutc": cutc, "cutx": cutx, "cuts": cuts,
            "cutt": cutt, "cutu": cutu,
        }
        if self.config.no_b_bias or cutb == 0:
            cut_params["calc_phi"] = False
        if self.config.no_c_bias or cutc == 0:
            cut_params["calc_psi"] = False
        if self.config.no_x_bias or cutx == 0:
            cut_params["calc_chi"] = False
        if self.config.no_s_bias or cuts == 0:
            cut_params["calc_omega"] = False
        if self.config.no_t_bias or cutt == 0:
            cut_params["calc_omega2"] = False
        if self.config.no_u_bias or cutu == 0:
            cut_params["calc_omega3"] = False
        return cut_params

    def _phase2_cutoffs(self, run_idx: int, coupling_scale: float,
                        phase2_start_run: int | None) -> dict:
        """Compute Phase 2 warmup cutoffs (log-space decay over 20 runs)."""
        warmup_runs = 20
        p2_start = phase2_start_run or run_idx
        runs_in_p2 = max(run_idx - p2_start, 0)
        decay = min(runs_in_p2 / warmup_runs, 1.0)

        cutb_start, cutb_target = 1.0, 0.05
        cutc_start = 4.0 * coupling_scale
        cutc_target = 0.5 * coupling_scale

        cutb = cutb_start * (cutb_target / cutb_start) ** decay
        cutc = cutc_start * (cutc_target / cutc_start) ** decay

        cut_params = {
            "cutb": cutb, "cutc": cutc,
            "cutx": cutc, "cuts": cutc,
            "cutt": cutc if not self.config.no_t_bias else 0.0,
            "cutu": cutc if not self.config.no_u_bias else 0.0,
        }
        cut_params["calc_omega2"] = not self.config.no_t_bias
        cut_params["calc_omega3"] = not self.config.no_u_bias
        if runs_in_p2 < warmup_runs:
            print(f"  Phase 2 warmup ({runs_in_p2}/{warmup_runs}): "
                  f"cutb={cutb:.3f} cutc={cutc:.3f}")
        return cut_params

    def _phase3_cutoffs(self, run_idx: int, coupling_scale: float,
                        alf_info: dict, input_folder: Path) -> dict:
        """Compute Phase 3 cutoffs (tight, with recovery for skewed populations)."""
        prev_pop_file = input_folder / f"analysis{run_idx - 1}" / "pop_strict.dat"
        if prev_pop_file.exists():
            try:
                prev_pops = np.loadtxt(prev_pop_file)
                nsubs = alf_info.get("nsubs", [len(prev_pops)])
                col = 0
                for n in nsubs:
                    site_pops = prev_pops[col:col + n]
                    if len(site_pops) > 0 and site_pops.max() - site_pops.min() > 0.7:
                        print("  Phase 3 recovery: pop diff > 70% → using Phase 2 cutoffs")
                        return {
                            "cutb": 0.05,
                            "cutc": 0.5 * coupling_scale,
                            "cutx": 0.5 * coupling_scale,
                            "cuts": 0.5 * coupling_scale,
                            "cutt": 0.5 * coupling_scale if not self.config.no_t_bias else 0.0,
                            "cutu": 0.5 * coupling_scale if not self.config.no_u_bias else 0.0,
                            "calc_omega2": not self.config.no_t_bias,
                            "calc_omega3": not self.config.no_u_bias,
                        }
                    col += n
            except (ValueError, OSError) as e:
                print(f"  Phase 3 recovery: could not read pop_strict.dat: {e}")

        cut_params = {
            "cutb": 0.02,
            "cutc": 0.2 * coupling_scale,
            "cutx": 0.1 * coupling_scale,
            "cuts": 0.1 * coupling_scale,
            "cutt": 0.1 * coupling_scale if not self.config.no_t_bias else 0.0,
            "cutu": 0.1 * coupling_scale if not self.config.no_u_bias else 0.0,
        }
        cut_params["calc_omega2"] = not self.config.no_t_bias
        cut_params["calc_omega3"] = not self.config.no_u_bias
        return cut_params

    # ------------------------------------------------------------------
    # WHAM/LMALF execution
    # ------------------------------------------------------------------

    def run_with_retry(
        self,
        run_idx: int,
        nf: int,
        ms: int,
        msprof: int,
        cut_params: dict,
        alf_info: dict,
        phase: PhaseType,
        max_attempts: int = 3,
    ) -> tuple[bool, str]:
        """Run analysis with retry on failure or invalid output.

        Returns:
            Tuple of (success: bool, summary_message: str)
        """
        analysis_dir = Path.cwd()
        method = self.config.analysis_method
        log_file = analysis_dir / "analysis.log"

        with open(log_file, "a") as log_f:
            for attempt in range(max_attempts):
                try:
                    log_f.write(f"{method.upper()} attempt {attempt + 1}/{max_attempts}...\n")
                    log_f.flush()

                    with contextlib.redirect_stdout(log_f):
                        if method == "hybrid":
                            self._run_hybrid(nf, ms, msprof, cut_params, alf_info, phase)
                        elif method == "lmalf":
                            self._run_lmalf(nf, ms, msprof, cut_params, alf_info)
                        elif method == "nonlinear":
                            self._run_nonlinear(nf, ms, msprof, cut_params, alf_info)
                        else:
                            self._run_wham(nf, ms, msprof, cut_params, alf_info)

                    if self.is_output_invalid(analysis_dir, cut_params):
                        msg = f"{method.upper()} output invalid on attempt {attempt + 1}"
                        log_f.write(f"{msg}\n")
                        log_f.flush()
                        self.cleanup_invalid(analysis_dir)
                        continue

                    msg = f"{method.upper()} succeeded on attempt {attempt + 1}"
                    log_f.write(f"{msg}\n")
                    return True, msg

                except Exception as e:
                    msg = f"{method.upper()} attempt {attempt + 1} failed: {e}"
                    log_f.write(f"{msg}\n")
                    log_f.flush()
                    self.cleanup_invalid(analysis_dir)

                    if attempt == max_attempts - 1:
                        msg = f"{method.upper()} failed after {max_attempts} attempts"
                        log_f.write(f"{msg}\n")
                        return False, msg

        return False, f"{method.upper()} failed after {max_attempts} attempts"

    def _run_wham(self, nf: int, ms: int, msprof: int,
                  cut_params: dict, alf_info: dict) -> None:
        """Run WHAM analysis using bundled GPU library."""
        nsubs = alf_info["nsubs"]
        nblocks = alf_info["nblocks"]
        ntriangle = self.config.ntriangle

        if self._wham_lambda:
            from cphmd.wham import run_wham_from_memory
            run_wham_from_memory(
                lambda_arrays=self._wham_lambda,
                energy_matrix=self._wham_energy,
                nblocks=nblocks,
                nf=nf,
                temp=self.config.temperature,
                nts0=ms,
                nts1=msprof,
                use_gshift=self.config.use_gshift,
                nsubs=nsubs,
                g_imp_path="../G_imp",
                gshift_data=self._wham_gshift,
                output_dir=Path.cwd(),
                log_file="analysis.log",
                fnex=self.config.fnex,
                cutlsum=self.config.cutlsum,
                chi_offset=self.config.chi_offset,
                omega_decay=self.config.omega_decay,
                chi_offset_t=self.config.chi_offset_t,
                chi_offset_u=self.config.chi_offset_u,
                ntriangle=ntriangle,
            )
        else:
            from cphmd.wham import run_wham
            run_wham(
                analysis_dir=Path.cwd(),
                nf=nf,
                temp=self.config.temperature,
                nts0=ms,
                nts1=msprof,
                use_gshift=self.config.use_gshift,
                nsubs=nsubs,
                g_imp_path="../G_imp",
                log_file="analysis.log",
                fnex=self.config.fnex,
                cutlsum=self.config.cutlsum,
                chi_offset=self.config.chi_offset,
                omega_decay=self.config.omega_decay,
                chi_offset_t=self.config.chi_offset_t,
                chi_offset_u=self.config.chi_offset_u,
                ntriangle=ntriangle,
            )
        get_free_energy5(alf_info, ms=ms, msprof=msprof, ntriangle=ntriangle, **cut_params)

    def _invoke_lmalf(self, nf: int, ms: int, msprof: int,
                      alf_info: dict,
                      max_iter: int | None = None,
                      tolerance: float | None = None) -> bool:
        """Run LMALF optimization. Returns True if successful."""
        from cphmd.wham import run_lmalf_from_memory

        nsubs = alf_info["nsubs"]
        ntriangle = self.config.ntriangle

        if self._wham_lambda:
            lambda_combined = np.vstack(self._wham_lambda)
            print(f"[LMALF] {lambda_combined.shape[0]} frames, "
                  f"{lambda_combined.shape[1]} blocks (in-memory)")
        else:
            from cphmd.utils.lambda_io import find_lambda_files, read_lambda_values
            lambda_files = find_lambda_files(Path("data"))
            if not lambda_files:
                return False
            print(f"[LMALF] Found {len(lambda_files)} lambda files (file-based)")
            lambda_combined = np.vstack(
                [read_lambda_values(f) for f in lambda_files]
            )

        x_file, s_file = Path("x_prev.dat"), Path("s_prev.dat")
        x_prev = np.loadtxt(x_file) if x_file.exists() else None
        s_prev = np.loadtxt(s_file) if s_file.exists() else None

        if max_iter is None:
            max_iter = self.config.lmalf_max_iter
        if tolerance is None:
            tolerance = self.config.lmalf_tolerance

        print(f"[LMALF] Running optimization ({lambda_combined.shape[0]} frames)...")
        run_lmalf_from_memory(
            lambda_combined=lambda_combined,
            ensweight=None,
            nf=nf,
            temp=self.config.temperature,
            ms=ms,
            msprof=msprof,
            max_iter=max_iter,
            tolerance=tolerance,
            nsubs=nsubs,
            g_imp_path="../G_imp",
            x_prev=x_prev,
            s_prev=s_prev,
            output_dir=Path.cwd(),
            log_file="analysis.log",
            fnex=self.config.fnex,
            chi_offset=self.config.chi_offset,
            omega_decay=self.config.omega_decay,
            chi_offset_t=self.config.chi_offset_t,
            chi_offset_u=self.config.chi_offset_u,
            ntriangle=ntriangle,
        )
        return True

    def _run_lmalf(self, nf: int, ms: int, msprof: int,
                   cut_params: dict, alf_info: dict) -> None:
        """Run LMALF analysis using bundled GPU library."""
        from cphmd.core.alf_utils import get_free_energy_lm

        print("[LMALF] Starting analysis...", flush=True)
        if not self._invoke_lmalf(nf, ms, msprof, alf_info):
            raise FileNotFoundError(
                "No Lambda.*.*.parquet (or .dat) files found in data/"
            )
        print("[LMALF] LMALF optimization finished")

        out_file = Path("OUT.dat")
        if out_file.exists():
            out_data = np.loadtxt(out_file)
            if np.all(out_data == 0):
                print("[LMALF] Warning: LMALF produced all zeros - falling back to WHAM")
                self._run_wham(nf, ms, msprof, cut_params, alf_info)
                return

        ntriangle = self.config.ntriangle
        print("[LMALF] Converting OUT.dat to b/c/x/s.dat...")
        lm_keys = {"cutb", "cutc", "cutx", "cuts", "cutt", "cutu",
                   "cutc2", "cutx2", "cuts2"}
        lm_params = {k: v for k, v in cut_params.items() if k in lm_keys}
        get_free_energy_lm(alf_info, ms=ms, msprof=msprof, ntriangle=ntriangle, **lm_params)
        print("[LMALF] Analysis complete")

    def _invoke_nonlinear(self, nf: int, ms: int, msprof: int,
                          alf_info: dict,
                          max_iter: int | None = None,
                          tolerance: float | None = None) -> bool:
        """Run nonlinear L-BFGS optimization. Returns True if successful."""
        from cphmd.wham import run_nonlinear_from_memory

        nsubs = alf_info["nsubs"]
        ntriangle = self.config.ntriangle

        if self._wham_lambda:
            lambda_combined = np.vstack(self._wham_lambda)
            print(f"[NL] {lambda_combined.shape[0]} frames, "
                  f"{lambda_combined.shape[1]} blocks (in-memory)")
        else:
            from cphmd.utils.lambda_io import find_lambda_files, read_lambda_values
            lambda_files = find_lambda_files(Path("data"))
            if not lambda_files:
                return False
            print(f"[NL] Found {len(lambda_files)} lambda files (file-based)")
            lambda_combined = np.vstack(
                [read_lambda_values(f) for f in lambda_files]
            )

        x_file, s_file = Path("x_prev.dat"), Path("s_prev.dat")
        x_prev = np.loadtxt(x_file) if x_file.exists() else None
        s_prev = np.loadtxt(s_file) if s_file.exists() else None

        if max_iter is None:
            max_iter = self.config.lmalf_max_iter
        if tolerance is None:
            tolerance = self.config.lmalf_tolerance

        print(f"[NL] Running optimization ({lambda_combined.shape[0]} frames)...")
        run_nonlinear_from_memory(
            lambda_combined=lambda_combined,
            ensweight=None,
            nf=nf,
            temp=self.config.temperature,
            ms=ms,
            msprof=msprof,
            max_iter=max_iter,
            tolerance=tolerance,
            nsubs=nsubs,
            x_prev=x_prev,
            s_prev=s_prev,
            output_dir=Path.cwd(),
            log_file="analysis.log",
            fnex=self.config.fnex,
            chi_offset=self.config.chi_offset,
            omega_decay=self.config.omega_decay,
            chi_offset_t=self.config.chi_offset_t,
            chi_offset_u=self.config.chi_offset_u,
            ntriangle=ntriangle,
        )
        return True

    def _run_nonlinear(self, nf: int, ms: int, msprof: int,
                       cut_params: dict, alf_info: dict) -> None:
        """Run nonlinear L-BFGS analysis."""
        from cphmd.core.alf_utils import get_free_energy_lm

        print("[NL] Starting analysis...", flush=True)
        if not self._invoke_nonlinear(nf, ms, msprof, alf_info):
            raise FileNotFoundError(
                "No Lambda.*.*.parquet (or .dat) files found in data/"
            )
        print("[NL] Optimization finished")

        out_file = Path("OUT.dat")
        if out_file.exists():
            out_data = np.loadtxt(out_file)
            if np.all(out_data == 0):
                print("[NL] Warning: produced all zeros - falling back to WHAM")
                self._run_wham(nf, ms, msprof, cut_params, alf_info)
                return

        ntriangle = self.config.ntriangle
        print("[NL] Converting OUT.dat to b/c/x/s.dat...")
        lm_keys = {"cutb", "cutc", "cutx", "cuts", "cutt", "cutu",
                   "cutc2", "cutx2", "cuts2"}
        lm_params = {k: v for k, v in cut_params.items() if k in lm_keys}
        get_free_energy_lm(alf_info, ms=ms, msprof=msprof, ntriangle=ntriangle, **lm_params)
        print("[NL] Analysis complete")

    def _run_hybrid(self, nf: int, ms: int, msprof: int,
                    cut_params: dict, alf_info: dict, phase: PhaseType) -> None:
        """Run WHAM followed by short LMALF refinement."""
        self._run_wham(nf, ms, msprof, cut_params, alf_info)

        if phase != 1:
            return

        from cphmd.core.alf_utils import get_free_energy_lm

        print("[Hybrid] LMALF refinement (5 iterations)...")
        if not self._invoke_lmalf(nf, ms, msprof, alf_info, max_iter=5):
            print("[Hybrid] No lambda files found, skipping LMALF refinement")
            return

        out_file = Path("OUT.dat")
        if out_file.exists():
            out_data = np.loadtxt(out_file)
            if np.all(out_data == 0):
                print("[Hybrid] LMALF produced all zeros - keeping WHAM output")
                return

        ntriangle = self.config.ntriangle
        lm_keys = {"cutb", "cutc", "cutx", "cuts", "cutt", "cutu",
                   "cutc2", "cutx2", "cuts2"}
        lm_params = {k: v for k, v in cut_params.items() if k in lm_keys}
        get_free_energy_lm(alf_info, ms=ms, msprof=msprof, ntriangle=ntriangle, **lm_params)
        print("[Hybrid] LMALF refinement complete")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def is_output_invalid(analysis_dir: Path, cut_params: dict | None = None) -> bool:
        """Check for invalid WHAM output (all-zero, NaN, or Inf).

        Only checks files whose parameter types are active (non-zero cutoff).
        """
        file_cut_map = {
            'b.dat': 'cutb', 'c.dat': 'cutc',
            'x.dat': 'cutx', 's.dat': 'cuts',
            'b_sum.dat': 'cutb', 'c_sum.dat': 'cutc',
            'x_sum.dat': 'cutx', 's_sum.dat': 'cuts',
            't.dat': 'cutt', 'u.dat': 'cutu',
            't_sum.dat': 'cutt', 'u_sum.dat': 'cutu',
        }
        for fname, cut_key in file_cut_map.items():
            if cut_params and cut_params.get(cut_key, 1.0) == 0:
                continue
            fpath = analysis_dir / fname
            if not fpath.exists():
                continue
            try:
                data = np.loadtxt(fpath)
                if data.size == 0:
                    print(f"WHAM validation: {fname} is empty")
                    return True
                if np.all(data == 0):
                    print(f"WHAM validation: {fname} contains all zeros")
                    return True
                if np.any(np.isnan(data)):
                    print(f"WHAM validation: {fname} contains NaN")
                    return True
                if np.any(np.isinf(data)):
                    print(f"WHAM validation: {fname} contains Inf")
                    return True
            except Exception as e:
                print(f"WHAM validation: error reading {fname}: {e}")
                return True
        return False

    @staticmethod
    def cleanup_invalid(analysis_dir: Path) -> None:
        """Remove invalid WHAM output files for retry."""
        files_to_remove = [
            'b.dat', 'c.dat', 'x.dat', 's.dat', 't.dat', 'u.dat',
            'b_sum.dat', 'c_sum.dat', 'x_sum.dat', 's_sum.dat',
            't_sum.dat', 'u_sum.dat',
        ]
        for fname in files_to_remove:
            fpath = analysis_dir / fname
            if fpath.exists():
                fpath.unlink()
        multisite_dir = analysis_dir / 'multisite'
        if multisite_dir.exists():
            shutil.rmtree(multisite_dir)
