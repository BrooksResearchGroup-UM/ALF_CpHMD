"""Bias analysis via WHAM/LMALF for ALF simulations.

Handles GPU WHAM, LMALF, and hybrid analysis with retry logic,
cutoff computation, and output validation.
"""

import contextlib
import logging
import shutil
from pathlib import Path
from typing import Literal

import numpy as np

from cphmd.core.free_energy import get_free_energy5

logger = logging.getLogger(__name__)

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
        # Fused packed data (set by prepare_data, preferred over lambda/energy)
        self._packed_D = None
        self._packed_sim_indices = None
        self._packed_frame_counts = None
        self._packed_total_frames = 0
        self._packed_nf = 0
        # MPI state (set by caller for distributed analysis)
        self._comm = None
        self._rank = 0
        self._nranks = 1
        self._gpu_id = 0

    def set_mpi(self, comm, rank: int, nranks: int, gpu_id: int = 0) -> None:
        """Set MPI communicator for distributed WHAM analysis."""
        self._comm = comm
        self._rank = rank
        self._nranks = nranks
        self._gpu_id = gpu_id

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        alf_info: dict,
        run_idx: int,
        phase: PhaseType,
    ) -> int:
        """Compute WHAM inputs as pre-packed D_h data (fused path).

        Uses compute_packed_wham_data[_distributed] to build the flat D_h
        array directly, avoiding the intermediate energy_matrix that doubles
        peak memory. Also keeps self._wham_lambda for the LMALF path.

        Args:
            alf_info: ALF info dict
            run_idx: Current run number
            phase: Current simulation phase

        Returns:
            nf: Number of frames for WHAM
        """
        # Number of analysis windows: config override or phase defaults (7, 5, 3)
        defaults = {1: 7, 2: 5, 3: 3}
        aw = getattr(self.config, "analysis_window", None)
        if aw is None:
            n_win = defaults[phase]
        elif isinstance(aw, list):
            n_win = aw[phase - 1]  # [phase1, phase2, phase3]
        else:
            n_win = int(aw)

        im5 = max(run_idx - (n_win - 1), 1)
        ask = getattr(self.config, "analysis_skip", 1)
        if isinstance(ask, list):
            skipE = ask[phase - 1]
        else:
            skipE = int(ask) if ask else 1

        if self._nranks > 1 and self._comm is not None:
            from cphmd.core.alf_utils import compute_packed_wham_data_distributed
            (
                self._packed_D,
                self._packed_sim_indices,
                self._packed_frame_counts,
                self._wham_gshift,
                nf,
                self._packed_total_frames,
            ) = compute_packed_wham_data_distributed(
                alf_info, im5, run_idx, skipE=skipE,
                comm=self._comm, rank=self._rank, nranks=self._nranks,
            )
        else:
            from cphmd.core.alf_utils import compute_packed_wham_data
            (
                self._packed_D,
                self._packed_sim_indices,
                self._packed_frame_counts,
                self._wham_gshift,
                nf,
                self._packed_total_frames,
            ) = compute_packed_wham_data(alf_info, im5, run_idx, skipE=skipE)

        self._packed_nf = nf
        self._wham_energy = None  # No longer stored

        # Extract lambda views from packed D for LMALF path (np.vstack at line ~448).
        # Views instead of .copy() — avoids duplicating total_frames × NL × 8 bytes.
        # Safe because _packed_D and _wham_lambda are freed together (alf_runner line ~1888).
        if nf > 0 and self._packed_D.size > 0:
            NL = alf_info["nblocks"]
            ndim = NL + nf + 3
            D_2d = self._packed_D.reshape(-1, ndim)
            self._wham_lambda = []
            offset = 0
            for i in range(nf):
                n_i = self._packed_frame_counts[i]
                self._wham_lambda.append(D_2d[offset:offset + n_i, 1:1 + NL])
                offset += n_i
            self._D_2d_ref = D_2d  # prevent reshape view from being GC'd
        else:
            self._wham_lambda = None
            self._D_2d_ref = None

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
        xs_enabled: bool = False,
        nreps: int = 1,
    ) -> dict:
        """Compute cutoff parameters based on phase and run index.

        Args:
            phase: Current simulation phase
            run_idx: Current run number
            coupling_scale: sqrt(2/max_nsubs) scaling factor
            phase2_start_run: Run index when Phase 2 started
            alf_info: ALF info dict (for nsubs)
            input_folder: Path to input folder (for pop_strict files)
            xs_enabled: Whether Phase 1 x/s coverage gate is satisfied
            nreps: Number of pH replicas (>1 = replica mode with loose cutoffs)

        Returns:
            dict of cutoff/calc parameters for get_free_energy5
        """
        # Check for manual per-phase cutoff overrides
        override = {1: self.config.phase1_cutoffs,
                    2: self.config.phase2_cutoffs,
                    3: self.config.phase3_cutoffs}.get(phase)

        if phase == 1:
            cut_params = self._phase1_cutoffs(run_idx, xs_enabled=xs_enabled)
        elif phase == 2:
            cut_params = self._phase2_cutoffs(
                run_idx, coupling_scale, phase2_start_run, nreps=nreps)
        else:
            cut_params = self._phase3_cutoffs(
                run_idx, coupling_scale, alf_info, input_folder, nreps=nreps
            )

        if override:
            for key in ("cutb", "cutc", "cutx", "cuts"):
                if key in override:
                    cut_params[key] = float(override[key])
            print(f"  Manual cutoff overrides applied for Phase {phase}: "
                  + ", ".join(f"{k}={v}" for k, v in override.items()))


        return cut_params

    def _phase1_cutoffs(self, run_idx: int, xs_enabled: bool = False) -> dict:
        """Compute fixed staged cutoffs for Phase 1.

        Note: cutx/cuts are always enabled from the start (no coverage gate).
        """
        if run_idx < 20:
            cutb, cutc = 5.0, 20.0
        else:
            cutb, cutc = 2.5, 10.0
        cutx = self.config.phase1_xs_cutoff
        cuts = self.config.phase1_xs_cutoff
        cutt, cutu = 0.0, 0.0
        print(f"  Phase 1 cutoffs: cutb={cutb:.1f} cutc={cutc:.1f}"
              f" cutx=cuts={cutx:.1f} (fixed, run {run_idx})")

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
                        phase2_start_run: int | None,
                        nreps: int = 1) -> dict:
        """Compute Phase 2 cutoffs.

        Single replica: warmup with coupling_scale (log-space decay over 20 runs).
        Multi-replica: fixed loose cutoffs (no coupling_scale needed).
        """
        if nreps > 1:
            cut_params = {
                "cutb": 2.0, "cutc": 8.0, "cutx": 2.0, "cuts": 1.0,
                "cutt": 1.0 if not self.config.no_t_bias else 0.0,
                "cutu": 1.0 if not self.config.no_u_bias else 0.0,
            }
            cut_params["calc_omega2"] = not self.config.no_t_bias
            cut_params["calc_omega3"] = not self.config.no_u_bias
            print("  Phase 2 cutoffs (replica mode): "
                  "cutb=2.0 cutc=8.0 cutx=2.0 cuts=1.0")
            return cut_params

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
                        alf_info: dict, input_folder: Path,
                        nreps: int = 1) -> dict:
        """Compute Phase 3 cutoffs.

        Single replica: tight cutoffs with recovery for skewed populations.
        Multi-replica: fixed cutoffs (no coupling_scale, no recovery).
        """
        if nreps > 1:
            cut_params = {
                "cutb": 0.5, "cutc": 2.0, "cutx": 0.5, "cuts": 0.25,
                "cutt": 0.25 if not self.config.no_t_bias else 0.0,
                "cutu": 0.25 if not self.config.no_u_bias else 0.0,
            }
            cut_params["calc_omega2"] = not self.config.no_t_bias
            cut_params["calc_omega3"] = not self.config.no_u_bias
            print("  Phase 3 cutoffs (replica mode): "
                  "cutb=0.5 cutc=2.0 cutx=0.5 cuts=0.25")
            return cut_params
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

        MPI-aware: all ranks participate in WHAM calls (MPI collectives
        require all ranks). File I/O and stdout redirect are rank-0 only.
        Retry/validation decisions are synchronized across ranks.

        Returns:
            Tuple of (success: bool, summary_message: str)
        """
        analysis_dir = Path.cwd()
        method = self.config.analysis_method
        log_file = analysis_dir / "analysis.log"
        is_distributed = self._nranks > 1 and self._comm is not None
        is_rank0 = self._rank == 0
        if is_distributed and method != "wham":
            raise ValueError(
                f"analysis_method='{method}' is not supported in distributed MPI mode. "
                "Use analysis_method='wham' or run single-rank analysis."
            )

        for attempt in range(max_attempts):
            # --- Log attempt (rank 0 only) ---
            if is_rank0:
                with open(log_file, "a") as log_f:
                    log_f.write(
                        f"{method.upper()} attempt {attempt + 1}/{max_attempts}...\n"
                    )

            # --- Execute analysis ---
            error = None
            try:
                if is_distributed:
                    # Distributed: ALL ranks must enter _run_wham together
                    # (MPI collectives inside require all ranks to participate).
                    # Log redirect must not prevent entry — open log defensively.
                    _log_f = None
                    _redirect_ctx = None
                    if is_rank0:
                        try:
                            _log_f = open(log_file, "a")
                            _redirect_ctx = contextlib.redirect_stdout(_log_f)
                            _redirect_ctx.__enter__()
                        except OSError:
                            _log_f = None
                            _redirect_ctx = None
                    try:
                        self._run_wham(nf, ms, msprof, cut_params, alf_info, phase)
                    finally:
                        if _redirect_ctx is not None:
                            _redirect_ctx.__exit__(None, None, None)
                        if _log_f is not None:
                            _log_f.close()
                else:
                    # Single-rank: existing behavior
                    with open(log_file, "a") as log_f:
                        with contextlib.redirect_stdout(log_f):
                            if method == "hybrid":
                                self._run_hybrid(
                                    nf, ms, msprof, cut_params, alf_info, phase
                                )
                            elif method == "lmalf":
                                self._run_lmalf(nf, ms, msprof, cut_params, alf_info, phase)
                            elif method == "nonlinear":
                                self._run_nonlinear(
                                    nf, ms, msprof, cut_params, alf_info, phase
                                )
                            else:
                                self._run_wham(nf, ms, msprof, cut_params, alf_info, phase)
            except Exception as e:
                error = str(e)

            # --- Sync error status across ranks ---
            # Exceptions from run_wham_distributed are synchronized
            # (all ranks raise together due to internal bcast/allreduce)
            if is_distributed:
                errors = self._comm.allgather(error)
                any_error = any(e is not None for e in errors)
            else:
                any_error = error is not None

            if any_error:
                if is_rank0:
                    err_detail = errors if is_distributed else error
                    msg = f"{method.upper()} attempt {attempt + 1} failed"
                    with open(log_file, "a") as log_f:
                        log_f.write(f"{msg}: {err_detail}\n")
                    self.cleanup_invalid(analysis_dir)
                if is_distributed:
                    self._comm.Barrier()
                if attempt == max_attempts - 1:
                    return False, f"{method.upper()} failed after {max_attempts} attempts"
                continue

            # --- Validate output (rank 0 decides, broadcasts result) ---
            if is_rank0:
                valid = not self.is_output_invalid(analysis_dir, cut_params)
            else:
                valid = True

            if is_distributed:
                valid = self._comm.bcast(valid, root=0)

            if valid:
                msg = f"{method.upper()} succeeded on attempt {attempt + 1}"
                if is_rank0:
                    with open(log_file, "a") as log_f:
                        log_f.write(f"{msg}\n")
                return True, msg

            # --- Invalid output: retry ---
            if is_rank0:
                msg = f"{method.upper()} output invalid on attempt {attempt + 1}"
                with open(log_file, "a") as log_f:
                    log_f.write(f"{msg}\n")
                self.cleanup_invalid(analysis_dir)
            if is_distributed:
                self._comm.Barrier()

        return False, f"{method.upper()} failed after {max_attempts} attempts"

    def _run_wham(self, nf: int, ms: int, msprof: int,
                  cut_params: dict, alf_info: dict,
                  phase: "PhaseType" = 3) -> None:
        """Run WHAM analysis using bundled GPU library."""
        from cphmd.core.alf_runner import ALFConfig

        nsubs = alf_info["nsubs"]
        nblocks = alf_info["nblocks"]
        ntriangle = self.config.ntriangle
        ew = ALFConfig.resolve_endpoint_weight(self.config.endpoint_weight, phase)
        ed = ALFConfig.resolve_endpoint_weight(self.config.endpoint_decay, phase)

        wham_kwargs = {
            "gpu_id": self._gpu_id,
            "nblocks": nblocks,
            "nf": nf,
            "temp": self.config.temperature,
            "nts0": ms,
            "nts1": msprof,
            "use_gshift": self.config.use_gshift,
            "nsubs": nsubs,
            "g_imp_path": "../G_imp",
            "output_dir": Path.cwd(),
            "log_file": "analysis.log",
            "fnex": self.config.fnex,
            "cutlsum": self.config.cutlsum,
            "chi_offset": self.config.chi_offset,
            "omega_decay": self.config.omega_decay,
            "chi_offset_t": self.config.chi_offset_t,
            "chi_offset_u": self.config.chi_offset_u,
            "ntriangle": ntriangle,
            "endpoint_weight": ew,
            "endpoint_decay": ed,
        }

        if self._nranks > 1 and self._comm is not None and self._packed_D is not None:
            # Distributed WHAM from pre-packed data (no pickle, no energy_matrix)
            from cphmd.wham import run_wham_distributed_from_packed
            run_wham_distributed_from_packed(
                D_flat=self._packed_D,
                sim_indices=self._packed_sim_indices,
                frame_counts=self._packed_frame_counts,
                total_frames=self._packed_total_frames,
                gshift_data=self._wham_gshift,
                comm=self._comm,
                rank=self._rank,
                nranks=self._nranks,
                **wham_kwargs,
            )
            # SVD solve is rank-0 only; broadcast success/failure so all
            # ranks raise together instead of deadlocking at a Barrier.
            svd_error = None
            if self._rank == 0:
                try:
                    get_free_energy5(
                        alf_info, ms=ms, msprof=msprof,
                        ntriangle=ntriangle, **cut_params,
                    )
                except Exception as exc:
                    svd_error = str(exc)
                    logger.error(f"get_free_energy5 failed: {svd_error}")
            svd_error = self._comm.bcast(svd_error, root=0)
            if svd_error is not None:
                raise RuntimeError(f"get_free_energy5 failed on rank 0: {svd_error}")
        elif self._packed_D is not None and self._packed_D.size > 0:
            from cphmd.wham import run_wham_from_packed
            run_wham_from_packed(
                D_flat=self._packed_D,
                sim_indices=self._packed_sim_indices,
                frame_counts=self._packed_frame_counts,
                total_frames=self._packed_total_frames,
                gshift_data=self._wham_gshift,
                **wham_kwargs,
            )
            get_free_energy5(alf_info, ms=ms, msprof=msprof, ntriangle=ntriangle, **cut_params)
        elif self._wham_lambda:
            # Fallback: old energy_matrix path (backward compat)
            from cphmd.wham import run_wham_from_memory
            run_wham_from_memory(
                lambda_arrays=self._wham_lambda,
                energy_matrix=self._wham_energy,
                gshift_data=self._wham_gshift,
                **wham_kwargs,
            )
            get_free_energy5(alf_info, ms=ms, msprof=msprof, ntriangle=ntriangle, **cut_params)
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
                endpoint_weight=ew,
                endpoint_decay=ed,
                gpu_id=self._gpu_id,
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
            gpu_id=self._gpu_id,
        )
        return True

    def _run_lmalf(self, nf: int, ms: int, msprof: int,
                   cut_params: dict, alf_info: dict,
                   phase: "PhaseType" = 3) -> None:
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
                self._run_wham(nf, ms, msprof, cut_params, alf_info, phase)
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
            gpu_id=self._gpu_id,
        )
        return True

    def _run_nonlinear(self, nf: int, ms: int, msprof: int,
                       cut_params: dict, alf_info: dict,
                       phase: PhaseType = 3) -> None:
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
                self._run_wham(nf, ms, msprof, cut_params, alf_info, phase=phase)
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
        self._run_wham(nf, ms, msprof, cut_params, alf_info, phase=phase)

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
        """Check for invalid WHAM output (NaN, Inf, or all biases zero).

        Individual files can legitimately be all-zeros (e.g. x.dat in Phase 1).
        Only flags failure when ALL active bias files are zeros combined —
        meaning WHAM produced no useful updates at all.
        """
        file_cut_map = {
            'b.dat': 'cutb', 'c.dat': 'cutc',
            'x.dat': 'cutx', 's.dat': 'cuts',
            'b_sum.dat': 'cutb', 'c_sum.dat': 'cutc',
            'x_sum.dat': 'cutx', 's_sum.dat': 'cuts',
            't.dat': 'cutt', 'u.dat': 'cutu',
            't_sum.dat': 'cutt', 'u_sum.dat': 'cutu',
        }
        all_zero = True  # Track whether ALL active files are zeros
        checked_any = False
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
                if np.any(np.isnan(data)):
                    print(f"WHAM validation: {fname} contains NaN")
                    return True
                if np.any(np.isinf(data)):
                    print(f"WHAM validation: {fname} contains Inf")
                    return True
                checked_any = True
                if not np.all(data == 0):
                    all_zero = False
            except Exception as e:
                print(f"WHAM validation: error reading {fname}: {e}")
                return True
        if checked_any and all_zero:
            print("WHAM validation: all bias files contain zeros (no updates)")
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
