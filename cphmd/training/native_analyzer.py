from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cphmd.core.alf_utils import set_vars_from_analysis_dir
from cphmd.core.bias_analyzer import BiasAnalyzer
from cphmd.core.g_imp_provisioner import GImpProvisioner
from cphmd.simulation.context import LoopState, RunContext
from cphmd.training.bias_snapshot import BiasSnapshot
from cphmd.training.lambda_compactor import compact_analysis_lambda
from cphmd.training.segment_cache import SegmentCache


@dataclass
class NativeALFAnalyzer:
    """Bridge native segment archives into the existing ALF WHAM analyzer."""

    config: Any
    ctx: RunContext
    alf_info: dict[str, Any]
    work_dir: Path
    bias_analyzer: BiasAnalyzer | None = None
    initial_bias_provider: Callable[[], BiasSnapshot] | None = None
    last_phase: int | None = None
    last_stop_requested: bool = False
    last_stop_reason: str | None = None

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir)
        self.alf_info = _normalized_alf_info(self.alf_info, self.config, self.ctx)
        if self.bias_analyzer is None:
            self.bias_analyzer = BiasAnalyzer(self.config)
        if self.ctx.comm is not None:
            size = getattr(self.ctx.comm, "Get_size", lambda: 1)()
            self.bias_analyzer.set_mpi(
                self.ctx.comm,
                self.ctx.rank,
                int(size),
                gpu_id=self.ctx.gpu_id,
            )
        self._phase2_start_run: int | None = None
        self._needs_stop_confirmation = False
        self._ewbs_state = None
        self._load_native_convergence_state()

    def analyze(self, *, state: LoopState, cache: SegmentCache) -> BiasSnapshot:
        if not cache.segment_ids:
            raise RuntimeError("ALF cycle requested before any segment ids were cached")

        analysis_idx = state.cycle_idx + 1
        analysis_dir = self.work_dir / f"analysis{analysis_idx}"
        is_rank0 = self.ctx.rank == 0

        if is_rank0:
            self._ensure_initial_analysis()
            self._prepare_analysis_dir(analysis_idx)
            self._ensure_g_imp(state.phase)
        self._barrier()

        self._compact_rank_lambda_file(analysis_idx, cache.segment_ids)
        self._barrier()

        home = os.getcwd()
        try:
            os.chdir(analysis_dir)
            nf = self.bias_analyzer.prepare_data(
                self.alf_info,
                analysis_idx,
                state.phase,
            )
            nsubs = tuple(int(value) for value in self.alf_info["nsubs"])
            ms, msprof = self._ntersite()
            coupling_scale = (2.0 / max(max(nsubs), 2)) ** 0.5 if nsubs else 1.0
            cut_params = self.bias_analyzer.compute_cutoffs(
                state.phase,
                analysis_idx,
                coupling_scale,
                phase2_start_run=self._phase2_start_run,
                alf_info=self.alf_info,
                input_folder=self.work_dir,
                nreps=int(self.alf_info["nreps"]),
            )
            self._apply_disabled_bias_flags(cut_params, state.phase)

            success, msg = self.bias_analyzer.run_with_retry(
                run_idx=analysis_idx,
                nf=nf,
                ms=ms,
                msprof=msprof,
                cut_params=cut_params,
                alf_info=self.alf_info,
                phase=state.phase,
                max_attempts=3,
            )
            self._release_packed_data()

            snapshot = None
            if is_rank0:
                if not success:
                    nblocks = int(self.alf_info["nblocks"])
                    np.savetxt("b.dat", np.zeros((1, nblocks)), fmt=" %10.5f")
                    np.savetxt("c.dat", np.zeros((nblocks, nblocks)), fmt=" %10.5f")
                    np.savetxt("x.dat", np.zeros((nblocks, nblocks)), fmt=" %10.5f")
                    np.savetxt("s.dat", np.zeros((nblocks, nblocks)), fmt=" %10.5f")
                with open("analysis.log", "a") as handle:
                    handle.write(f"{msg}\n")
                set_vars_from_analysis_dir(
                    analysis_dir,
                    self.alf_info,
                    step=analysis_idx + 1,
                )
                snapshot = BiasSnapshot.from_analysis_dir(analysis_dir, nsubs=nsubs)
                self._update_native_convergence(
                    analysis_idx=analysis_idx,
                    phase=state.phase,
                    cut_params=cut_params,
                    nsubs=nsubs,
                )
        finally:
            os.chdir(home)

        snapshot = self._broadcast_snapshot(snapshot)
        self._broadcast_native_convergence_state()
        if not isinstance(snapshot, BiasSnapshot):
            raise RuntimeError("ALF analysis did not produce a bias snapshot")
        return snapshot

    def initialize_initial_analysis(self) -> BiasSnapshot:
        snapshot = None
        if self.ctx.rank == 0:
            snapshot = self._ensure_initial_analysis()
        self._barrier()
        snapshot = self._broadcast_snapshot(snapshot)
        if isinstance(snapshot, BiasSnapshot):
            return snapshot
        nsubs = tuple(int(value) for value in self.alf_info["nsubs"])
        return BiasSnapshot.from_analysis_dir(self.work_dir / "analysis0", nsubs=nsubs)

    def _ensure_initial_analysis(self) -> BiasSnapshot:
        analysis0 = self.work_dir / "analysis0"
        block_path = self._initial_block_path()
        if (analysis0 / "b_sum.dat").exists():
            self._write_initial_block_state_hash(analysis0, block_path)
            nsubs = tuple(int(value) for value in self.alf_info["nsubs"])
            return BiasSnapshot.from_analysis_dir(analysis0, nsubs=nsubs)
        analysis0.mkdir(parents=True, exist_ok=True)
        nsubs = tuple(int(value) for value in self.alf_info["nsubs"])
        if self.initial_bias_provider is None:
            snapshot = BiasSnapshot.from_block_str(block_path, nsubs=nsubs)
        else:
            snapshot = self.initial_bias_provider()
        _write_prev_and_zero_biases(analysis0, snapshot)
        set_vars_from_analysis_dir(analysis0, self.alf_info, step=1)
        self._write_initial_block_state_hash(analysis0, block_path)
        return snapshot

    def _initial_block_path(self) -> Path:
        for block_path in (
            self.work_dir / "prep" / "block.str",
            self.ctx.run_dir / "prep" / "block.str",
        ):
            if block_path.exists():
                return block_path
        raise FileNotFoundError(
            f"Cannot initialize ALF biases; missing {self.work_dir / 'prep' / 'block.str'}"
        )

    def _write_initial_block_state_hash(self, analysis0: Path, block_path: Path) -> None:
        digest = hashlib.sha256()
        for path in [
            block_path,
            analysis0 / "b_prev.dat",
            analysis0 / "c_prev.dat",
            analysis0 / "x_prev.dat",
            analysis0 / "s_prev.dat",
        ]:
            if not path.exists():
                continue
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        value = digest.hexdigest()
        for out_path in (
            analysis0 / "initial_block_state.sha256",
            self.work_dir / "initial_block_state.sha256",
        ):
            out_path.write_text(value + "\n", encoding="utf-8")

    def _prepare_analysis_dir(self, analysis_idx: int) -> None:
        analysis_dir = self.work_dir / f"analysis{analysis_idx}"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        data_dir = analysis_dir / "data"
        if data_dir.exists():
            shutil.rmtree(data_dir)
        data_dir.mkdir()
        prev = self.work_dir / f"analysis{analysis_idx - 1}"
        for name in ("b", "c", "x", "s", "t", "u"):
            src = prev / f"{name}_sum.dat"
            if src.exists():
                shutil.copy(src, analysis_dir / f"{name}_prev.dat")
        for name in ("t", "u"):
            prev_path = analysis_dir / f"{name}_prev.dat"
            if not prev_path.exists():
                nblocks = int(self.alf_info["nblocks"])
                np.savetxt(prev_path, np.zeros((nblocks, nblocks)), fmt=" %10.5f")

    def _compact_rank_lambda_file(
        self,
        analysis_idx: int,
        segment_ids: tuple[int, ...],
    ) -> Path:
        return compact_analysis_lambda(
            run_dir=self.ctx.run_dir,
            analysis_idx=analysis_idx,
            replica_idx=self.ctx.rank,
            segment_ids=segment_ids,
            lambda_headers=self.ctx.lambda_headers,
            output_root=self.work_dir,
            replica_ph_values=self.ctx.replica_ph_values,
        )

    def _ensure_g_imp(self, phase: int) -> None:
        resolve = getattr(self.config, "resolve_g_imp_bins", None)
        bins = resolve(self.config.g_imp_bins, phase) if resolve is not None else None
        provisioner = GImpProvisioner(
            input_folder=self.work_dir,
            nsubs=list(self.alf_info["nsubs"]),
            fnex=float(getattr(self.config, "fnex", 5.5)),
            cutlsum=float(getattr(self.config, "cutlsum", 0.8)),
            g_imp_bins=bins,
        )
        provisioner.ensure_available()
        self._write_g_imp_provenance(provisioner)

    def _write_g_imp_provenance(self, provisioner: GImpProvisioner) -> None:
        g_imp_dir = self.work_dir / "G_imp"
        files = sorted(path for path in g_imp_dir.glob("*.dat") if path.is_file())
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        target = None
        if g_imp_dir.is_symlink():
            try:
                target = str(g_imp_dir.resolve())
            except OSError:
                target = str(os.readlink(g_imp_dir))
        payload = {
            "path": str(g_imp_dir),
            "is_symlink": g_imp_dir.is_symlink(),
            "target": target,
            "file_count": len(files),
            "sha256": digest.hexdigest(),
            "bins": provisioner.detect_bins(g_imp_dir) if g_imp_dir.exists() else None,
            "fnex": provisioner.fnex,
            "cutlsum": provisioner.cutlsum,
            "nsubs": list(provisioner.nsubs),
        }
        (self.work_dir / "g_imp_provenance.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _ntersite(self) -> tuple[int, int]:
        value = self.alf_info.get("ntersite")
        if value is not None:
            return int(value[0]), int(value[1])
        coupling = int(getattr(self.config, "coupling", 0))
        profile = getattr(self.config, "coupling_profile", None)
        return coupling, int(coupling > 0 if profile is None else bool(profile))

    def _apply_disabled_bias_flags(self, cut_params: dict[str, Any], phase: int) -> None:
        if phase == 1:
            return
        flag_map = {
            "no_b_bias": "calc_phi",
            "no_c_bias": "calc_psi",
            "no_x_bias": "calc_chi",
            "no_s_bias": "calc_omega",
            "no_t_bias": "calc_omega2",
            "no_u_bias": "calc_omega3",
        }
        for attr, key in flag_map.items():
            if getattr(self.config, attr, False):
                cut_params[key] = False

    def _release_packed_data(self) -> None:
        self.bias_analyzer._wham_lambda = None
        self.bias_analyzer._D_2d_ref = None
        self.bias_analyzer._packed_D = None
        self.bias_analyzer._packed_sim_indices = None
        self.bias_analyzer._packed_frame_counts = None

    def _update_native_convergence(
        self,
        *,
        analysis_idx: int,
        phase: int,
        cut_params: dict[str, Any],
        nsubs: tuple[int, ...],
    ) -> None:
        self.last_phase = phase
        self.last_stop_requested = False
        self.last_stop_reason = None
        self._update_ewbs_state(analysis_idx)
        lambda_data = self._load_analysis_lambda_data(analysis_idx)
        if lambda_data is None:
            self._save_native_convergence_state()
            return
        self._write_population_file(analysis_idx, lambda_data, nsubs)

        phase_after = phase
        if getattr(self.config, "auto_phase_switch", False):
            phase_after = self._check_auto_phase(
                analysis_idx=analysis_idx,
                phase=phase,
                lambda_data=lambda_data,
                cut_params=cut_params,
                nsubs=nsubs,
            )
        self.last_phase = phase_after

        if getattr(self.config, "auto_stop", False) and phase_after == 3:
            self._check_auto_stop(
                analysis_idx=analysis_idx,
                lambda_data=lambda_data,
                nsubs=nsubs,
            )
        else:
            self._needs_stop_confirmation = False
        self._save_native_convergence_state()

    def _update_ewbs_state(self, analysis_idx: int) -> None:
        from cphmd.core.phase_switcher import EWBSState, ewbs_bottleneck_type, update_ewbs_state

        if self._ewbs_state is None:
            self._ewbs_state = EWBSState()
        analysis_dir = self.work_dir / f"analysis{analysis_idx}"
        try:
            b = np.loadtxt(analysis_dir / "b.dat")
            c = np.loadtxt(analysis_dir / "c.dat")
            x = np.loadtxt(analysis_dir / "x.dat")
            s = np.loadtxt(analysis_dir / "s.dat")
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"Warning: Cannot update EWBS from analysis{analysis_idx}: {exc}")
            return
        value = update_ewbs_state(self._ewbs_state, b, c, x, s)
        print(
            f"EWBS: {value:.4f} "
            f"(bottleneck={ewbs_bottleneck_type(self._ewbs_state)})"
        )

    def _load_analysis_lambda_data(self, analysis_idx: int):
        from cphmd.core.phase_switcher import load_lambda_data

        lambda_data, _ = load_lambda_data(self.work_dir / f"analysis{analysis_idx}" / "data")
        return lambda_data

    def _write_population_file(
        self,
        analysis_idx: int,
        lambda_data,
        nsubs: tuple[int, ...],
    ) -> None:
        from cphmd.core.phase_switcher import (
            calculate_populations,
            load_lambda_data,
            write_populations_file,
        )

        analysis_dir = self.work_dir / f"analysis{analysis_idx}"
        data_dir = analysis_dir / "data"
        ncentral = int(self.alf_info["nreps"]) // 2
        central_data, _ = load_lambda_data(data_dir, replica_idx=ncentral)
        pop_input = central_data if central_data is not None else lambda_data
        pop_data = calculate_populations(
            pop_input,
            thresholds=(0.8, 0.97),
            nsubs=list(nsubs),
        )
        write_populations_file(analysis_dir / "populations.dat", pop_data)

    def generate_hh_plots(
        self,
        *,
        analysis_idx: int,
        phase: int,
        output_dir: Path,
        nsubs: tuple[int, ...],
    ) -> None:
        if not getattr(self.config, "ph", False):
            return
        patch_info = self._patch_info()
        if patch_info is None:
            return
        nreps = int(self.alf_info["nreps"])
        if nreps < 3:
            return

        from cphmd.analysis.henderson_hasselbalch import generate_hh_analysis
        from cphmd.core.cphmd_params import compute_all_site_parameters, get_delta_pKa_for_phase

        params = compute_all_site_parameters(patch_info, self.config.temperature)
        pka_shift = {
            str(site_id): params.effective_pH - site_params.pH0
            for site_id, site_params in params.sites.items()
        }
        generate_hh_analysis(
            run_idx=analysis_idx,
            data_dir=self.work_dir / f"analysis{analysis_idx}" / "data",
            patch_info=patch_info,
            pH=params.effective_pH,
            delta_pKa=get_delta_pKa_for_phase(phase),
            nreps=nreps,
            output_dir=output_dir,
            ncentral=nreps // 2,
            nsubs=list(nsubs),
            pka_shift=pka_shift,
        )

    def _check_auto_phase(
        self,
        *,
        analysis_idx: int,
        phase: int,
        lambda_data,
        cut_params: dict[str, Any],
        nsubs: tuple[int, ...],
    ) -> int:
        from cphmd.core.phase_switcher import check_phase_transition

        min_phase1 = int(self.config.phase_transition.min_phase1_runs)
        if phase == 1 and analysis_idx < min_phase1:
            print(
                f"Phase check: Staying in phase 1: run {analysis_idx}<{min_phase1} "
                "(minimum Phase 1 duration)"
            )
            return phase

        phase2_run_count = None
        if phase == 2 and self._phase2_start_run is not None:
            phase2_run_count = analysis_idx - self._phase2_start_run

        new_phase, reason = check_phase_transition(
            phase,
            lambda_data,
            config=self.config.phase_transition,
            **self._cphmd_phase_kwargs(analysis_idx, phase, nsubs),
            nsubs=list(nsubs),
            connectivity=cut_params.get("connectivity"),
            phase2_run_count=phase2_run_count,
            ewbs_state=self._ewbs_state,
            expected_pops=self._expected_populations(nsubs),
        )
        if new_phase != phase:
            print(f"PHASE TRANSITION: {phase} -> {new_phase}")
            print(f"  Reason: {reason}")
            if new_phase == 2 and self._phase2_start_run is None:
                self._phase2_start_run = analysis_idx
            return int(new_phase)
        print(f"Phase check: {reason}")
        return phase

    def _check_auto_stop(
        self,
        *,
        analysis_idx: int,
        lambda_data,
        nsubs: tuple[int, ...],
    ) -> None:
        from cphmd.core.phase_switcher import StopCriteriaConfig, check_phase3_stop

        timestep_fs = 4.0 if getattr(self.config, "hmr", False) else 2.0
        stop_config = StopCriteriaConfig(timestep_fs=timestep_fs, max_frac_diff=0.02)
        should_stop, reason, _ = check_phase3_stop(
            lambda_data,
            stop_config,
            bias_history=self._bias_history(analysis_idx, stop_config.bias_window),
            nsubs=list(nsubs),
            ewbs_state=self._ewbs_state,
            expected_pops=self._expected_populations(nsubs),
        )
        if should_stop and self._needs_stop_confirmation:
            self.last_stop_requested = True
            self.last_stop_reason = reason
            self._needs_stop_confirmation = False
            print(f"CONVERGENCE CONFIRMED at run {analysis_idx}: {reason}")
            (self.work_dir / "CONVERGED").write_text(
                f"Converged at run {analysis_idx}\n{reason}\n",
                encoding="utf-8",
            )
        elif should_stop:
            self._needs_stop_confirmation = True
            print(f"CONVERGENCE CANDIDATE at run {analysis_idx}: {reason}")
        else:
            self._needs_stop_confirmation = False
            print(f"Stop check: {reason}")

    def _bias_history(self, analysis_idx: int, window: int):
        rows = []
        for idx in range(max(1, analysis_idx - window + 1), analysis_idx + 1):
            path = self.work_dir / f"analysis{idx}" / "b_sum.dat"
            if not path.exists():
                continue
            try:
                rows.append(np.loadtxt(path).ravel())
            except (ValueError, OSError):
                continue
        if len(rows) < 2:
            return None
        return np.asarray(rows, dtype=np.float64)

    def _cphmd_phase_kwargs(
        self,
        analysis_idx: int,
        phase: int,
        nsubs: tuple[int, ...],
    ) -> dict[str, Any]:
        patch_info = self._patch_info()
        if patch_info is None:
            return {}
        kwargs: dict[str, Any] = {"patch_info": patch_info}
        nreps = int(self.alf_info["nreps"])
        if phase >= 2 and getattr(self.config, "ph", False) and nreps > 3:
            from cphmd.core.cphmd_params import compute_all_site_parameters, get_delta_pKa_for_phase

            params = compute_all_site_parameters(patch_info, self.config.temperature)
            kwargs.update(
                {
                    "data_dir": self.work_dir / f"analysis{analysis_idx}" / "data",
                    "effective_ph": params.effective_pH,
                    "delta_pka": get_delta_pKa_for_phase(phase),
                    "nreps": nreps,
                }
            )
        return kwargs

    def _expected_populations(self, nsubs: tuple[int, ...]):
        patch_info = self._patch_info()
        if patch_info is None or not getattr(self.config, "ph", False):
            return None
        from cphmd.core.cphmd_params import compute_all_site_parameters
        from cphmd.core.expected_populations import compute_expected_populations

        params = compute_all_site_parameters(patch_info, self.config.temperature)
        return compute_expected_populations(patch_info, params.effective_pH, list(nsubs))

    def _patch_info(self):
        patches_path = self.work_dir / "prep" / "patches.dat"
        if not patches_path.exists():
            patches_path = self.ctx.run_dir / "prep" / "patches.dat"
        if not patches_path.exists():
            return None
        import pandas as pd

        patch_info = pd.read_csv(patches_path)
        if "site" not in patch_info.columns or "sub" not in patch_info.columns:
            if "SELECT" not in patch_info.columns:
                return patch_info
            patch_info[["site", "sub"]] = patch_info["SELECT"].str.extract(r"(?i)s(\d+)s(\d+)")
        patch_info["site"] = patch_info["site"].astype(int)
        patch_info["sub"] = patch_info["sub"].astype(int)
        return patch_info

    def _native_convergence_path(self) -> Path:
        return self.work_dir / "native_convergence_state.json"

    def _load_native_convergence_state(self) -> None:
        payload_path = self._native_convergence_path()
        if payload_path.exists():
            try:
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
                self._phase2_start_run = payload.get("phase2_start_run")
                self._needs_stop_confirmation = bool(payload.get("needs_stop_confirmation", False))
            except (OSError, json.JSONDecodeError):
                pass
        ewbs_path = self.work_dir / "ewbs_state.json"
        if ewbs_path.exists():
            try:
                from cphmd.core.phase_switcher import EWBSState

                self._ewbs_state = EWBSState.load(ewbs_path)
            except (OSError, json.JSONDecodeError, ValueError):
                self._ewbs_state = None

    def _save_native_convergence_state(self) -> None:
        payload = {
            "phase2_start_run": self._phase2_start_run,
            "needs_stop_confirmation": self._needs_stop_confirmation,
            "last_phase": self.last_phase,
            "last_stop_requested": self.last_stop_requested,
            "last_stop_reason": self.last_stop_reason,
        }
        self._native_convergence_path().write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if self._ewbs_state is not None:
            self._ewbs_state.save(self.work_dir / "ewbs_state.json")

    def _broadcast_snapshot(self, snapshot: BiasSnapshot | None) -> BiasSnapshot | None:
        if self.ctx.comm is None:
            return snapshot
        return self.ctx.comm.bcast(snapshot, root=0)

    def _broadcast_native_convergence_state(self) -> None:
        payload = None
        if self.ctx.rank == 0:
            payload = {
                "last_phase": self.last_phase,
                "last_stop_requested": self.last_stop_requested,
                "last_stop_reason": self.last_stop_reason,
                "phase2_start_run": self._phase2_start_run,
                "needs_stop_confirmation": self._needs_stop_confirmation,
            }
        if self.ctx.comm is not None:
            payload = self.ctx.comm.bcast(payload, root=0)
        if isinstance(payload, dict):
            self.last_phase = payload.get("last_phase")
            self.last_stop_requested = bool(payload.get("last_stop_requested", False))
            self.last_stop_reason = payload.get("last_stop_reason")
            self._phase2_start_run = payload.get("phase2_start_run")
            self._needs_stop_confirmation = bool(payload.get("needs_stop_confirmation", False))

    def _barrier(self) -> None:
        if self.ctx.comm is not None:
            self.ctx.comm.Barrier()


def _normalized_alf_info(
    alf_info: dict[str, Any],
    config: Any,
    ctx: RunContext,
) -> dict[str, Any]:
    info = dict(alf_info)
    nsubs = tuple(int(value) for value in info.get("nsubs", _nsubs_from_context(ctx)))
    info["nsubs"] = list(nsubs)
    info["nblocks"] = int(info.get("nblocks", sum(nsubs)))
    replica_count = len(ctx.replica_ph_values) if ctx.replica_ph_values else 1
    info["nreps"] = int(getattr(config, "nreps", None) or replica_count)
    info.setdefault("ncentral", info["nreps"] // 2)
    info.setdefault("nnodes", 1)
    info.setdefault("temp", float(ctx.temperature))
    info.setdefault("name", ctx.simulation_name)
    info.setdefault("engine", "charmm")
    coupling = int(getattr(config, "coupling", 0))
    profile = getattr(config, "coupling_profile", None)
    info.setdefault("ntersite", [coupling, int(coupling > 0 if profile is None else bool(profile))])
    info.setdefault("fnex", float(getattr(config, "fnex", 5.5)))
    return info


def _nsubs_from_context(ctx: RunContext) -> tuple[int, ...]:
    return tuple(
        sum(1 for value in ctx.nsubsites[1:] if value == site)
        for site in range(1, ctx.nsites + 1)
    )


def _write_prev_and_zero_biases(path: Path, snapshot: BiasSnapshot) -> None:
    nblocks = snapshot.b.shape[1]
    zero_vec = np.zeros((1, nblocks), dtype=np.float64)
    zero_mat = np.zeros((nblocks, nblocks), dtype=np.float64)
    np.savetxt(path / "b_prev.dat", snapshot.b, fmt=" %10.5f")
    np.savetxt(path / "b.dat", zero_vec, fmt=" %10.5f")
    for name, arr in (("c", snapshot.c), ("x", snapshot.x), ("s", snapshot.s)):
        np.savetxt(path / f"{name}_prev.dat", arr, fmt=" %10.5f")
        np.savetxt(path / f"{name}.dat", zero_mat, fmt=" %10.5f")
    for name in ("t", "u"):
        np.savetxt(path / f"{name}_prev.dat", zero_mat, fmt=" %10.5f")
        np.savetxt(path / f"{name}.dat", zero_mat, fmt=" %10.5f")
