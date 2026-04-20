from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cphmd.core.alf_utils import set_vars_from_analysis_dir
from cphmd.core.bias_analyzer import BiasAnalyzer
from cphmd.core.g_imp_provisioner import GImpProvisioner
from cphmd.simulation.context import LoopState, RunContext
from cphmd.training.bias_snapshot import BiasSnapshot
from cphmd.training.segment_cache import SegmentCache
from cphmd.utils.lambda_io import write_lambda_parquet


@dataclass
class NativeALFAnalyzer:
    """Bridge native segment archives into the existing ALF WHAM analyzer."""

    config: Any
    ctx: RunContext
    alf_info: dict[str, Any]
    work_dir: Path
    bias_analyzer: BiasAnalyzer | None = None

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

    def analyze(self, *, state: LoopState, cache: SegmentCache) -> BiasSnapshot:
        if not cache.lambda_arrays:
            raise RuntimeError("ALF cycle requested before any lambda segments were cached")

        analysis_idx = state.cycle_idx + 1
        analysis_dir = self.work_dir / f"analysis{analysis_idx}"
        is_rank0 = self.ctx.rank == 0

        if is_rank0:
            self._ensure_initial_analysis()
            self._prepare_analysis_dir(analysis_idx)
            self._ensure_g_imp(state.phase)
        self._barrier()

        self._write_rank_lambda_files(analysis_dir, cache)
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
                phase2_start_run=None,
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
        finally:
            os.chdir(home)

        snapshot = self._broadcast_snapshot(snapshot)
        if not isinstance(snapshot, BiasSnapshot):
            raise RuntimeError("ALF analysis did not produce a bias snapshot")
        return snapshot

    def _ensure_initial_analysis(self) -> None:
        analysis0 = self.work_dir / "analysis0"
        block_path = self._initial_block_path()
        if (analysis0 / "b_sum.dat").exists():
            self._write_initial_block_state_hash(analysis0, block_path)
            return
        analysis0.mkdir(parents=True, exist_ok=True)
        nsubs = tuple(int(value) for value in self.alf_info["nsubs"])
        snapshot = BiasSnapshot.from_block_str(block_path, nsubs=nsubs)
        _write_prev_and_zero_biases(analysis0, snapshot)
        set_vars_from_analysis_dir(analysis0, self.alf_info, step=1)
        self._write_initial_block_state_hash(analysis0, block_path)

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

    def _write_rank_lambda_files(self, analysis_dir: Path, cache: SegmentCache) -> None:
        data_dir = analysis_dir / "data"
        columns = ["time", *self.ctx.lambda_headers]
        for repeat_idx, lambda_matrix in enumerate(cache.lambda_arrays):
            lambda_matrix = np.asarray(lambda_matrix, dtype=np.float32)
            times = np.arange(lambda_matrix.shape[0], dtype=np.float32)
            out = np.column_stack([times, lambda_matrix])
            write_lambda_parquet(
                data_dir / f"Lambda.{repeat_idx}.{self.ctx.rank}.parquet",
                out,
                column_names=columns,
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

    def _broadcast_snapshot(self, snapshot: BiasSnapshot | None) -> BiasSnapshot | None:
        if self.ctx.comm is None:
            return snapshot
        return self.ctx.comm.bcast(snapshot, root=0)

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
