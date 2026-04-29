from types import SimpleNamespace

import pytest

from cphmd.cli.run_cmd import _alf_config_from_native_config, _alf_repeats_for_phase
from cphmd.config.loader import load_config


def test_native_alf_defaults_use_full_step_intervals():
    config = SimpleNamespace()

    assert (
        _alf_repeats_for_phase(
            config,
            1,
            nsteps_per_segment=1000,
            time_step_ps=0.004,
        )
        == 40
    )
    assert (
        _alf_repeats_for_phase(
            config,
            2,
            nsteps_per_segment=1000,
            time_step_ps=0.004,
        )
        == 250
    )
    assert (
        _alf_repeats_for_phase(
            config,
            3,
            nsteps_per_segment=1000,
            time_step_ps=0.004,
        )
        == 2500
    )


def test_rex_legacy_single_repeat_does_not_shorten_alf_iteration():
    config = SimpleNamespace(phase2_repeats=1)

    assert (
        _alf_repeats_for_phase(
            config,
            2,
            nsteps_per_segment=1000,
            time_step_ps=0.004,
            replica_exchange_enabled=True,
        )
        == 250
    )


def test_non_rex_legacy_single_repeat_is_preserved():
    config = SimpleNamespace(phase2_repeats=1)

    assert (
        _alf_repeats_for_phase(
            config,
            2,
            nsteps_per_segment=1000,
            time_step_ps=0.004,
            replica_exchange_enabled=False,
        )
        == 1
    )


def test_phase_length_keys_still_conflict_when_both_explicit():
    config = SimpleNamespace(phase2_repeats=1, phase2_iteration_steps=250000)

    with pytest.raises(ValueError, match="conflicting Phase 2"):
        _alf_repeats_for_phase(
            config,
            2,
            nsteps_per_segment=1000,
            time_step_ps=0.004,
            replica_exchange_enabled=True,
        )


def test_native_loader_saves_lambda_every_step_by_default(tmp_path):
    cfg_path = tmp_path / "cphmd_config.yaml"
    cfg_path.write_text(
        """
master_seed: 1
alf:
  input_folder: .
""".lstrip(),
        encoding="utf-8",
    )

    assert load_config(cfg_path).nsavl == 1


def test_native_loader_keeps_explicit_lambda_save_steps(tmp_path):
    cfg_path = tmp_path / "cphmd_config.yaml"
    cfg_path.write_text(
        """
master_seed: 1
alf:
  input_folder: .
  lambda_save_steps: 10
""".lstrip(),
        encoding="utf-8",
    )

    assert load_config(cfg_path).nsavl == 10


def test_alf_config_uses_resolved_input_folder(tmp_path, monkeypatch):
    case_dir = tmp_path / "case"
    prep_dir = case_dir / "prep"
    prep_dir.mkdir(parents=True)
    for name in ("system.psf", "system.crd", "patches.dat", "box.dat", "fft.dat"):
        (prep_dir / name).write_text("", encoding="utf-8")
    cfg_path = case_dir / "cphmd_config.yaml"
    cfg_path.write_text(
        """
master_seed: 1
alf:
  input_folder: .
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    alf_config = _alf_config_from_native_config(load_config(cfg_path))

    assert alf_config.input_folder == case_dir.resolve()
