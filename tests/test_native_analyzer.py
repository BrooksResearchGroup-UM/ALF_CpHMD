from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cphmd.training.alf_hooks import ALFTrainingConfig
from cphmd.training.config import ALFConfig
from cphmd.training.native_analyzer import NativeALFAnalyzer


def test_native_patch_info_derives_site_columns_from_select(tmp_path):
    prep_dir = tmp_path / "prep"
    prep_dir.mkdir()
    (prep_dir / "patches.dat").write_text(
        "SEGID,RESID,PATCH,SELECT,ATOMS,TAG\n"
        "PROA,3,LYSO,s1s1,HE2 HE1 CE HZ2 HZ3 NZ HZ1,NONE\n"
        "PROA,3,LYSU,s1s2,HE2 HE1 CE HZ2 NZ HZ1,UPOS 10.4\n"
    )

    analyzer = object.__new__(NativeALFAnalyzer)
    analyzer.work_dir = tmp_path
    analyzer.ctx = SimpleNamespace(run_dir=tmp_path)

    patch_info = analyzer._patch_info()

    assert patch_info["site"].tolist() == [1, 1]
    assert patch_info["sub"].tolist() == [1, 2]


def test_population_plots_default_on():
    assert ALFConfig.__dataclass_fields__["generate_population_plots"].default is True
    cfg = ALFTrainingConfig(cycle_every_segments=1, end_cycle=1, cache_segments=1)
    assert cfg.generate_population_plots is True
    assert cfg.generate_hh_plots is True


def test_native_population_file_uses_central_replica(tmp_path):
    data_dir = tmp_path / "analysis1" / "data"
    data_dir.mkdir(parents=True)
    np.savetxt(data_dir / "Lambda.0.0.dat", [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    np.savetxt(data_dir / "Lambda.0.1.dat", [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    np.savetxt(data_dir / "Lambda.0.2.dat", [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    analyzer = object.__new__(NativeALFAnalyzer)
    analyzer.work_dir = tmp_path
    analyzer.alf_info = {"nreps": 3}

    analyzer._write_population_file(1, np.empty((0, 2)), (2,))

    data = np.loadtxt(tmp_path / "analysis1" / "populations.dat", comments="#")
    assert data[:, 5].tolist() == [0.0, 1.0]
