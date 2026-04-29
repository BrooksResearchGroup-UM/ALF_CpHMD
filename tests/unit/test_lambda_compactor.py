import numpy as np
import pyarrow.parquet as pq

from cphmd.core.alf_utils import _parse_expected_lambda_file
from cphmd.training.lambda_compactor import compact_analysis_lambda
from cphmd.utils.lambda_io import write_lambda_parquet


def test_compacted_lambda_uses_legacy_rerun_replica_name(tmp_path):
    run_dir = tmp_path / "run"
    metadata = {
        "replica_label": "0",
        "ph": "7.0",
        "temperature": "298.15",
    }
    data = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    for segment_id in (0, 1):
        write_lambda_parquet(
            run_dir / "res" / "rep00" / f"segment_{segment_id:06d}.parquet",
            data,
            column_names=["time", "s1s1", "s1s2"],
            metadata=metadata,
        )

    output = compact_analysis_lambda(
        run_dir=run_dir,
        analysis_idx=5,
        replica_idx=0,
        segment_ids=[0, 1],
        lambda_headers=["s1s1", "s1s2"],
        replica_ph_values=[7.0],
    )

    assert output == run_dir / "analysis5" / "data" / "Lambda.0.0.parquet"
    parquet_metadata = pq.ParquetFile(output).schema_arrow.metadata
    assert parquet_metadata[b"analysis_idx"] == b"5"
    assert parquet_metadata[b"rerun_idx"] == b"0"
    assert parquet_metadata[b"replica_idx"] == b"0"


def test_analysis_loader_accepts_legacy_rerun_lambda_name(tmp_path):
    path = tmp_path / "Lambda.0.4.parquet"
    path.touch()

    assert _parse_expected_lambda_file(path, expected_analysis_idx=17) == (17, 4)
