from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_SHARED_METADATA_KEYS = (
    "pH",
    "Temperature",
    "replica_label",
    "lambda_scale",
    "lambda_precision",
    "nblocks",
    "nsites",
    "nsubsites",
)
_METADATA_ALIASES = {
    "pH": ("pH", "ph"),
    "Temperature": ("Temperature", "temperature"),
}


def compact_analysis_lambda(
    *,
    run_dir: str | Path,
    analysis_idx: int,
    replica_idx: int,
    segment_ids: Sequence[int],
    lambda_headers: Sequence[str],
    output_root: str | Path | None = None,
    replica_ph_values: Sequence[float] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    output_root = run_dir if output_root is None else Path(output_root)
    segment_ids = tuple(sorted(int(segment_id) for segment_id in segment_ids))
    if not segment_ids:
        raise ValueError("segment_ids must not be empty")
    lambda_headers = tuple(str(header) for header in lambda_headers)
    expected_columns = ["time", *lambda_headers]
    replica_label = int(replica_idx)
    target_ph = _target_ph(replica_ph_values, replica_label)
    rerun_idx = 0

    output_path = output_root / f"analysis{analysis_idx}" / "data" / (
        f"Lambda.{rerun_idx}.{replica_idx}.parquet"
    )
    segment_infos = _select_segment_infos(
        run_dir,
        replica_label=replica_label,
        segment_ids=segment_ids,
        expected_columns=expected_columns,
        target_ph=target_ph,
    )
    expected_metadata = _expected_metadata(
        analysis_idx=analysis_idx,
        rerun_idx=rerun_idx,
        replica_idx=replica_idx,
        replica_label=replica_label,
        target_ph=target_ph,
        segment_ids=segment_ids,
        row_count=sum(info.num_rows for info in segment_infos),
        segment_fingerprints=tuple(info.fingerprint for info in segment_infos),
        shared_metadata=_shared_metadata(segment_infos),
    )

    if _matches_existing_output(output_path, expected_columns, expected_metadata):
        return output_path

    import pyarrow as pa
    import pyarrow.parquet as pq

    segment_tables = []
    for segment_info in segment_infos:
        table = pq.read_table(segment_info.path)
        segment_tables.append(table)

    table = pa.concat_tables(segment_tables) if len(segment_tables) > 1 else segment_tables[0]
    metadata = _expected_metadata(
        analysis_idx=analysis_idx,
        rerun_idx=rerun_idx,
        replica_idx=replica_idx,
        replica_label=replica_label,
        target_ph=target_ph,
        segment_ids=segment_ids,
        row_count=table.num_rows,
        segment_fingerprints=tuple(info.fingerprint for info in segment_infos),
        shared_metadata=_shared_metadata(segment_infos),
    )
    table = table.replace_schema_metadata(
        {key.encode("utf-8"): value.encode("utf-8") for key, value in metadata.items()}
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    pq.write_table(table, str(tmp_path))
    os.replace(tmp_path, output_path)
    return output_path


def _segment_path(run_dir: Path, replica_idx: int, segment_id: int) -> Path:
    return run_dir / "res" / f"rep{replica_idx:02d}" / f"segment_{segment_id:06d}.parquet"


def _target_ph(replica_ph_values: Sequence[float] | None, replica_label: int) -> float | None:
    if replica_ph_values is None:
        return None
    values = tuple(float(value) for value in replica_ph_values)
    if not values:
        return None
    if not 0 <= replica_label < len(values):
        raise ValueError("replica_idx must be within replica_ph_values for label compaction")
    return values[replica_label]


def _select_segment_infos(
    run_dir: Path,
    *,
    replica_label: int,
    segment_ids: tuple[int, ...],
    expected_columns: list[str],
    target_ph: float | None,
) -> list[_SegmentInfo]:
    infos: list[_SegmentInfo] = []
    for segment_id in segment_ids:
        if target_ph is None:
            legacy_path = _segment_path(run_dir, replica_label, segment_id)
            if legacy_path.exists():
                infos.append(_read_segment_info(legacy_path, expected_columns))
                continue
        infos.append(
            _select_segment_info_by_metadata(
                run_dir,
                replica_label=replica_label,
                segment_id=segment_id,
                expected_columns=expected_columns,
                target_ph=target_ph,
            )
        )
    return infos


def _select_segment_info_by_metadata(
    run_dir: Path,
    *,
    replica_label: int,
    segment_id: int,
    expected_columns: list[str],
    target_ph: float | None,
) -> _SegmentInfo:
    matches = []
    for path in _segment_candidates(run_dir, segment_id):
        info = _read_segment_info(path, expected_columns)
        if _matches_thermodynamic_label(info.metadata, replica_label, target_ph):
            matches.append(info)

    if not matches:
        target = f"replica_label={replica_label}"
        if target_ph is not None:
            target += f" or pH={target_ph}"
        raise FileNotFoundError(
            f"No segment_{segment_id:06d}.parquet found with {target} metadata"
        )
    if len(matches) > 1:
        paths = ", ".join(str(info.path) for info in matches)
        raise ValueError(
            f"Multiple segment_{segment_id:06d}.parquet files match replica_label "
            f"{replica_label}: {paths}"
        )
    return matches[0]


def _segment_candidates(run_dir: Path, segment_id: int) -> list[Path]:
    return sorted((run_dir / "res").glob(f"rep*/segment_{segment_id:06d}.parquet"))


def _matches_thermodynamic_label(
    metadata: dict[str, str],
    replica_label: int,
    target_ph: float | None,
) -> bool:
    label_value = _metadata_value(metadata, "replica_label")
    if label_value is not None:
        try:
            if int(label_value) == replica_label:
                return True
        except ValueError:
            pass

    if target_ph is None:
        return False
    ph_value = _metadata_value(metadata, "pH")
    if ph_value is None:
        return False
    try:
        return math.isclose(float(ph_value), target_ph, rel_tol=0.0, abs_tol=1e-6)
    except ValueError:
        return False


def _read_segment_info(path: Path, expected_columns: list[str]) -> _SegmentInfo:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(str(path))
    schema = parquet_file.schema_arrow
    if list(schema.names) != expected_columns:
        raise ValueError(f"{path} columns must be exactly {expected_columns!r}")
    with path.open("rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return _SegmentInfo(
        path=path,
        num_rows=int(parquet_file.metadata.num_rows),
        metadata=_decode_metadata(schema.metadata),
        fingerprint=digest,
    )


def _expected_metadata(
    *,
    analysis_idx: int,
    rerun_idx: int,
    replica_idx: int,
    replica_label: int,
    target_ph: float | None,
    segment_ids: tuple[int, ...],
    row_count: int | None = None,
    segment_fingerprints: tuple[str, ...] | None = None,
    shared_metadata: dict[str, str] | None = None,
) -> dict[str, str]:
    metadata = dict(shared_metadata or {})
    metadata.update(
        {
            "analysis_idx": str(int(analysis_idx)),
            "rerun_idx": str(int(rerun_idx)),
            "replica_idx": str(int(replica_idx)),
            "replica_label": str(int(replica_label)),
            "segment_ids": json.dumps(list(segment_ids), separators=(",", ":")),
        }
    )
    if target_ph is not None:
        metadata.setdefault("pH", str(float(target_ph)))
    if "pH" in metadata:
        metadata.setdefault("ph", metadata["pH"])
    if "Temperature" in metadata:
        metadata.setdefault("temperature", metadata["Temperature"])
    if row_count is not None:
        metadata["row_count"] = str(int(row_count))
    if segment_fingerprints is not None:
        metadata["segment_fingerprints"] = json.dumps(
            list(segment_fingerprints), separators=(",", ":")
        )
    return metadata


def _matches_existing_output(
    path: Path,
    expected_columns: list[str],
    expected_metadata: dict[str, str],
) -> bool:
    if not path.exists():
        return False

    import pyarrow.parquet as pq

    try:
        parquet_file = pq.ParquetFile(str(path))
    except Exception:
        return False
    schema = parquet_file.schema_arrow
    if list(schema.names) != expected_columns:
        return False

    metadata = _decode_metadata(schema.metadata)
    if metadata != expected_metadata:
        return False
    if metadata.get("row_count") != str(parquet_file.metadata.num_rows):
        return False
    return True


def _shared_metadata(segment_infos: Sequence[_SegmentInfo]) -> dict[str, str]:
    if not segment_infos:
        return {}

    shared: dict[str, str] = {}
    first = segment_infos[0].metadata
    for key in _SHARED_METADATA_KEYS:
        value = _metadata_value(first, key)
        if value is None:
            continue
        if all(_metadata_value(info.metadata, key) == value for info in segment_infos[1:]):
            shared[key] = value
    return shared


def _metadata_value(metadata: dict[str, str], key: str) -> str | None:
    for alias in _METADATA_ALIASES.get(key, (key,)):
        value = metadata.get(alias)
        if value is not None:
            return value
    return None


def _decode_metadata(metadata: dict[bytes, bytes] | None) -> dict[str, str]:
    return {
        key.decode("utf-8"): value.decode("utf-8")
        for key, value in (metadata or {}).items()
    }


@dataclass(frozen=True)
class _SegmentInfo:
    path: Path
    num_rows: int
    metadata: dict[str, str]
    fingerprint: str


__all__ = ["compact_analysis_lambda"]
