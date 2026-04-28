"""Convert legacy msld-py-prep ALF folders to the modern prep layout.

The converter is intentionally import-light: parsing and cache validation work
without pyCHARMM, while the PSF/CRD write path imports pyCHARMM only when a
regeneration is required.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import warnings
from collections import defaultdict, deque
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cphmd.native import system
from cphmd.native.types import AtomSelection

MANIFEST_VERSION = 9
_REQUIRED_PREP_FILES = (
    "system.psf",
    "system.crd",
    "patches.dat",
    "box.dat",
    "fft.dat",
)


@dataclass
class LegacySetupMetadata:
    """Metadata parsed from a legacy msld-py-prep setup script."""

    ligseg: str = "LIG"
    resname: str = "LIG"
    resnum: str = "1"
    box: float | None = None
    temp: float | None = None
    legacy_ph: float | None = None
    parameter_files: list[Path] = field(default_factory=list)


@dataclass
class LegacyConvertConfig:
    """Configuration for converting one legacy ALF/msld-py-prep folder."""

    input_folder: str | Path
    output_folder: str | Path | None = None
    setup_script: str | None = None
    force: bool = False
    ph_enabled: bool = False
    temperature: float | None = None
    toppar_dir: str | Path | None = None
    topology_files: list[str] = field(default_factory=list)
    replace_legacy_toppar: bool = False
    debug: bool = False


@dataclass
class LegacyConvertResult:
    """Result returned after conversion or cache reuse."""

    legacy_folder: Path
    output_folder: Path
    setup_script: str
    manifest_path: Path
    source_hash: str
    reused: bool
    legacy_ph: float | None
    extra_files: list[Path]


def find_legacy_setup_script(prep_dir: str | Path, setup_script: str | None = None) -> str:
    """Find the main legacy CHARMM setup script in ``prep_dir``."""
    prep_dir = Path(prep_dir).resolve()
    if setup_script is not None:
        path = prep_dir / setup_script
        if not path.exists():
            raise FileNotFoundError(f"Legacy setup script not found: {path}")
        return setup_script

    exclude = {"lpsites.inp", "toppar.str"}
    candidates = sorted(f.name for f in prep_dir.glob("*.inp") if f.name not in exclude)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No .inp setup script found in {prep_dir}. Set legacy_setup_script."
        )
    raise FileNotFoundError(
        f"Multiple .inp files in {prep_dir}: {candidates}. Set legacy_setup_script."
    )


def parse_legacy_alf_info(prep_dir: str | Path) -> dict[str, Any]:
    """Parse ``prep/alf_info.py`` and return a normalized dictionary."""
    prep_dir = Path(prep_dir).resolve()
    alf_info_path = prep_dir / "alf_info.py"
    if not alf_info_path.exists():
        raise FileNotFoundError(f"Required file not found: {alf_info_path}")

    namespace: dict[str, Any] = {"np": np}
    with _pushd(prep_dir.parent):
        with _legacy_alf_info_environment(), redirect_stdout(io.StringIO()):
            exec(alf_info_path.read_text(), namespace)
    if "alf_info" not in namespace:
        raise ValueError(f"alf_info.py must define 'alf_info' dict: {alf_info_path}")

    alf_info = dict(namespace["alf_info"])
    if "nsubs" not in alf_info:
        raise ValueError(f"Legacy alf_info.py must define alf_info['nsubs']: {alf_info_path}")

    alf_info["nsubs"] = [int(x) for x in alf_info["nsubs"]]
    alf_info["nblocks"] = int(sum(alf_info["nsubs"]))
    return alf_info


@contextmanager
def _pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@contextmanager
def _legacy_alf_info_environment():
    old_charmmexec = os.environ.get("CHARMMEXEC")
    os.environ.setdefault("CHARMMEXEC", "charmm")
    try:
        yield
    finally:
        if old_charmmexec is None:
            os.environ.pop("CHARMMEXEC", None)
        else:
            os.environ["CHARMMEXEC"] = old_charmmexec


def parse_legacy_setup_metadata(
    setup_script: str | Path,
    *,
    prep_dir: str | Path | None = None,
) -> LegacySetupMetadata:
    """Parse common msld-py-prep variables from a legacy setup script."""
    setup_path = Path(setup_script)
    prep = (Path(prep_dir) if prep_dir is not None else setup_path.parent).resolve()
    text = setup_path.read_text()

    metadata = LegacySetupMetadata()
    metadata.ligseg = _parse_set_value(text, "ligseg", metadata.ligseg).upper()
    metadata.resname = (_parse_legacy_core_resname(prep) or metadata.ligseg).upper()
    metadata.resnum = _parse_set_value(text, "resnum", metadata.resnum)

    box = _parse_set_value(text, "box", None)
    if box is not None:
        metadata.box = _try_float(box)
    temp = _parse_set_value(text, "temp", None)
    if temp is not None:
        metadata.temp = _try_float(temp)

    ph_match = re.search(r"!\s*RX!\s+phmd\s+ph\s+([+-]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if ph_match:
        metadata.legacy_ph = float(ph_match.group(1))

    parameter_files: list[Path] = []
    for match in re.finditer(
        r"(?im)^\s*read\s+param\b.*?\bname\s+(.+?)\s*(?:!.*)?$",
        text,
    ):
        path_text = match.group(1).strip()
        path = _resolve_legacy_path(path_text, prep)
        if path.name and path.suffix.lower() in {".prm", ".str"}:
            parameter_files.append(path)
    parameter_files.extend(_legacy_unit_parameter_files(text, prep))
    for match in re.finditer(
        r"""(?im)\bread\.prm\s*\(\s*['"]([^'"]+)['"]""",
        text,
    ):
        path = _resolve_legacy_path(match.group(1).strip(), prep)
        if path.name and path.suffix.lower() in {".prm", ".str"}:
            parameter_files.append(path)
    metadata.parameter_files = _dedupe_paths(parameter_files)
    return metadata


def _legacy_unit_parameter_files(text: str, prep_dir: Path) -> list[Path]:
    unit_paths: dict[str, Path] = {}
    for match in re.finditer(
        r"(?im)^\s*open\s+read\s+card\s+unit\s+(\d+)\s+name\s+(.+?)\s*(?:!.*)?$",
        text,
    ):
        unit, path_text = match.groups()
        unit_paths[unit] = _resolve_legacy_path(path_text, prep_dir)

    parameter_files: list[Path] = []
    for match in re.finditer(r"(?im)^\s*read\s+para(?:m)?\s+card\s+unit\s+(\d+)\b", text):
        path = unit_paths.get(match.group(1))
        if path is not None and path.suffix.lower() in {".prm", ".str"}:
            parameter_files.append(path)
    return parameter_files


def build_legacy_patches_dat(
    prep_dir: str | Path,
    nsubs: list[int],
    *,
    ligseg: str = "LIG",
    resname: str | None = None,
    resnum: str = "1",
    setup_script: str | Path | None = None,
) -> list[dict[str, str]]:
    """Build ``patches.dat`` rows from legacy patch RTFs and LDIN tags."""
    prep = Path(prep_dir)
    base_resname = (resname or ligseg).upper()
    tags = _legacy_ldin_tags(setup_script, nsubs) if setup_script is not None else {}
    script_atoms = _legacy_site_definition_atoms(setup_script) if setup_script is not None else {}
    source_subsites = (
        _legacy_block_call_site_subs(setup_script, nsubs)
        if setup_script is not None
        else _sequential_site_subs(nsubs)
    )
    rows: list[dict[str, str]] = []
    source_index = 0
    for site_idx, nsub in enumerate(nsubs, start=1):
        for sub_idx in range(1, nsub + 1):
            source_site_idx, source_sub_idx = source_subsites[source_index]
            source_index += 1
            rtf = prep / f"site{source_site_idx}_sub{source_sub_idx}_pres.rtf"
            atoms = script_atoms.get((source_site_idx, source_sub_idx))
            if atoms is None and rtf.exists():
                atoms = _read_patch_atoms(rtf)
            if not atoms:
                raise FileNotFoundError(f"Legacy patch RTF not found: {rtf}")
            rows.append(
                {
                    "SEGID": ligseg.upper(),
                    "RESID": str(resnum),
                    "PATCH": f"p{source_site_idx}_{source_sub_idx}",
                    "_BASE_RESNAME": base_resname,
                    "SELECT": f"s{site_idx}s{sub_idx}",
                    "ATOMS": " ".join(atoms),
                    "TAG": tags.get((site_idx, sub_idx), "NONE"),
                }
            )
    return rows


def _sequential_site_subs(nsubs: list[int]) -> list[tuple[int, int]]:
    return [
        (site_idx, sub_idx)
        for site_idx, nsub in enumerate(nsubs, start=1)
        for sub_idx in range(1, nsub + 1)
    ]


def _legacy_block_call_site_subs(
    setup_script: str | Path,
    nsubs: list[int],
) -> list[tuple[int, int]]:
    call_map: dict[int, tuple[int, int]] = {}
    for raw_line in Path(setup_script).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!"):
            continue
        match = re.search(r"(?i)\bcall\s+(\d+)\b.*\bsite(\d+)_sub(\d+)\b", line)
        if match is not None:
            call_map[int(match.group(1))] = (int(match.group(2)), int(match.group(3)))

    if not call_map:
        return _sequential_site_subs(nsubs)

    source_subsites: list[tuple[int, int]] = []
    block = 1
    for site_idx, nsub in enumerate(nsubs, start=1):
        for sub_idx in range(1, nsub + 1):
            block += 1
            source_subsites.append(call_map.get(block, (site_idx, sub_idx)))
    return source_subsites


def convert_legacy_system(config: LegacyConvertConfig) -> LegacyConvertResult:
    """Convert a legacy msld-py-prep folder to a cached modern prep folder."""
    legacy_folder = Path(config.input_folder).resolve()
    prep_dir = legacy_folder / "prep"
    if not prep_dir.exists():
        raise FileNotFoundError(f"Legacy input must contain prep/: {legacy_folder}")

    setup_name = find_legacy_setup_script(prep_dir, config.setup_script)
    setup_path = prep_dir / setup_name
    alf_info = parse_legacy_alf_info(prep_dir)
    metadata = parse_legacy_setup_metadata(setup_path, prep_dir=prep_dir)
    if metadata.box is None and "box" in alf_info:
        metadata.box = float(alf_info["box"])
    if metadata.temp is None:
        if config.temperature is not None:
            metadata.temp = float(config.temperature)
        elif "temp" in alf_info:
            metadata.temp = float(alf_info["temp"])
    if metadata.box is None:
        raise ValueError(
            "Legacy conversion requires a cubic box size in alf_info['box'] "
            "or in the setup script via 'set box = <value>'."
        )

    if metadata.legacy_ph is not None and not config.ph_enabled:
        warnings.warn(
            "Legacy setup contains '!RX! phmd ph ...', but modern config omits pH; "
            "keeping pH disabled.",
            UserWarning,
            stacklevel=2,
        )

    output_folder = _resolve_output_folder(legacy_folder, config.output_folder)
    output_prep = output_folder / "prep"
    manifest_path = output_folder / "legacy_import.json"

    sources = _legacy_source_files(prep_dir, setup_path, alf_info["nsubs"])
    source_hash = _hash_sources(sources)
    if not config.force and _cache_is_valid(manifest_path, output_prep, source_hash):
        manifest = json.loads(manifest_path.read_text())
        return LegacyConvertResult(
            legacy_folder=legacy_folder,
            output_folder=output_folder,
            setup_script=setup_name,
            manifest_path=manifest_path,
            source_hash=source_hash,
            reused=True,
            legacy_ph=manifest.get("legacy_ph"),
            extra_files=[Path(p) for p in manifest.get("extra_files", [])],
        )

    output_prep.mkdir(parents=True, exist_ok=True)
    rows = build_legacy_patches_dat(
        prep_dir,
        alf_info["nsubs"],
        ligseg=metadata.ligseg,
        resname=metadata.resname,
        resnum=metadata.resnum,
        setup_script=setup_path,
    )
    _validate_legacy_patch_rows_against_prebuilt_psf(prep_dir, rows, setup_name)
    _write_patches_dat(output_prep / "patches.dat", rows)
    _write_box_files(output_prep, float(metadata.box))
    extra_files = _copy_extra_files(metadata.parameter_files, output_prep)
    _copy_optional_file(prep_dir / "restrains.str", output_prep / "restrains.str")
    _copy_optional_file(prep_dir / "system_min.crd", output_prep / "system_min.crd")

    _write_scalar_files(output_prep, alf_info, metadata, output_folder.name)
    _write_legacy_structure(config, prep_dir, setup_path, output_prep, alf_info, metadata, rows)

    manifest = {
        "version": MANIFEST_VERSION,
        "source_hash": source_hash,
        "legacy_folder": str(legacy_folder),
        "setup_script": setup_name,
        "nsubs": alf_info["nsubs"],
        "ligseg": metadata.ligseg,
        "resname": metadata.resname,
        "resnum": metadata.resnum,
        "legacy_ph": metadata.legacy_ph,
        "extra_files": [str(p) for p in extra_files],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return LegacyConvertResult(
        legacy_folder=legacy_folder,
        output_folder=output_folder,
        setup_script=setup_name,
        manifest_path=manifest_path,
        source_hash=source_hash,
        reused=False,
        legacy_ph=metadata.legacy_ph,
        extra_files=extra_files,
    )


def _parse_set_value(text: str, name: str, default: str | None) -> str | None:
    match = re.search(rf"(?im)^\s*set\s+{re.escape(name)}\s*=\s*(\S+)", text)
    return match.group(1) if match else default


def _try_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_legacy_core_resname(prep_dir: Path) -> str | None:
    core_rtf = prep_dir / "core.rtf"
    if core_rtf.exists():
        for line in core_rtf.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].upper() == "RESI":
                return parts[1]

    core_pdb = prep_dir / "core.pdb"
    if core_pdb.exists():
        for line in core_pdb.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                parts = line.split()
                if len(parts) >= 4:
                    return parts[3]
    return None


def _resolve_legacy_path(path_text: str, prep_dir: Path) -> Path:
    path_text = path_text.strip().strip('"').strip("'")
    path_text = path_text.replace("@builddir", str(prep_dir))
    path = Path(path_text)
    if not path.is_absolute():
        if path.parts and path.parts[0] == prep_dir.name:
            path = prep_dir.parent / path
        else:
            path = prep_dir / path
    return path.resolve()


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _read_patch_atoms(rtf_path: Path) -> list[str]:
    if not rtf_path.exists():
        raise FileNotFoundError(f"Legacy patch RTF not found: {rtf_path}")
    atoms: list[str] = []
    for line in rtf_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].upper() == "ATOM":
            atoms.append(parts[1])
    if not atoms:
        raise ValueError(f"No ATOM records found in legacy patch RTF: {rtf_path}")
    return atoms


def _legacy_site_definition_atoms(setup_script: str | Path) -> dict[tuple[int, int], list[str]]:
    definitions: dict[tuple[int, int], list[str]] = {}
    current: tuple[int, int] | None = None
    atoms: list[str] = []

    for raw_line in Path(setup_script).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!"):
            continue
        match = re.match(r"(?i)^define\s+site(\d+)_sub(\d+)\b", line)
        if match:
            if current is not None and atoms:
                definitions[current] = atoms
            current = (int(match.group(1)), int(match.group(2)))
            atoms = []
            continue
        if current is None:
            continue
        for atom_match in re.finditer(r"(?i)\batom\s+\S+\s+\S+\s+(\S+)", line):
            atom = atom_match.group(1).strip()
            if atom.upper() != "NONE":
                atoms.append(atom)
        if re.search(r"(?i)\bend\b", line):
            if atoms:
                definitions[current] = atoms
            current = None
            atoms = []
    if current is not None and atoms:
        definitions[current] = atoms
    return definitions


def _legacy_ldin_tags(setup_script: str | Path, nsubs: list[int]) -> dict[tuple[int, int], str]:
    tags: dict[tuple[int, int], str] = {}
    block_to_site_sub: dict[int, tuple[int, int]] = {}
    block = 1
    for site_idx, nsub in enumerate(nsubs, start=1):
        for sub_idx in range(1, nsub + 1):
            block += 1
            block_to_site_sub[block] = (site_idx, sub_idx)

    for line in Path(setup_script).read_text().splitlines():
        if "!RX!" not in line or not line.lstrip().lower().startswith("ldin"):
            continue
        head, rx = line.split("!RX!", 1)
        head_tokens = head.split()
        if len(head_tokens) < 2:
            continue
        try:
            block_idx = int(head_tokens[1])
        except ValueError:
            continue
        site_sub = block_to_site_sub.get(block_idx)
        if site_sub is None:
            continue
        rx_tokens = rx.split()
        tag = rx_tokens[0].upper() if rx_tokens else "NONE"
        if tag == "NONE" or len(rx_tokens) < 2:
            tags[site_sub] = tag
        else:
            tags[site_sub] = f"{tag} {rx_tokens[1]}"
    return tags


def _resolve_output_folder(legacy_folder: Path, output_folder: str | Path | None) -> Path:
    if output_folder is None:
        return legacy_folder / ".cphmd" / "converted"
    output = Path(output_folder)
    if not output.is_absolute():
        output = legacy_folder / output
    return output.resolve()


def _legacy_source_files(prep_dir: Path, setup_path: Path, nsubs: list[int]) -> list[Path]:
    files = [
        prep_dir / "alf_info.py",
        setup_path,
        prep_dir / "core.pdb",
        prep_dir / "core.rtf",
        prep_dir / "full_ligand.prm",
        prep_dir / "solvent.pdb",
        prep_dir / "lpsites.inp",
        prep_dir / "restrains.str",
        prep_dir / "system_min.crd",
        prep_dir / "minimized.psf",
        prep_dir / "minimized.crd",
        prep_dir / "minimized.pdb",
    ]
    for site_idx, sub_idx in _legacy_block_call_site_subs(setup_path, nsubs):
        files.append(prep_dir / f"site{site_idx}_sub{sub_idx}_pres.rtf")
        files.append(prep_dir / f"site{site_idx}_sub{sub_idx}_frag.pdb")
    return [path for path in files if path.exists()]


def _hash_sources(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: str(p)):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _cache_is_valid(manifest_path: Path, output_prep: Path, source_hash: str) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False
    if manifest.get("version") != MANIFEST_VERSION:
        return False
    if manifest.get("source_hash") != source_hash:
        return False
    required_exist = all((output_prep / name).exists() for name in _REQUIRED_PREP_FILES)
    extras_exist = all(Path(path).exists() for path in manifest.get("extra_files", []))
    return required_exist and extras_exist


def _write_patches_dat(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w") as f:
        f.write("SEGID,RESID,PATCH,SELECT,ATOMS,TAG\n")
        for row in rows:
            f.write("{SEGID},{RESID},{PATCH},{SELECT},{ATOMS},{TAG}\n".format(**row))


def _write_box_files(output_prep: Path, box: float) -> None:
    with (output_prep / "box.dat").open("w") as f:
        f.write("CUBic\n")
        f.write(f"{box} {box} {box}\n")
        f.write("90.0 90.0 90.0\n")
    (output_prep / "size.dat").write_text(f"{box}\n")
    fft = _fft_number(box)
    (output_prep / "fft.dat").write_text(f"{fft} {fft} {fft}\n")


def _fft_number(length: float) -> int:
    """Find the smallest even 2/3/5-smooth FFT number >= ``length``."""
    n = max(2, int(np.ceil(length)))
    limit = max(n * 2, 256)
    for candidate in range(2, limit + 1):
        value = candidate
        for factor in (2, 3, 5):
            while value % factor == 0:
                value //= factor
        if value == 1 and candidate % 2 == 0 and candidate >= n:
            return candidate
    return n if n % 2 == 0 else n + 1


def _copy_extra_files(paths: list[Path], output_prep: Path) -> list[Path]:
    copied: list[Path] = []
    for src in paths:
        if not src.exists():
            raise FileNotFoundError(f"Legacy parameter file not found: {src}")
        dst = output_prep / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        copied.append(dst.resolve())
    return _dedupe_paths(copied)


def _copy_optional_file(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def _write_scalar_files(
    output_prep: Path,
    alf_info: dict[str, Any],
    metadata: LegacySetupMetadata,
    name: str,
) -> None:
    values = {
        "name": name,
        "nsubs": " ".join(str(x) for x in alf_info["nsubs"]),
        "nblocks": str(int(sum(alf_info["nsubs"]))),
        "ncentral": str(alf_info.get("ncentral", 0)),
        "nnodes": str(alf_info.get("nnodes", 1)),
        "nreps": str(alf_info.get("nreps", 1)),
        "temp": str(metadata.temp if metadata.temp is not None else alf_info.get("temp", 298.15)),
        "engine": "charmm",
        "fnex": str(alf_info.get("fnex", 5.5)),
        "cutlsum": str(alf_info.get("cutlsum", 0.8)),
        "g_imp_bins": str(alf_info.get("g_imp_bins", 32)),
        "ntersite": str(alf_info.get("ntersite", [1, 1])),
    }
    for key, value in values.items():
        (output_prep / key).write_text(f"{value}\n")


def _load_replacement_toppar(toppar_dir: Path, topology_files: list[str]) -> None:
    prm_files = [f for f in topology_files if f.endswith(".prm")]
    rtf_files = [f for f in topology_files if f.endswith(".rtf")]
    str_files = [f for f in topology_files if f.endswith(".str")]

    for index, filename in enumerate(rtf_files):
        system.read_rtf(toppar_dir / filename, append=index > 0)
    for index, filename in enumerate(prm_files):
        system.read_param(toppar_dir / filename, append=index > 0)
    for filename in str_files:
        system.stream_file(toppar_dir / filename)


def _write_legacy_structure(
    config: LegacyConvertConfig,
    prep_dir: Path,
    setup_path: Path,
    output_prep: Path,
    alf_info: dict[str, Any],
    metadata: LegacySetupMetadata,
    patch_rows: list[dict[str, str]],
) -> None:
    if _has_prebuilt_legacy_structure(prep_dir):
        _copy_prebuilt_legacy_structure(prep_dir, output_prep, patch_rows)
        return

    if not config.debug:
        system.set_prnlev(-1)
        system.set_warn_level(-1)
    system.set_bomb_level(-2)
    system.delete_atoms(AtomSelection())

    skip_legacy_toppar = config.replace_legacy_toppar
    if skip_legacy_toppar:
        if not config.toppar_dir or not config.topology_files:
            raise ValueError("replace_legacy_toppar=True requires toppar_dir and topology_files.")
        _load_replacement_toppar(Path(config.toppar_dir), config.topology_files)

    script = _structure_only_script(
        setup_path.read_text(),
        prep_dir=Path(prep_dir.name),
        box=float(metadata.box),
        temp=metadata.temp if metadata.temp is not None else alf_info.get("temp", 298.15),
        skip_legacy_toppar=skip_legacy_toppar,
        patch_rows=patch_rows,
        nsubs=alf_info["nsubs"],
    )
    processed = output_prep / ".legacy_structure_build.inp"
    processed.write_text(script)
    try:
        with _pushd(prep_dir.parent):
            system.stream_file(processed)
        system.write_psf(output_prep / "system.psf")
        system.write_coor(output_prep / "system.crd")
        system.write_coor_pdb(output_prep / "system.pdb")
    finally:
        processed.unlink(missing_ok=True)
        system.crystal_free()
        system.delete_atoms(AtomSelection())
        system.set_bomb_level(0)
        if not config.debug:
            system.set_warn_level(5)
            system.set_prnlev(5)


def _has_prebuilt_legacy_structure(prep_dir: Path) -> bool:
    return all((prep_dir / f"minimized.{suffix}").exists() for suffix in ("psf", "crd"))


def _validate_legacy_patch_rows_against_prebuilt_psf(
    prep_dir: Path,
    patch_rows: list[dict[str, str]],
    setup_script: str,
) -> None:
    psf_path = _prebuilt_legacy_psf(prep_dir)
    if psf_path is None:
        return
    psf_atoms = _read_psf_atom_keys(psf_path)
    if not psf_atoms:
        return

    empty_rows: list[str] = []
    for row in patch_rows:
        segid = row["SEGID"].upper()
        resid = str(row["RESID"])
        row_atoms = row["ATOMS"].split()
        if not any((segid, resid, atom.upper()) in psf_atoms for atom in row_atoms):
            empty_rows.append(f"{row['PATCH']}/{row['SELECT']}")
    if empty_rows:
        sample = ", ".join(empty_rows[:5])
        raise ValueError(
            f"Legacy setup script {setup_script!r} generated alchemical selections with no "
            f"matching atoms in {psf_path.name}: {sample}. Check legacy_setup_script; use "
            "the grouped final simulation script, not a full-library build template."
        )


def _prebuilt_legacy_psf(prep_dir: Path) -> Path | None:
    for name in ("minimized.psf", "system.psf"):
        path = prep_dir / name
        if path.exists():
            return path
    return None


def _read_psf_atom_keys(path: Path) -> set[tuple[str, str, str]]:
    atoms: set[tuple[str, str, str]] = set()
    remaining_atoms: int | None = None
    for line in path.read_text().splitlines():
        if remaining_atoms is None and "!NATOM" in line:
            remaining_atoms = int(line.split()[0])
            continue
        if remaining_atoms:
            parts = line.split()
            if len(parts) >= 5:
                atoms.add((parts[1].upper(), str(parts[2]), parts[4].upper()))
            remaining_atoms -= 1
    return atoms


def _copy_prebuilt_legacy_structure(
    prep_dir: Path,
    output_prep: Path,
    patch_rows: list[dict[str, str]],
) -> None:
    for suffix in ("psf", "crd"):
        shutil.copy2(prep_dir / f"minimized.{suffix}", output_prep / f"system.{suffix}")
    if (prep_dir / "minimized.pdb").exists():
        shutil.copy2(prep_dir / "minimized.pdb", output_prep / "system.pdb")
    elif (prep_dir / "solvent.pdb").exists():
        shutil.copy2(prep_dir / "solvent.pdb", output_prep / "system.pdb")


def _structure_only_script(
    script_text: str,
    *,
    prep_dir: Path,
    box: float,
    temp: float,
    skip_legacy_toppar: bool,
    patch_rows: list[dict[str, str]],
    nsubs: list[int],
) -> str:
    lines: list[str] = []
    inserted_counts = False
    for line in script_text.splitlines():
        if re.match(r"(?i)^\s*BLOCK\b", line):
            break
        if skip_legacy_toppar and re.match(r"(?i)^\s*stream\b.*toppar", line.strip()):
            lines.append("! Legacy toppar.str skipped; topology files preloaded")
            continue
        line = re.sub(r"(?i)^\s*set\s+builddir\s*=\s*\S+", f"set builddir = {prep_dir}", line)
        line = re.sub(r"(?i)^\s*set\s+box\s*=\s*\S+", f"set box = {box}", line)
        line = re.sub(r"(?i)^\s*set\s+temp\s*=\s*\S+", f"set temp = {temp}", line)
        if skip_legacy_toppar:
            line = re.sub(r"(?i)^(\s*read\s+rtf\s+)card\b", r"\1append card", line)
            line = re.sub(r"(?i)^(\s*read\s+para\s+)card\b", r"\1append card", line)
        lines.append(line)
        if not inserted_counts and re.match(r"(?i)^\s*bomblev\b", line):
            lines.extend(_legacy_count_parameter_lines(nsubs))
            inserted_counts = True
    lines.append("")
    lines.append("return")
    return "\n".join(lines) + "\n"


def _legacy_count_parameter_lines(nsubs: list[int]) -> list[str]:
    lines = [f"set nsites = {len(nsubs)}", f"set nblocks = {sum(nsubs)}"]
    lines.extend(f"set nsubs{idx} = {value}" for idx, value in enumerate(nsubs, start=1))
    return lines


def _rewrite_legacy_state_resnames(
    path: Path,
    patch_rows: list[dict[str, str]],
    fmt: str,
) -> None:
    """Patch CHARMM output files so legacy state atoms have state residue names."""
    queues = _legacy_state_resname_queues(patch_rows)
    lines = path.read_text().splitlines(keepends=True)

    if fmt == "psf":
        rewritten = _rewrite_psf_state_resnames(lines, queues)
    elif fmt == "crd":
        rewritten = _rewrite_crd_state_resnames(lines, queues)
    elif fmt == "pdb":
        rewritten = _rewrite_pdb_state_resnames(lines, queues)
    else:
        raise ValueError(f"Unsupported legacy rewrite format: {fmt}")

    missing = {key: list(values) for key, values in queues.items() if values}
    if missing:
        sample = ", ".join(f"{key}:{values[0][0]}" for key, values in list(missing.items())[:5])
        raise ValueError(f"Could not assign legacy state residue names in {path}: {sample}")

    path.write_text("".join(rewritten))


def _legacy_state_resname_queues(
    patch_rows: list[dict[str, str]],
) -> dict[tuple[str, str, str], deque[tuple[str, int]]]:
    assignments: dict[tuple[str, str, str], deque[tuple[str, int]]] = defaultdict(deque)
    for state_offset, row in enumerate(patch_rows, start=1):
        key_prefix = (
            row["SEGID"].upper(),
            str(row["RESID"]),
        )
        patch = row["PATCH"].upper()
        for atom in row["ATOMS"].split():
            assignments[(*key_prefix, atom.upper())].append((patch, state_offset))
    return assignments


def _rewrite_psf_state_resnames(
    lines: list[str],
    queues: dict[tuple[str, str, str], deque[tuple[str, int]]],
) -> list[str]:
    rewritten: list[str] = []
    remaining_atoms: int | None = None
    for line in lines:
        if remaining_atoms is None and "!NATOM" in line:
            remaining_atoms = int(line.split()[0])
            rewritten.append(line)
            continue
        if remaining_atoms:
            line = _rewrite_tokenized_atom_line(line, queues, key_idx=(1, 2, 4), resname_idx=3)
            remaining_atoms -= 1
        rewritten.append(line)
    return rewritten


def _rewrite_crd_state_resnames(
    lines: list[str],
    queues: dict[tuple[str, str, str], deque[tuple[str, int]]],
) -> list[str]:
    rewritten: list[str] = []
    remaining_atoms: int | None = None
    current_identity: tuple[str, str, str] | None = None
    current_resseq = 0
    for line in lines:
        parts = line.split()
        if remaining_atoms is None and len(parts) >= 2 and parts[1].upper() == "EXT":
            remaining_atoms = int(parts[0])
            rewritten.append(line)
            continue
        if remaining_atoms:
            line = _rewrite_crd_atom_line(
                line,
                queues,
                key_idx=(7, 8, 3),
                resname_idx=2,
            )
            tokens = list(re.finditer(r"\S+", line))
            if len(tokens) > 8:
                identity = (
                    tokens[7].group().upper(),
                    tokens[8].group(),
                    tokens[2].group().upper(),
                )
                if identity != current_identity:
                    current_identity = identity
                    current_resseq += 1
                line = f"{line[:10]}{current_resseq:10d}{line[20:]}"
            remaining_atoms -= 1
        rewritten.append(line)
    return rewritten


def _rewrite_pdb_state_resnames(
    lines: list[str],
    queues: dict[tuple[str, str, str], deque[tuple[str, int]]],
) -> list[str]:
    rewritten: list[str] = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            line = _rewrite_tokenized_atom_line(line, queues, key_idx=(10, 4, 2), resname_idx=3)
        rewritten.append(line)
    return rewritten


def _rewrite_tokenized_atom_line(
    line: str,
    queues: dict[tuple[str, str, str], deque[tuple[str, int]]],
    *,
    key_idx: tuple[int, int, int],
    resname_idx: int,
) -> str:
    tokens = list(re.finditer(r"\S+", line))
    max_idx = max(*key_idx, resname_idx)
    if len(tokens) <= max_idx:
        return line

    segid = tokens[key_idx[0]].group().upper()
    resid = tokens[key_idx[1]].group()
    atom = tokens[key_idx[2]].group().upper()
    queue = queues.get((segid, resid, atom))
    if not queue:
        return line

    next_token = tokens[resname_idx + 1] if len(tokens) > resname_idx + 1 else None
    return _replace_token(line, tokens[resname_idx], queue.popleft()[0], next_token)


def _rewrite_crd_atom_line(
    line: str,
    queues: dict[tuple[str, str, str], deque[tuple[str, int]]],
    *,
    key_idx: tuple[int, int, int],
    resname_idx: int,
) -> str:
    tokens = list(re.finditer(r"\S+", line))
    max_idx = max(*key_idx, resname_idx)
    if len(tokens) <= max_idx:
        return line

    segid = tokens[key_idx[0]].group().upper()
    resid = tokens[key_idx[1]].group()
    atom = tokens[key_idx[2]].group().upper()
    queue = queues.get((segid, resid, atom))

    if queue:
        patch, _state_offset = queue.popleft()
        next_resname = tokens[resname_idx + 1] if len(tokens) > resname_idx + 1 else None
        return _replace_token(line, tokens[resname_idx], patch, next_resname)
    return line


def _replace_token(
    line: str,
    token: re.Match[str],
    value: str,
    next_token: re.Match[str] | None = None,
) -> str:
    end = next_token.start() if next_token is not None else token.end()
    width = end - token.start()
    return f"{line[:token.start()]}{value:<{width}}{line[end:]}"
