"""
Solvation module for CpHMD system preparation.

This backend uses crimm for solvent box construction and ion placement, then
reloads the solvated PSF/CRD into pyCHARMM for optional restrained minimization
and final artifact generation.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from cphmd import TOPPAR_DIR
from cphmd.native import system
from cphmd.native.types import AtomSelection

# Historical CLI/config values are kept for compatibility, even though the
# crimm backend currently supports only the cubic and octahedral cases.
CrystalType = Literal[
    "CUBIC",
    "TETRAGONAL",
    "ORTHORHOMBIC",
    "MONOCLINIC",
    "TRICLINIC",
    "HEXAGONAL",
    "RHOMBOHEDRAL",
    "OCTAHEDRAL",
    "RHDO",
]

IonMethod = Literal["AN", "SLTCAP"]

_SUPPORTED_CRYSTAL_TYPES = {
    "CUBIC": "cube",
    "OCTAHEDRAL": "octa",
}
_ION_METHOD_MAP = {
    "AN": "add_neutralize",
    "SLTCAP": "sltcap",
}
_DEFAULT_BOX_ANGLES = {
    "CUBIC": (90.0, 90.0, 90.0),
    "OCTAHEDRAL": (109.4712206344907, 109.4712206344907, 109.4712206344907),
}
_MIN_SOLUTE_ION_DISTANCE = 2.6
_WATERBOX_EXPR = "segid SOLV .or. segid IONS"
_MOLECULE_EXPR = ".not. (segid SOLV .or. segid IONS)"
_WATERBOX_SELECTION = f"{_WATERBOX_EXPR} end"
_MOLECULE_SELECTION = f"{_MOLECULE_EXPR} end"


@dataclass
class SolvationConfig:
    """Configuration for solvation.

    Attributes:
        input_file: Path to input structure (without extension).
        output_dir: Output directory.
        crystal_type: Crystal box type.
        padding: Padding around molecule in Angstroms.
        salt_concentration: Salt concentration in M.
        positive_ion: Positive ion type (e.g., "POT", "SOD").
        negative_ion: Negative ion type (e.g., "CLA").
        temperature: Temperature in Kelvin.
        skip_ions: If True, skip ion placement.
        ion_method: Ion placement algorithm ("AN" or "SLTCAP").
        min_ion_distance: Minimum distance between ions in Angstroms.
        minimize: If True, run a short restrained pyCHARMM minimization after solvation.
        toppar_dir: Path to topology directory.
        topology_files: List of topology files to load.
        extra_files: Additional topology/parameter files for custom ligands.
        selected_residues: Optional patch-style residue filters used to decide
            which explicit ligand reference states should affect ion counts.
        ligand_patches: Optional patch metadata copied from the patch config.
    """

    input_file: str | Path
    output_dir: str | Path = "solvated"
    crystal_type: CrystalType = "OCTAHEDRAL"
    padding: float = 10.0
    salt_concentration: float = 0.10
    positive_ion: str = "POT"
    negative_ion: str = "CLA"
    temperature: float = 298.15
    skip_ions: bool = False
    ion_method: IonMethod = "SLTCAP"
    min_ion_distance: float = 5.0
    minimize: bool = False
    toppar_dir: Path | None = None
    topology_files: list[str] = field(
        default_factory=lambda: [
            "top_all36_prot.rtf",
            "par_all36m_prot.prm",
            "toppar_water_ions.str",
            "top_all36_na.rtf",
            "par_all36_na.prm",
            "top_all36_cgenff.rtf",
            "par_all36_cgenff.prm",
        ]
    )
    extra_files: list[str | Path] = field(default_factory=list)
    selected_residues: list[str] = field(default_factory=list)
    ligand_patches: list[object] = field(default_factory=list)

    def __post_init__(self):
        if self.toppar_dir is not None:
            self.toppar_dir = Path(self.toppar_dir)


def _input_base_path(input_file: str | Path) -> Path:
    path = Path(input_file)
    return path.with_suffix("") if path.suffix else path


def _normalize_crystal_type(crystal_type: str) -> str:
    normalized = crystal_type.upper()
    if normalized not in _SUPPORTED_CRYSTAL_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_CRYSTAL_TYPES))
        raise ValueError(
            f"crimm solvation backend currently supports only {supported}. "
            f"Received crystal_type={crystal_type!r}."
        )
    return _SUPPORTED_CRYSTAL_TYPES[normalized]


def _normalize_ion_method(ion_method: str) -> str:
    normalized = ion_method.upper()
    if normalized not in _ION_METHOD_MAP:
        supported = ", ".join(sorted(_ION_METHOD_MAP))
        raise ValueError(f"Unsupported ion_method {ion_method!r}. Supported values: {supported}.")
    return _ION_METHOD_MAP[normalized]


def _default_box_angles(crystal_type: str) -> tuple[float, float, float]:
    normalized = crystal_type.upper()
    if normalized not in _DEFAULT_BOX_ANGLES:
        raise ValueError(f"No default box angles available for crystal_type={crystal_type!r}.")
    return _DEFAULT_BOX_ANGLES[normalized]


def _coerce_integral_charge(
    charge: float,
    label: str = "charge",
    tolerance: float = 1e-3,
) -> int:
    rounded = int(round(charge))
    if abs(charge - rounded) > tolerance:
        raise ValueError(f"{label} must be integral for ion placement, got {charge:.6f}.")
    return rounded


def _normalize_ligand_patch_defs(ligand_patches: list[object]) -> list[object]:
    if not ligand_patches:
        return []

    from cphmd.core.patching import LigandPatchDef

    return [
        LigandPatchDef(**patch_def) if isinstance(patch_def, dict) else patch_def
        for patch_def in ligand_patches
    ]


def _has_explicit_reference_patches(ligand_patches: list[object]) -> bool:
    for patch_def in ligand_patches:
        if getattr(patch_def, "reference_patch", None):
            return True
        for site in getattr(patch_def, "sites", None) or []:
            if getattr(site, "reference_patch", None):
                return True
    return False


def _read_topology_files(config: SolvationConfig, verbose: bool = True) -> None:
    """Load CHARMM topology and parameter files for the pyCHARMM stage."""
    toppar_dir = config.toppar_dir or TOPPAR_DIR

    if not verbose:
        system.set_prnlev(0)

    system.set_bomb_level(-1)

    prm_files = [f for f in config.topology_files if f.endswith(".prm")]
    rtf_files = [f for f in config.topology_files if f.endswith(".rtf")]
    str_files = [f for f in config.topology_files if f.endswith(".str")]

    if rtf_files:
        system.read_rtf(toppar_dir / rtf_files[0])
        for filename in rtf_files[1:]:
            system.read_rtf(toppar_dir / filename, append=True)

    if prm_files:
        system.read_param(toppar_dir / prm_files[0])
        for filename in prm_files[1:]:
            system.read_param(toppar_dir / filename, append=True)

    for filename in str_files:
        system.stream_file(toppar_dir / filename)

    for extra_file in config.extra_files:
        extra_path = Path(extra_file)
        if extra_path.suffix == ".rtf":
            system.read_rtf(extra_path, append=True)
        elif extra_path.suffix == ".prm":
            system.read_param(extra_path, append=True)
        elif extra_path.suffix == ".str":
            system.stream_file(extra_path)

    for patch_def in _normalize_ligand_patch_defs(config.ligand_patches):
        _load_charmm_ligand_patch_file(Path(getattr(patch_def, "patch_file")))

    system.set_bomb_level(0)
    system.set_iofmt(extended=True)

    if not verbose:
        system.set_prnlev(5)


def _clear_charmm_atoms() -> None:
    if system.get_natom() > 0:
        system.delete_atoms(AtomSelection())


def _ensure_input_crd(input_base: Path) -> Path:
    psf_path = input_base.with_suffix(".psf")
    crd_path = input_base.with_suffix(".crd")
    pdb_path = input_base.with_suffix(".pdb")

    if crd_path.exists():
        return crd_path
    if not psf_path.exists():
        raise FileNotFoundError(f"Input PSF not found: {psf_path}")
    if not pdb_path.exists():
        raise FileNotFoundError(f"Input coordinates not found: {crd_path} or {pdb_path}")

    system.read_psf(psf_path)
    system.read_pdb(pdb_path, resid=True)
    system.write_coor(crd_path)
    _clear_charmm_atoms()
    return crd_path


def _check_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _split_stream_rtf_and_prm(stream_text: str, source: Path) -> tuple[str, list[str]]:
    lines = stream_text.splitlines()
    try:
        end_index = next(i for i, line in enumerate(lines) if line.strip().upper() == "END")
    except StopIteration as exc:
        raise ValueError(
            f"Unsupported stream topology file {source}. Expected a direct RTF/PRM toppar block."
        ) from exc

    rtf_block = "\n".join(lines[: end_index + 1])
    prm_lines = [line.rstrip() for line in lines[end_index + 1 :]]
    return rtf_block, prm_lines


def _extract_charmm_rtf_block(stream_text: str, source: Path) -> str:
    lines = stream_text.splitlines()
    try:
        start_index = (
            next(
                i
                for i, line in enumerate(lines)
                if line.strip().upper().startswith("READ RTF CARD")
            )
            + 1
        )
    except StopIteration as exc:
        raise ValueError(
            f"Unsupported ligand patch stream file {source}. "
            "Expected a 'read rtf card' topology block."
        ) from exc

    try:
        end_index = next(
            i
            for i, line in enumerate(lines[start_index:], start=start_index)
            if line.strip().upper() == "END"
        )
    except StopIteration as exc:
        raise ValueError(
            f"Unsupported ligand patch stream file {source}. "
            "Expected an END line terminating the RTF block."
        ) from exc

    rtf_lines = [line.rstrip() for line in lines[start_index : end_index + 1]]
    return "\n".join(rtf_lines) + "\n"


def _extract_charmm_prm_lines(stream_text: str, source: Path) -> list[str]:
    lines = stream_text.splitlines()
    try:
        start_index = (
            next(
                i
                for i, line in enumerate(lines)
                if line.strip().upper().startswith("READ PARAM CARD")
            )
            + 1
        )
    except StopIteration:
        return []

    try:
        end_index = next(
            i
            for i, line in enumerate(lines[start_index:], start=start_index)
            if line.strip().upper() == "END"
        )
    except StopIteration as exc:
        raise ValueError(
            f"Unsupported ligand patch stream file {source}. "
            "Expected an END line terminating the parameter block."
        ) from exc

    return [line.rstrip() for line in lines[start_index : end_index + 1]]


def _load_charmm_ligand_patch_file(patch_path: Path) -> None:
    suffix = patch_path.suffix.lower()
    if suffix == ".rtf":
        system.read_rtf(patch_path, append=True)
        return

    if suffix != ".str":
        system.read_rtf(patch_path, append=True)
        return

    rtf_block = _extract_charmm_rtf_block(
        patch_path.read_text(encoding="utf-8"),
        patch_path,
    )
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".rtf",
            prefix="cphmd_ligand_patch_",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(rtf_block)
            temp_path = Path(handle.name)
        system.read_rtf(temp_path, append=True)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _ensure_crimm_topology_set_compatibility(topology_set) -> None:
    """Provide attributes expected by TopologyGenerator.fill_ic().

    crimm's ResidueTopologySet exposes `patches` and `patched_defs`, but
    CGENFFTopologySet currently does not. The load_psf_crd() path still
    assumes those attributes exist when filling IC tables, so add empty
    containers for the ligand-only case.
    """
    if not hasattr(topology_set, "patches"):
        topology_set.patches = []
    if not hasattr(topology_set, "patched_defs"):
        topology_set.patched_defs = {}


def _ensure_crimm_cgenff_loaders(topology_generator):
    from crimm.Modeller.TopoLoader import CGENFFTopologySet, ParameterLoader

    cgenff_defs = topology_generator.res_def_dict.get("cgenff")
    if cgenff_defs is None:
        cgenff_defs = CGENFFTopologySet()
        topology_generator.res_def_dict["cgenff"] = cgenff_defs
    _ensure_crimm_topology_set_compatibility(cgenff_defs)

    cgenff_params = topology_generator.param_dict.get("cgenff")
    if cgenff_params is None:
        cgenff_params = ParameterLoader("cgenff")
        topology_generator.param_dict["cgenff"] = cgenff_params

    return cgenff_defs, cgenff_params


def _load_crimm_extra_files(topology_generator, extra_files: list[str | Path]) -> None:
    if not extra_files:
        return

    from crimm.IO.PRMParser import categorize_lines, parse_line_dict

    cgenff_defs, cgenff_params = _ensure_crimm_cgenff_loaders(topology_generator)

    for extra_file in extra_files:
        extra_path = Path(extra_file)
        if not extra_path.exists():
            raise FileNotFoundError(f"Extra topology file not found: {extra_path}")

        file_text = extra_path.read_text(encoding="utf-8")
        suffix = extra_path.suffix.lower()

        if suffix == ".rtf":
            if not cgenff_defs.load_rtf_block(file_text):
                raise ValueError(f"Failed to parse ligand topology file: {extra_path}")
            continue

        if suffix == ".prm":
            prm_lines = [line.rstrip() for line in file_text.splitlines()]
        elif suffix == ".str":
            rtf_block, prm_lines = _split_stream_rtf_and_prm(file_text, extra_path)
            if not cgenff_defs.load_rtf_block(rtf_block):
                raise ValueError(f"Failed to parse ligand topology block from: {extra_path}")
        else:
            raise ValueError(
                f"Unsupported extra topology file suffix for crimm solvation: {extra_path.suffix}"
            )

        if prm_lines:
            cgenff_params._raw_data_strings.extend(prm_lines)
            cgenff_params.param_dict.update(parse_line_dict(categorize_lines(prm_lines)))


def _load_crimm_ligand_patch_files(topology_generator, ligand_patches: list[object]) -> None:
    if not ligand_patches:
        return

    from crimm.IO.PRMParser import categorize_lines, parse_line_dict

    cgenff_defs, cgenff_params = _ensure_crimm_cgenff_loaders(topology_generator)

    for patch_def in ligand_patches:
        patch_path = Path(getattr(patch_def, "patch_file"))
        if not patch_path.exists():
            raise FileNotFoundError(f"Ligand patch file not found: {patch_path}")

        file_text = patch_path.read_text(encoding="utf-8")
        suffix = patch_path.suffix.lower()

        if suffix == ".rtf":
            rtf_block = file_text
            prm_lines: list[str] = []
        elif suffix == ".str":
            rtf_block = _extract_charmm_rtf_block(file_text, patch_path)
            prm_lines = _extract_charmm_prm_lines(file_text, patch_path)
        else:
            raise ValueError(
                "Unsupported ligand patch file suffix for crimm solvation: " f"{patch_path.suffix}"
            )

        if not cgenff_defs.load_rtf_block(rtf_block):
            raise ValueError(f"Failed to parse ligand patch topology file: {patch_path}")

        if prm_lines:
            cgenff_params._raw_data_strings.extend(prm_lines)
            cgenff_params.param_dict.update(parse_line_dict(categorize_lines(prm_lines)))


def _load_crimm_model(config: SolvationConfig, psf_path: Path, crd_path: Path):
    from crimm.Modeller.TopoLoader import TopologyGenerator

    topology_generator = TopologyGenerator()
    ligand_patches = _normalize_ligand_patch_defs(config.ligand_patches)
    _load_crimm_extra_files(topology_generator, config.extra_files)
    _load_crimm_ligand_patch_files(topology_generator, ligand_patches)

    try:
        model = topology_generator.load_psf_crd(str(psf_path), str(crd_path), QUIET=True)
    except Exception as exc:  # pragma: no cover - exercised in env-dependent runs
        extra_hint = ""
        if config.extra_files:
            extra_hint = (
                " Check that the supplied ligand toppar files match the " "PSF/CRD residue names."
            )
        raise ValueError(
            "crimm could not reconstruct the input PSF/CRD into an OrganizedModel." f"{extra_hint}"
        ) from exc
    return model, topology_generator


def _count_chain_residues(chains: list) -> int:
    return sum(len(list(chain.get_residues())) for chain in chains)


def _is_hydrogen_atom_name(atom_name: str) -> bool:
    normalized = atom_name.strip().upper().lstrip("0123456789")
    return normalized.startswith("H")


def _heavy_atom_scaffold(records: list[dict[str, object]]) -> set[tuple[str, int, str]]:
    scaffold: set[tuple[str, int, str]] = set()
    for record in records:
        atom_name = str(record["atom_name"]).strip().upper()
        if _is_hydrogen_atom_name(atom_name):
            continue
        scaffold.add(
            (
                str(record["seg_id"]).strip().upper(),
                int(record["res_id"]),
                atom_name,
            )
        )
    return scaffold


def _validate_reference_scaffold(
    input_records: list[dict[str, object]],
    reference_records: list[dict[str, object]],
) -> None:
    input_scaffold = _heavy_atom_scaffold(input_records)
    reference_scaffold = _heavy_atom_scaffold(reference_records)
    if input_scaffold == reference_scaffold:
        return

    removed = sorted(input_scaffold - reference_scaffold)
    added = sorted(reference_scaffold - input_scaffold)
    details: list[str] = []
    if removed:
        details.append(
            "removed heavy atoms="
            + ", ".join(f"{seg}:{resid}:{atom}" for seg, resid, atom in removed[:5])
        )
    if added:
        details.append(
            "added heavy atoms="
            + ", ".join(f"{seg}:{resid}:{atom}" for seg, resid, atom in added[:5])
        )
    raise ValueError(
        "Canonical reference patches changed the heavy-atom scaffold required for "
        "solvation back-projection. " + "; ".join(details)
    )


def _model_atom_records(model) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for chain in getattr(model, "non_solvent", list(model)):
        for residue in chain.get_residues():
            seg_id = str(getattr(residue, "segid", "") or getattr(chain, "id", "")).strip()
            res_id = int(residue.id[1])
            for atom in residue:
                records.append(
                    {
                        "seg_id": seg_id,
                        "res_id": res_id,
                        "atom_name": atom.name,
                    }
                )
    return records


def _reference_patch_targets_from_input(
    config: SolvationConfig,
    psf_path: Path,
    crd_path: Path,
) -> list[tuple[str, int, str]]:
    ligand_patches = _normalize_ligand_patch_defs(config.ligand_patches)
    if not _has_explicit_reference_patches(ligand_patches):
        return []

    from cphmd.core.patching import PatchParser, Universe

    toppar_dir = config.toppar_dir or TOPPAR_DIR
    _clear_charmm_atoms()
    try:
        system.read_psf(psf_path)
        system.read_coor(crd_path)

        patch_parser = PatchParser(
            segment_path=toppar_dir / "my_files" / "titratable_residues.str",
            topology_path=toppar_dir / "top_all36_prot.rtf",
        )
        for patch_def in ligand_patches:
            patch_parser.register_ligand(patch_def)

        return _reference_patch_targets(config, patch_parser, Universe())
    finally:
        _clear_charmm_atoms()


def _model_residue_lookup(model) -> dict[tuple[str, int], object]:
    lookup: dict[tuple[str, int], object] = {}
    for chain in getattr(model, "non_solvent", list(model)):
        for residue in chain.get_residues():
            seg_id = str(getattr(residue, "segid", "") or getattr(chain, "id", "")).strip().upper()
            lookup[(seg_id, int(residue.id[1]))] = residue
    return lookup


def _remove_crimm_undefined_atoms(residue) -> None:
    detach_ids = [atom.id for atom in residue.undefined_atoms]
    for atom in residue.undefined_atoms:
        for atom_group in residue.atom_groups:
            if atom in atom_group:
                atom_group.remove(atom)
    for atom_id in detach_ids:
        residue.detach_child(atom_id)
    residue.undefined_atoms = []


def _patched_residue_definition(topology_generator, residue, patch_name):
    from crimm.Modeller.TopoLoader import ResiduePatcher

    res_def = residue.topo_definition
    patched_def_name = res_def.resname + "_" + patch_name

    for res_def_container in topology_generator.res_def_dict.values():
        if residue.resname in res_def_container:
            break
    else:  # pragma: no cover - depends on crimm internal state
        raise ValueError(
            f"No crimm topology set contains residue {residue.resname} for patch {patch_name}."
        )

    if patched_def_name in res_def_container.patched_defs:
        return res_def_container.patched_defs[patched_def_name]

    patcher = ResiduePatcher()
    patch_def = res_def_container[patch_name]
    patched_res_def = patcher.patch_residue_definition(res_def, patch_def, patch_loc="MIDCHAIN")
    actual_total = sum(float(atom_def.charge) for atom_def in patched_res_def.atom_dict.values())
    if res_def.total_charge is not None and patch_def.total_charge is not None:
        expected_total = float(res_def.total_charge) + float(patch_def.total_charge)
    else:
        expected_total = float(round(actual_total))
    delta = expected_total - actual_total
    if abs(delta) > 1e-6:
        candidate_names = [
            name
            for name in patch_def.atom_dict
            if name in patched_res_def.atom_dict and not _is_hydrogen_atom_name(name)
        ]
        if not candidate_names:
            candidate_names = [
                name for name in patch_def.atom_dict if name in patched_res_def.atom_dict
            ]
        if not candidate_names:
            raise ValueError(
                "Canonical reference patch "
                f"{patch_name} cannot be charge-normalized because no patched atoms remain."
            )
        if abs(delta) > 0.05:
            raise ValueError(
                "Canonical reference patch "
                f"{patch_name} changes residue charge by {delta:+.4f} e, which is larger than "
                "the supported normalization tolerance."
            )
        patched_res_def[candidate_names[0]].charge += delta
    patched_res_def.total_charge = expected_total
    res_def_container.patched_defs[patched_def_name] = patched_res_def
    return patched_res_def


def _apply_reference_patch_to_residue(
    topology_generator,
    residue,
    patch_name: str,
) -> None:
    chain_type = str(getattr(residue.parent, "chain_type", ""))
    if chain_type in ("Polypeptide(L)", "Polyribonucleotide", "Polydeoxyribonucleotide"):
        topology_generator.patch_residue(residue, patch_name, QUIET=True)
        return

    patched_res_def = _patched_residue_definition(topology_generator, residue, patch_name)
    present_atom_names = {atom.name for atom in residue}
    missing_atom_names = sorted(set(patched_res_def.atom_dict) - present_atom_names)
    if missing_atom_names:
        raise ValueError(
            "Canonical reference patch "
            f"{patch_name} requires atoms not present in the input template: "
            + ", ".join(missing_atom_names[:8])
        )

    topology_generator._apply_topo_def_for_loading(residue, patched_res_def, QUIET=True)
    _remove_crimm_undefined_atoms(residue)


def _apply_reference_patches_to_model(
    config: SolvationConfig,
    model,
    topology_generator,
    psf_path: Path,
    crd_path: Path,
) -> list[tuple[str, int, str]]:
    from crimm.Modeller.TopoLoader import clear_atom_neighbors

    targets = _reference_patch_targets_from_input(config, psf_path, crd_path)
    if not targets:
        return []

    input_records = _model_atom_records(model)
    residue_lookup = _model_residue_lookup(model)
    touched_chains = set()
    applied_targets: list[tuple[str, int, str]] = []

    for seg_id, resid, patch_name in targets:
        key = (seg_id.strip().upper(), resid)
        residue = residue_lookup.get(key)
        if residue is None:
            raise ValueError(
                "Canonical reference-state target "
                f"{patch_name} for {seg_id}:{resid} was not found in the crimm model."
            )
        try:
            _apply_reference_patch_to_residue(topology_generator, residue, patch_name)
        except Exception as exc:
            raise ValueError(
                "Failed to materialize canonical reference patch "
                f"{patch_name} on {seg_id}:{resid} for internal solvation."
            ) from exc
        touched_chains.add(residue.parent)
        applied_targets.append((seg_id, resid, patch_name))

    for chain in touched_chains:
        clear_atom_neighbors(chain)
        if getattr(chain, "topology", None) is not None:
            chain.topology.update()

    _validate_reference_scaffold(input_records, _model_atom_records(model))

    print("Using canonical reference microstate for internal solvation:")
    for seg_id, resid, patch_name in applied_targets:
        print(f"  {seg_id}:{resid} -> {patch_name}")

    return applied_targets


def _reference_patch_targets(
    config: SolvationConfig,
    patch_parser,
    universe,
) -> list[tuple[str, int, str]]:
    from cphmd.core.patching import (
        _detect_disulfide_bonds,
        _parse_selection_criteria,
        _should_patch_residue,
    )

    criteria = _parse_selection_criteria(config.selected_residues)
    targets: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()

    for resname in patch_parser.residues:
        residues = [
            (row["seg_id"], int(row["res_id"]), row["res_name"])
            for _, row in universe.universe[universe.universe["res_name"] == resname][
                ["seg_id", "res_id", "res_name"]
            ]
            .drop_duplicates()
            .iterrows()
        ]

        if resname == "CYS":
            residues, _ = _detect_disulfide_bonds(universe, residues)

        if resname in patch_parser.site_keys:
            for seg_id, resid, _ in residues:
                for site_key in patch_parser.site_keys[resname]:
                    reference_patch = patch_parser.reference_patches.get(site_key)
                    if reference_patch is None:
                        continue
                    if not _should_patch_residue(seg_id, resid, site_key, criteria):
                        continue
                    key = (seg_id, resid, reference_patch)
                    if key not in seen:
                        targets.append(key)
                        seen.add(key)
            continue

        reference_patch = patch_parser.reference_patches.get(resname)
        if reference_patch is None:
            continue
        for seg_id, resid, residue_name in residues:
            if not _should_patch_residue(seg_id, resid, residue_name, criteria):
                continue
            key = (seg_id, resid, reference_patch)
            if key not in seen:
                targets.append(key)
                seen.add(key)

    return targets


def _solvate_with_crimm(
    config: SolvationConfig,
    psf_path: Path,
    crd_path: Path,
):
    from crimm.Modeller.Solvator import Solvator

    model, topology_generator = _load_crimm_model(config, psf_path, crd_path)
    _apply_reference_patches_to_model(config, model, topology_generator, psf_path, crd_path)
    solvator = Solvator(model)
    box_type = _normalize_crystal_type(config.crystal_type)

    print("Using crimm solvation backend")
    print(f"Building {box_type} solvent box with padding {config.padding:.2f} A")

    solvator.solvate(
        cutoff=config.padding,
        solvcut=2.10,
        remove_existing_water=True,
        remove_existing_ions=True,
        orient_coords=False,
        box_type=box_type,
    )

    water_count = _count_chain_residues(model.solvent)
    print(f"Water molecules after solvation: {water_count}")

    if not config.skip_ions:
        solvator.add_ions(
            concentration=config.salt_concentration,
            method=_normalize_ion_method(config.ion_method),
            cation=config.positive_ion.upper(),
            anion=config.negative_ion.upper(),
            min_dist_solute=_MIN_SOLUTE_ION_DISTANCE,
            min_dist_ion=config.min_ion_distance,
        )
        ion_count = _count_chain_residues(model.ion)
        print(f"Ions after placement: {ion_count}")

    return model


def _export_crimm_model(model, output_dir: Path) -> tuple[Path, Path]:
    from crimm.IO import write_crd, write_psf

    psf_path = output_dir / ".crimm_solvated.psf"
    crd_path = output_dir / ".crimm_solvated.crd"
    write_psf(model, str(psf_path))
    write_crd(model, str(crd_path))
    return psf_path, crd_path


def _write_reference_waterbox_files(
    solvated_psf: Path,
    solvated_crd: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    waterbox_psf = output_dir / ".crimm_waterbox.psf"
    waterbox_crd = output_dir / ".crimm_waterbox.crd"

    _clear_charmm_atoms()
    system.set_bomb_level(-1)
    system.read_psf(solvated_psf)
    system.set_bomb_level(0)
    system.read_coor(solvated_crd)
    system.delete_atoms(AtomSelection(raw=_MOLECULE_EXPR))
    system.write_psf(waterbox_psf)
    system.write_coor(waterbox_crd)
    _clear_charmm_atoms()

    return waterbox_psf, waterbox_crd


def _assemble_compatibility_solvated_system(
    original_psf: Path,
    original_crd: Path,
    waterbox_psf: Path,
    waterbox_crd: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    combined_psf = output_dir / ".compat_solvated.psf"
    combined_crd = output_dir / ".compat_solvated.crd"

    _clear_charmm_atoms()
    system.set_bomb_level(-1)
    system.read_psf(original_psf)
    system.read_coor(original_crd)
    system.read_psf(waterbox_psf, append=True)
    system.set_bomb_level(0)
    system.read_coor(waterbox_crd, append=True)
    system.write_psf(combined_psf)
    system.write_coor(combined_crd)
    _clear_charmm_atoms()

    return combined_psf, combined_crd


def _box_parameters_from_model(config: SolvationConfig, model) -> tuple[list[float], list[float]]:
    solvation_info = getattr(model, "_solvation_info", {})
    box_dim = solvation_info.get("box_dim")
    if box_dim is None:
        raise RuntimeError("crimm did not expose box_dim in model._solvation_info")

    angles = solvation_info.get("angles") or _default_box_angles(config.crystal_type)
    dimensions = [float(box_dim), float(box_dim), float(box_dim)]
    return dimensions, [float(angle) for angle in angles]


def _write_box_file(
    output_dir: Path,
    crystal_type: str,
    dimensions: list[float],
    angles: list[float],
) -> None:
    with open(output_dir / "box.dat", "w", encoding="utf-8") as handle:
        handle.write(f"{crystal_type.upper()}\n")
        handle.write(" ".join(str(value) for value in dimensions) + "\n")
        handle.write(" ".join(str(value) for value in angles) + "\n")


def _define_charmm_crystal(
    crystal_type: str,
    dimensions: list[float],
    angles: list[float],
) -> None:
    normalized = crystal_type.strip().upper()
    supported_types = {
        "CUBIC",
        "OCTAHEDRAL",
        "RHDO",
        "ORTHORHOMBIC",
        "ORTH",
        "TETRAGONAL",
        "TETR",
        "MONOCLINIC",
        "MONO",
        "TRICLINIC",
        "TRIC",
        "HEXAGONAL",
        "HEXA",
        "RHOMBOHEDRAL",
        "RHOM",
    }
    if normalized not in supported_types:
        raise ValueError(f"Unsupported crystal type for CHARMM output setup: {crystal_type!r}")

    a, b, c = dimensions
    alpha, beta, gamma = angles

    system.crystal_free()
    system.crystal_define(
        shape=normalized,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )


def _load_solvated_system_for_charmm_output(
    config: SolvationConfig,
    psf_path: Path,
    crd_path: Path,
    dimensions: list[float],
    angles: list[float],
) -> None:
    _clear_charmm_atoms()
    system.set_bomb_level(-1)
    system.read_psf(psf_path)
    system.set_bomb_level(0)
    system.read_coor(crd_path)

    _define_charmm_crystal(config.crystal_type, dimensions, angles)
    system.crystal_build(cutoff=14.0)
    system.image_setup(byres=True, segid_list=["SOLV", "IONS"])
    system.image_setup(byres=False, selection=AtomSelection(raw=_MOLECULE_EXPR))
    system.nbonds_setup(
        ctonnb=10,
        ctofnb=12,
        cutnb=14,
        cutim=14,
        wmin=1.0,
        switch=False,
        fswitch=True,
        vswitch=True,
    )
    system.define_molecule("MOL", AtomSelection(raw=_MOLECULE_EXPR))


def _minimize_loaded_system() -> None:
    system.energy_show()
    initial_energy = system.energy_get_total()

    for force in (100, 50, 25, 5):
        system.cons_harm_force(force, AtomSelection(raw="MOL .and. (.not. hydrogen)"))
        system.minimize_sd(nsteps=50)
        system.minimize_abnr(nsteps=100)
        system.cons_harm_clear()

    system.minimize_sd(nsteps=1000)
    final_energy = system.energy_get_total()

    print(f"Energy before minimization: {initial_energy} kcal/mol")
    print(f"Energy after minimization: {final_energy} kcal/mol")
    print(f"Energy change after minimization: {initial_energy - final_energy} kcal/mol")


def _maybe_minimize_loaded_system(config: SolvationConfig) -> None:
    if config.minimize:
        _minimize_loaded_system()
    else:
        print("Skipping post-solvation minimization")


def _write_final_outputs(
    output_dir: Path,
    config: SolvationConfig,
    dimensions: list[float],
) -> None:
    box_label = ":".join(f"{value:.3f}" for value in dimensions)
    molecule_title = (
        "Molecule with Minimization (part with waterbox.*)"
        if config.minimize
        else "Molecule after Solvation (part with waterbox.*)"
    )

    system.write_coor(output_dir / "solvated.crd")
    system.write_psf(output_dir / "solvated.psf")
    system.write_coor_pdb(output_dir / "solvated.pdb")

    system.write_coor(
        output_dir / "molecule.crd",
        title=molecule_title,
        selection=AtomSelection(raw=_MOLECULE_SELECTION),
    )
    system.write_psf(
        output_dir / "molecule.psf",
        title=molecule_title,
        selection=AtomSelection(raw=_MOLECULE_SELECTION),
    )

    system.set_bomb_level(-1)
    system.write_coor(
        output_dir / "waterbox.crd",
        title=f"{config.crystal_type.upper()} Waterbox with box size {box_label}",
        selection=AtomSelection(raw=_WATERBOX_SELECTION),
    )
    system.write_psf(
        output_dir / "waterbox.psf",
        title=f"{config.crystal_type.upper()} Waterbox with box size {box_label}",
        selection=AtomSelection(raw=_WATERBOX_SELECTION),
    )
    system.set_bomb_level(0)


def solvate_system(config: SolvationConfig) -> Path:
    """Solvate a molecular system in a water box using crimm."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_base = _input_base_path(config.input_file)
    psf_path = input_base.with_suffix(".psf")
    if not psf_path.exists():
        raise FileNotFoundError(f"Input PSF not found: {psf_path}")

    _clear_charmm_atoms()
    _read_topology_files(config)

    if _check_gpu():
        system.blade_on()

    crd_path = _ensure_input_crd(input_base)
    model = _solvate_with_crimm(config, psf_path, crd_path)
    dimensions, angles = _box_parameters_from_model(config, model)
    _write_box_file(output_dir, config.crystal_type, dimensions, angles)

    exported_psf, exported_crd = _export_crimm_model(model, output_dir)
    waterbox_psf = waterbox_crd = combined_psf = combined_crd = None
    try:
        waterbox_psf, waterbox_crd = _write_reference_waterbox_files(
            exported_psf,
            exported_crd,
            output_dir,
        )
        combined_psf, combined_crd = _assemble_compatibility_solvated_system(
            psf_path,
            crd_path,
            waterbox_psf,
            waterbox_crd,
            output_dir,
        )
        _load_solvated_system_for_charmm_output(
            config,
            combined_psf,
            combined_crd,
            dimensions,
            angles,
        )
        _maybe_minimize_loaded_system(config)
        _write_final_outputs(output_dir, config, dimensions)
    finally:
        exported_psf.unlink(missing_ok=True)
        exported_crd.unlink(missing_ok=True)
        if waterbox_psf is not None:
            waterbox_psf.unlink(missing_ok=True)
        if waterbox_crd is not None:
            waterbox_crd.unlink(missing_ok=True)
        if combined_psf is not None:
            combined_psf.unlink(missing_ok=True)
        if combined_crd is not None:
            combined_crd.unlink(missing_ok=True)

    print("Solvation completed")
    return output_dir
