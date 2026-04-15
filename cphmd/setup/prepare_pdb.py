"""Prepare PSF/CRD/PDB inputs from a local PDB file or RCSB entry."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PrepareSourceType = Literal["auto", "file", "rcsb"]

_VALID_SOURCE_TYPES = {"auto", "file", "rcsb"}
_RCSB_ID_PATTERN = re.compile(r"^[A-Za-z0-9]{4}$")


@dataclass
class PreparePDBConfig:
    """Configuration for preparing a solvatable input system from a PDB source."""

    input_source: str | Path
    output_dir: str | Path = "pdb"
    output_name: str = "input"
    source_type: PrepareSourceType = "auto"
    include_solvent: bool = False
    include_hydrogens: bool = False
    use_bio_assembly: bool = True
    rename_charmm_ions: bool = True
    rename_solvent_oxygen: bool = True
    drop_ligands: bool = False
    coerce: bool = False
    prot_first_patch: str = "ACE"
    prot_last_patch: str = "CT3"
    na_first_patch: str = "5TER"
    na_last_patch: str = "3PHO"
    auto_correct_first_patch: bool = True
    build_coords: bool = True
    preserve_ic: bool = True
    solvent_model: str = "TIP3"
    cgenff_executable_path: str | Path | None = None
    cgenff_output_path: str | Path | None = None
    quiet: bool = True


def _normalize_source_type(source_type: str) -> PrepareSourceType:
    normalized = source_type.lower()
    if normalized not in _VALID_SOURCE_TYPES:
        supported = ", ".join(sorted(_VALID_SOURCE_TYPES))
        raise ValueError(f"Unsupported source_type {source_type!r}. Supported values: {supported}.")
    return normalized  # type: ignore[return-value]


def _looks_like_rcsb_id(value: str) -> bool:
    return bool(_RCSB_ID_PATTERN.fullmatch(value.strip()))


def _resolve_source_type(input_source: str | Path, source_type: str) -> PrepareSourceType:
    normalized = _normalize_source_type(source_type)
    if normalized != "auto":
        return normalized

    path = Path(input_source)
    if path.exists() or path.suffix:
        return "file"
    if _looks_like_rcsb_id(str(input_source)):
        return "rcsb"
    return "file"


def _output_base_path(output_dir: str | Path, output_name: str) -> Path:
    return Path(output_dir) / Path(output_name).with_suffix("").name


def _first_model(structure):
    try:
        return structure.models[0]
    except Exception:
        return next(iter(structure))


def _load_local_pdb_model(config: PreparePDBConfig):
    from crimm.IO import PDBParser
    from crimm.StructEntities.OrganizedModel import OrganizedModel

    pdb_path = Path(config.input_source)
    if not pdb_path.exists():
        raise FileNotFoundError(f"Input PDB not found: {pdb_path}")

    parser = PDBParser(
        first_model_only=True,
        include_solvent=config.include_solvent,
        QUIET=config.quiet,
    )
    structure = parser.get_structure(str(pdb_path), structure_id=pdb_path.stem)
    return OrganizedModel(
        _first_model(structure),
        rename_charmm_ions=config.rename_charmm_ions,
        rename_solvent_oxygen=config.rename_solvent_oxygen,
    )


def _load_rcsb_model(config: PreparePDBConfig):
    from crimm.Fetchers import fetch_rcsb

    model = fetch_rcsb(
        str(config.input_source),
        first_model_only=True,
        use_bio_assembly=config.use_bio_assembly,
        include_solvent=config.include_solvent,
        include_hydrogens=config.include_hydrogens,
        organize=True,
        rename_charmm_ions=config.rename_charmm_ions,
        rename_solvent_oxygen=config.rename_solvent_oxygen,
    )
    if model is None:
        raise ValueError(f"Failed to fetch RCSB entry {config.input_source!r}.")
    return model


def _count_ligand_like_chains(model) -> int:
    seen: set[int] = set()
    total = 0
    for attr in ("ligand", "co_solvent", "phos_ligand"):
        for chain in getattr(model, attr, []) or []:
            identity = id(chain)
            if identity in seen:
                continue
            seen.add(identity)
            if _chain_is_attached(model, chain):
                total += 1
    return total


def _chain_is_attached(model, chain) -> bool:
    child_list = getattr(model, "child_list", None)
    child_dict = getattr(model, "child_dict", None)
    if child_list is None and child_dict is None:
        return True

    if child_list is not None and any(child is chain for child in child_list):
        return True

    chain_id = getattr(chain, "id", None)
    if child_dict is not None and chain_id in child_dict:
        return child_dict[chain_id] is chain or child_list is None

    return False


def _drop_ligand_like_chains(model) -> int:
    chains_by_attr = {}
    chains_to_drop = {}
    for attr in ("ligand", "co_solvent", "phos_ligand"):
        chains = list(getattr(model, attr, []) or [])
        chains_by_attr[attr] = chains
        for chain in chains:
            chains_to_drop.setdefault(id(chain), chain)

    for chain in chains_to_drop.values():
        try:
            model.detach_child(chain.id)
        except Exception:
            model.child_list = [child for child in model.child_list if child is not chain]
            model.child_dict = {child.id: child for child in model.child_list}

    dropped = set(chains_to_drop)
    for attr, chains in chains_by_attr.items():
        try:
            setattr(model, attr, [chain for chain in chains if id(chain) not in dropped])
        except Exception:
            pass
    return len(chains_to_drop)


def _validate_supported_input(model, config: PreparePDBConfig) -> None:
    ligand_chain_count = _count_ligand_like_chains(model)
    if ligand_chain_count > 0 and config.cgenff_executable_path is None:
        raise ValueError(
            "Ligand-like chains were detected in the input structure, but no "
            "cgenff_executable_path was provided. Prepare standard protein/nucleic "
            "systems directly from PDB, or supply a CGenFF executable for ligand topology."
        )


def _generate_topology(model, config: PreparePDBConfig):
    from crimm.Modeller.TopoLoader import TopologyGenerator

    topology_generator = TopologyGenerator(
        cgenff_excutable_path=(
            str(config.cgenff_executable_path) if config.cgenff_executable_path else None
        ),
        cgenff_output_path=str(config.cgenff_output_path) if config.cgenff_output_path else None,
    )
    topology_generator.generate_model(
        model,
        coerce=config.coerce,
        prot_first_patch=config.prot_first_patch,
        prot_last_patch=config.prot_last_patch,
        na_first_patch=config.na_first_patch,
        na_last_patch=config.na_last_patch,
        auto_correct_first_patch=config.auto_correct_first_patch,
        build_coords=config.build_coords,
        preserve_ic=config.preserve_ic,
        solvent_model=config.solvent_model,
        QUIET=config.quiet,
    )
    return topology_generator


def _validate_generated_topology(model) -> None:
    for attr in ("ligand", "co_solvent", "phos_ligand"):
        for chain in getattr(model, attr, []) or []:
            if getattr(chain, "topology", None) is None:
                raise ValueError(
                    "CRIMM did not generate topology for all ligand-like chains. "
                    "Check the CGenFF executable configuration or prepare the ligand externally."
                )


def _write_pdb_from_crd(crd_path: Path, pdb_path: Path) -> None:
    """Write a simple visualization PDB from a CHARMM CARD/EXT CRD file."""

    lines: list[str] = []
    atom_records_started = False
    for line in crd_path.read_text(encoding="utf-8").splitlines():
        if not atom_records_started:
            if line.strip() and not line.startswith("*"):
                atom_records_started = True
            continue
        if len(line) < 100:
            continue

        atom_number = int(line[0:10])
        residue_number = int(line[10:20])
        residue_name = line[22:30].strip()
        atom_name = line[32:40].strip()
        x = float(line[40:60])
        y = float(line[60:80])
        z = float(line[80:100])
        segid = line[102:110].strip()
        chain_id = segid[:1] or " "

        lines.append(
            f"ATOM  {atom_number:5d} {atom_name:<4s} {residue_name:>3s} {chain_id}"
            f"{residue_number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00      {segid:<4s}"
        )
    lines.append("END")
    pdb_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_model(model, output_base: Path) -> Path:
    from crimm.IO import write_crd, write_psf

    output_base.parent.mkdir(parents=True, exist_ok=True)

    psf_path = output_base.with_suffix(".psf")
    crd_path = output_base.with_suffix(".crd")
    pdb_path = output_base.with_suffix(".pdb")

    write_psf(model, str(psf_path))
    write_crd(model, str(crd_path))
    _write_pdb_from_crd(crd_path, pdb_path)
    return output_base


def prepare_pdb_system(config: PreparePDBConfig) -> Path:
    """Prepare PSF/CRD/PDB inputs from a local PDB file or RCSB entry."""

    source_type = _resolve_source_type(config.input_source, config.source_type)
    if source_type == "file":
        model = _load_local_pdb_model(config)
    else:
        model = _load_rcsb_model(config)

    if config.drop_ligands:
        _drop_ligand_like_chains(model)
    _validate_supported_input(model, config)
    _generate_topology(model, config)
    _validate_generated_topology(model)
    output_base = _output_base_path(config.output_dir, config.output_name)
    return _export_model(model, output_base)
