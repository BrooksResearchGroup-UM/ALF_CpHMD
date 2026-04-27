"""Native pyCHARMM system boundary wrappers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Sequence

from cphmd import TOPPAR_DIR
from cphmd.native.errors import SystemLoadError, wrap_exception
from cphmd.native.types import (
    AtomRecord,
    AtomSelection,
    CellParameters,
    ResidueRecord,
    TopologySnapshot,
)
from cphmd.utils.charmm_path import qpath

_CELL_CACHE: CellParameters | None = None


def _pycharmm_read():
    import pycharmm.read as read

    return read


def _pycharmm_write():
    import pycharmm.write as write

    return write


def _pycharmm_select_module():
    import pycharmm.select as select

    return select


def _pycharmm_lingo():
    import pycharmm.lingo as lingo

    return lingo


def _pycharmm_psf():
    import pycharmm.psf as psf

    return psf


def _pycharmm_select_atoms():
    import pycharmm

    return pycharmm.SelectAtoms


def _pycharmm_select_atoms_module():
    import pycharmm.select_atoms as select_atoms

    return select_atoms


def _pycharmm_coor():
    import pycharmm.coor as coor

    return coor


def _pycharmm_generate():
    import pycharmm.generate as generate

    return generate


def _pycharmm_restraints():
    import pycharmm.restraints as restraints

    return restraints


def _pycharmm_blade():
    import pycharmm.blade as blade

    return blade


def _pycharmm_block():
    import pycharmm.block as block

    return block


def _pycharmm_ic():
    import pycharmm.ic as ic

    return ic


def _pycharmm_crystal():
    import pycharmm.crystal as crystal

    return crystal


def _pycharmm_image():
    import pycharmm.image as image

    return image


def _pycharmm_energy():
    import pycharmm.energy as energy

    return energy


def _pycharmm_minimize():
    import pycharmm.minimize as minimize

    return minimize


def _pycharmm_shake():
    import pycharmm.shake as shake

    return shake


def _pycharmm_settings():
    import pycharmm.settings as settings

    return settings


def _pycharmm_miscom():
    import pycharmm.miscom as miscom

    return miscom


def _pycharmm_root():
    import pycharmm

    return pycharmm


def _invalidate_cache() -> None:
    global _CELL_CACHE
    _CELL_CACHE = None


def _selection_expr(selection: AtomSelection | None) -> str | None:
    if selection is None:
        return None
    if selection.raw:
        return selection.raw.strip()

    clauses: list[str] = []
    if selection.segid is not None:
        clauses.append(f"segid {selection.segid}")
    if selection.resid is not None:
        clauses.append(f"resid {selection.resid}")
    if selection.resname is not None:
        clauses.append(f"resname {selection.resname}")
    if selection.atom_name is not None:
        clauses.append(f"type {selection.atom_name}")
    if selection.hydrogens:
        clauses.append("hydrogen")
    return " .and. ".join(clauses) if clauses else "all"


def _script_selection(selection: AtomSelection) -> str:
    expr = _selection_expr(selection)
    if expr is None:
        return "sele all end"
    lowered = expr.lower()
    if lowered.startswith(("sele ", "select ")) and lowered.endswith(" end"):
        return expr
    if lowered.endswith(" end"):
        expr = expr[:-4].rstrip()
    return f"sele {expr} end"


def _with_select(kwargs: dict[str, Any], selection: AtomSelection | None) -> dict[str, Any]:
    if selection is not None:
        kwargs["selection"] = _write_selection(selection)
    return kwargs


def _write_selection(selection: AtomSelection):
    if selection.raw:
        expr = _raw_write_selection_expr(selection)
        raw_object = _raw_write_selection_object(expr)
        if raw_object is not None:
            return raw_object
        parser = getattr(_pycharmm_select_atoms_module(), "parse_selection", None)
        if parser is not None:
            return parser(expr)
        if expr.lower() == "all":
            return _pycharmm_select_atoms()(select_all=True)
        raise SystemLoadError("raw write selections require pycharmm.select_atoms.parse_selection")
    return _select_atoms(selection)


def _select_atoms(selection: AtomSelection):
    if selection.raw:
        return None
    kwargs: dict[str, Any] = {"update": True}
    if selection.segid is not None:
        kwargs["seg_id"] = selection.segid
    if selection.resid is not None:
        kwargs["res_id"] = str(selection.resid)
    if selection.resname is not None:
        kwargs["res_name"] = selection.resname
    if selection.atom_name is not None:
        kwargs["atom_type"] = selection.atom_name
    if selection.hydrogens:
        kwargs["hydrogens"] = True
    if len(kwargs) == 1:
        kwargs["select_all"] = True
    return _pycharmm_select_atoms()(**kwargs)


def _raw_write_selection_expr(selection: AtomSelection) -> str:
    expr = _selection_expr(selection)
    if expr is None:
        return "all"
    lowered = expr.lower()
    if lowered.startswith("sele "):
        expr = expr[5:].strip()
    elif lowered.startswith("select "):
        expr = expr[7:].strip()
    if expr.lower().endswith(" end"):
        expr = expr[:-4].rstrip()
    expr = re.sub(r"\.(and|or|not)\.", r"\1", expr, flags=re.IGNORECASE)
    return expr


def _raw_write_selection_object(expr: str):
    normalized = " ".join(expr.split())
    negate = False
    inner = normalized
    match = re.fullmatch(r"not\s*\((.*)\)", normalized, flags=re.IGNORECASE)
    if match is not None:
        negate = True
        inner = match.group(1).strip()
    if not re.fullmatch(r"segid\s+\S+(?:\s+or\s+segid\s+\S+)*", inner, flags=re.IGNORECASE):
        return None

    segids = re.findall(r"\bsegid\s+(\S+)", inner, flags=re.IGNORECASE)
    if not segids:
        return None

    select_atoms = _pycharmm_select_atoms()
    selected = select_atoms(seg_id=segids[0], update=True)
    for segid in segids[1:]:
        selected = selected | select_atoms(seg_id=segid, update=True)
    return ~selected if negate else selected


def _safe_int(value: Any) -> int | str:
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def read_rtf(path: Path | str, *, append: bool = False) -> None:
    try:
        _pycharmm_read().rtf(qpath(path), append=append)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"reading RTF {path}") from exc


def read_param(path: Path | str, *, append: bool = False, flex: bool = True) -> None:
    try:
        _pycharmm_read().prm(qpath(path), append=append, flex=flex)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"reading parameter file {path}") from exc


def read_psf(path: Path | str, *, card: bool = True, append: bool = False) -> None:
    if not card:
        raise SystemLoadError("read_psf only supports CHARMM card PSF files")
    try:
        kwargs = {"append": append} if append else {}
        _pycharmm_read().psf_card(qpath(path), **kwargs)
        _invalidate_cache()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"reading PSF {path}") from exc


def read_coor(path: Path | str, *, card: bool = True, append: bool = False) -> None:
    if not card:
        raise SystemLoadError("read_coor only supports CHARMM card coordinate files")
    try:
        kwargs = {"append": append} if append else {}
        _pycharmm_read().coor_card(qpath(path), **kwargs)
        _invalidate_cache()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"reading coordinates {path}") from exc


def read_pdb(path: Path | str, *, resid: bool = False) -> None:
    try:
        kwargs = {"resid": resid} if resid else {}
        _pycharmm_read().pdb(qpath(path), **kwargs)
        _invalidate_cache()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"reading PDB coordinates {path}") from exc


def read_sequence_string(sequence: str) -> None:
    try:
        _pycharmm_read().sequence_string(sequence)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "reading sequence string") from exc


def write_psf(
    path: Path | str,
    *,
    card: bool = True,
    selection: AtomSelection | None = None,
    title: str | None = None,
) -> None:
    if not card:
        raise SystemLoadError("write_psf only supports CHARMM card PSF files")
    kwargs = _with_select({}, selection)
    try:
        _pycharmm_write().psf_card(qpath(path), title=title or "", **kwargs)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"writing PSF {path}") from exc


def write_coor(
    path: Path | str,
    *,
    card: bool = True,
    selection: AtomSelection | None = None,
    title: str | None = None,
) -> None:
    if not card:
        raise SystemLoadError("write_coor only supports CHARMM card coordinate files")
    kwargs = _with_select({}, selection)
    try:
        _pycharmm_write().coor_card(qpath(path), title=title or "", **kwargs)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"writing coordinates {path}") from exc


def write_coor_pdb(
    path: Path | str,
    *,
    selection: AtomSelection | None = None,
    title: str | None = None,
) -> None:
    kwargs = _with_select({}, selection)
    try:
        _pycharmm_write().coor_pdb(qpath(path), title=title or "", **kwargs)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"writing PDB coordinates {path}") from exc


def get_natom() -> int:
    try:
        return int(_pycharmm_psf().get_natom())
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "querying atom count") from exc


def _row_value(positions: Any, index: int, key: str) -> float:
    if hasattr(positions, "iloc"):
        return float(positions.iloc[index][key])
    return float(positions[index][key])


def _residue_index_for_atom(ibase: Sequence[int], atom_index: int) -> int:
    for res_index in range(max(len(ibase) - 1, 0)):
        start = int(ibase[res_index])
        end = int(ibase[res_index + 1])
        if start <= atom_index < end:
            return res_index
    return min(atom_index, max(len(ibase) - 2, 0))


def _segid_for_residue(segids: Sequence[str], nictot: Sequence[int], residue_index: int) -> str:
    if not segids:
        return ""
    if len(segids) == 1:
        return str(segids[0]).strip()
    if nictot:
        if len(nictot) == len(segids) + 1 and int(nictot[0]) == 0:
            for seg_index, (start, end) in enumerate(zip(nictot, nictot[1:])):
                if int(start) <= residue_index < int(end):
                    return str(segids[seg_index]).strip()
        total = 0
        for seg_index, count in enumerate(nictot[: len(segids)]):
            total += int(count)
            if residue_index < total:
                return str(segids[seg_index]).strip()
    return str(segids[min(residue_index, len(segids) - 1)]).strip()


def get_topology_snapshot(*, include_cell: bool = False) -> TopologySnapshot:
    try:
        psf = _pycharmm_psf()
        coor = _pycharmm_coor()
        natom = int(psf.get_natom())
        atom_names = list(psf.get_atype())
        residue_names = list(psf.get_res())
        residue_ids = list(psf.get_resid())
        segids = list(psf.get_segid())
        masses = list(psf.get_amass())
        charges = list(psf.get_charges())
        ibase = list(psf.get_ibase())
        nictot = list(psf.get_nictot()) if hasattr(psf, "get_nictot") else []
        positions = coor.get_positions()

        atoms: list[AtomRecord] = []
        for atom_index in range(natom):
            residue_index = _residue_index_for_atom(ibase, atom_index)
            atoms.append(
                AtomRecord(
                    segid=_segid_for_residue(segids, nictot, residue_index),
                    resid=_safe_int(residue_ids[residue_index]),
                    resname=str(residue_names[residue_index]).strip(),
                    atom_name=str(atom_names[atom_index]).strip(),
                    x=_row_value(positions, atom_index, "x"),
                    y=_row_value(positions, atom_index, "y"),
                    z=_row_value(positions, atom_index, "z"),
                    mass=float(masses[atom_index]),
                    charge=float(charges[atom_index]),
                )
            )
        residues: list[ResidueRecord] = []
        for residue_index in range(len(residue_names)):
            residues.append(
                ResidueRecord(
                    segid=_segid_for_residue(segids, nictot, residue_index),
                    resid=_safe_int(residue_ids[residue_index]),
                    resname=str(residue_names[residue_index]).strip(),
                )
            )
        return TopologySnapshot(
            atoms=tuple(atoms),
            natom=natom,
            cell=get_cell_parameters() if include_cell else None,
            residues=tuple(residues),
        )
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "querying topology snapshot") from exc


def get_cell_parameters() -> CellParameters:
    """Return current unit-cell lengths and angles.

    pyCHARMM exposes the scalar cell values through image APIs but does not
    provide an authoritative crystal-shape getter here, so ``shape`` is
    reported as ``None`` instead of fabricating a discriminated type.
    """

    global _CELL_CACHE
    if _CELL_CACHE is not None:
        return _CELL_CACHE
    try:
        a, b, c, alpha, beta, gamma = _pycharmm_image().get_ucell()
        _CELL_CACHE = CellParameters(
            a=float(a),
            b=float(b),
            c=float(c),
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
            shape=None,
        )
        return _CELL_CACHE
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "querying cell parameters") from exc


def coor_stat() -> dict[str, float]:
    try:
        return dict(_pycharmm_coor().stat())
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "querying coordinate statistics") from exc


def patch(patch_name: str, segid: str, resid: int | str, *, setup: bool = True) -> None:
    try:
        _pycharmm_generate().patch(patch_name, f"{segid} {resid}", setup=setup)
        _invalidate_cache()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"applying patch {patch_name}") from exc


def _select_residue_atom_names(segid: str, resid: int | str, atom_names: Sequence[str]):
    return _pycharmm_select_atoms()(
        seg_id=segid,
        res_id=str(resid),
        atom_type=list(atom_names),
        update=True,
    )


def replicate_atoms(
    segid: str,
    resid: int | str,
    atom_names: Sequence[str],
    *,
    replica_segid: str,
    nreplica: int,
    setup: bool = True,
) -> None:
    try:
        selection = _select_residue_atom_names(segid, resid, atom_names)
        _pycharmm_generate().replica(
            selection=selection,
            segid=replica_segid,
            nreplica=nreplica,
            setup=setup,
        )
        _invalidate_cache()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"replicating residue {segid}:{resid}") from exc


def delete_atoms_by_names(segid: str, resid: int | str, atom_names: Sequence[str]) -> None:
    try:
        selection = _select_residue_atom_names(segid, resid, atom_names)
        ok = _pycharmm_psf().delete_atoms(selection)
        if ok is False:
            raise SystemLoadError("pyCHARMM psf.delete_atoms returned failure")
        _invalidate_cache()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"deleting atoms from {segid}:{resid}") from exc


def join_segments(segid_1: str, segid_2: str, *, renumber: bool = False) -> None:
    try:
        _pycharmm_generate().join(segid_1, segid_2, renumber=renumber)
        _invalidate_cache()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"joining segments {segid_1} {segid_2}") from exc


def generate_segment(
    segid: str,
    *,
    first: str = "NTER",
    last: str = "CTER",
    setup: bool = True,
    angle: bool = True,
    dihe: bool = True,
) -> None:
    try:
        _pycharmm_generate().new_segment(
            seg_name=segid,
            first_patch=first,
            last_patch=last,
            setup_ic=setup,
            angle=angle,
            dihedral=dihe,
        )
        _invalidate_cache()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"generating segment {segid}") from exc


def ic_prm_fill(*, comp: bool = False) -> None:
    try:
        _pycharmm_ic().prm_fill(comp)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "filling internal coordinates") from exc


def ic_seed(atoms: Sequence[tuple[int, str]]) -> None:
    if len(atoms) != 3:
        raise ValueError("ic_seed requires exactly three (resid, atom_name) entries")
    try:
        (res1, atom1), (res2, atom2), (res3, atom3) = atoms
        _pycharmm_ic().seed(res1, atom1, res2, atom2, res3, atom3)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "seeding internal coordinates") from exc


def ic_build(*, seed_atoms: Sequence[tuple[int, str]] | None = None) -> None:
    try:
        if seed_atoms is not None:
            ic_seed(seed_atoms)
        _pycharmm_ic().build()
        _invalidate_cache()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "building internal coordinates") from exc


def coor_orient() -> None:
    try:
        _pycharmm_coor().orient()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "orienting coordinates") from exc


def hbuild(selection: AtomSelection | None = None) -> None:
    try:
        kwargs: dict[str, Any] = {}
        if selection is not None:
            select_atoms = _select_atoms(selection)
            if select_atoms is None:
                _pycharmm_lingo().charmm_script(f"hbuild {_script_selection(selection)}")
                _invalidate_cache()
                return
            kwargs["selection"] = select_atoms
        _pycharmm_coor().hbuild(**kwargs)
        _invalidate_cache()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "building hydrogens") from exc


def coor_volume(
    *,
    selection: AtomSelection | None = None,
    space: int | None = None,
) -> dict[str, Any]:
    try:
        kwargs: dict[str, Any] = {}
        if selection is not None:
            select_atoms = _select_atoms(selection)
            if select_atoms is None:
                parts = ["coor volume"]
                if space is not None:
                    parts.append(f"space {space}")
                parts.append(_script_selection(selection))
                _pycharmm_lingo().charmm_script(" ".join(parts))
                builtins = _pycharmm_lingo().get_charmm_builtins()
                return {
                    key: builtins[key]
                    for key in ("NVAC", "NSEL", "VOLUME", "VOLU", "FREEVOL")
                    if key in builtins
                }
            kwargs["selection"] = select_atoms
        if space is not None:
            kwargs["space"] = space
        return dict(_pycharmm_coor().volume(**kwargs))
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "computing coordinate volume") from exc


def rename_atoms(selection: AtomSelection, new_name: str) -> None:
    try:
        select_atoms = _select_atoms(selection)
        if select_atoms is None:
            _pycharmm_lingo().charmm_script(
                f"rename atom {new_name} {_script_selection(selection)}"
            )
        else:
            _pycharmm_generate().rename("ATOM", new_name, selection=select_atoms)
        _invalidate_cache()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"renaming atoms to {new_name}") from exc


def rename_residues(selection: AtomSelection, new_resname: str) -> None:
    try:
        select_atoms = _select_atoms(selection)
        if select_atoms is None:
            _pycharmm_lingo().charmm_script(
                f"rename resn {new_resname} {_script_selection(selection)}"
            )
        else:
            _pycharmm_generate().rename("RESN", new_resname, selection=select_atoms)
        _invalidate_cache()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"renaming residues to {new_resname}") from exc


def delete_atoms(selection: AtomSelection) -> None:
    try:
        select_atoms = _select_atoms(selection)
        if select_atoms is None:
            _pycharmm_lingo().charmm_script(f"delete atom {_script_selection(selection)}")
        else:
            ok = _pycharmm_psf().delete_atoms(select_atoms)
            if ok is False:
                raise SystemLoadError("pyCHARMM psf.delete_atoms returned failure")
        _invalidate_cache()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "deleting atoms") from exc


def delete_connectivity(
    first_selection: AtomSelection,
    second_selection: AtomSelection,
    *,
    psort: bool = False,
) -> None:
    try:
        first_atoms = _select_atoms(first_selection)
        if first_atoms is None:
            raise SystemLoadError(
                "delete_connectivity requires structured AtomSelection objects"
            )
        second_atoms = _select_atoms(second_selection)
        if second_atoms is None:
            raise SystemLoadError(
                "delete_connectivity requires structured AtomSelection objects"
            )
        ok = _pycharmm_psf().delete_connectivity(first_atoms, second_atoms, psort=psort)
        if ok is False:
            raise SystemLoadError("pyCHARMM psf.delete_connectivity returned failure")
        _invalidate_cache()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "deleting atom connectivity") from exc


def psf_hmr(*, factor: float = 3.0, resnames_exclude: list[str] | None = None) -> None:
    if factor != 3.0:
        raise ValueError("pyCHARMM psf.hmr supports only factor=3.0")
    try:
        if resnames_exclude is None:
            _pycharmm_psf().hmr()
        else:
            _pycharmm_psf().hmr(resnames_exclude=resnames_exclude)
        _invalidate_cache()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "applying hydrogen mass repartitioning") from exc


def crystal_free() -> None:
    try:
        _pycharmm_crystal().free()
        _invalidate_cache()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "freeing crystal state") from exc


def crystal_define(
    *,
    shape: str,
    a: float,
    b: float,
    c: float,
    alpha: float = 90.0,
    beta: float = 90.0,
    gamma: float = 90.0,
) -> None:
    crystal = _pycharmm_crystal()
    normalized = shape.lower()
    try:
        if normalized in {"cubic", "cubi"}:
            crystal.define_cubic(a)
        elif normalized in {"octa", "octahedral"}:
            crystal.define_octa(a)
        elif normalized == "rhdo":
            crystal.define_rhdo(a)
        elif normalized in {"ortho", "orthorhombic", "orth"}:
            crystal.define_ortho(a, b, c)
        elif normalized in {"tetra", "tetragonal", "tetr"}:
            crystal.define_tetra(a, c)
        elif normalized in {"mono", "monoclinic"}:
            crystal.define_mono(a, b, c, beta)
        elif normalized in {"tric", "triclinic"}:
            crystal.define_tri(a, b, c, alpha, beta, gamma)
        elif normalized in {"hexa", "hexagonal"}:
            crystal.define_hexa(a, c)
        elif normalized in {"rhom", "rhombohedral"}:
            crystal.define_rhombo(a, alpha)
        else:
            raise ValueError(f"unsupported crystal shape {shape!r}")
        _invalidate_cache()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"defining crystal {shape}") from exc


def crystal_build(*, cutoff: float) -> None:
    try:
        _pycharmm_crystal().build(cutoff)
        _invalidate_cache()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "building crystal images") from exc


def image_setup(
    *,
    byres: bool = True,
    segid_list: list[str] | None = None,
    selection: AtomSelection | None = None,
    xcen: float = 0.0,
    ycen: float = 0.0,
    zcen: float = 0.0,
) -> None:
    if selection is not None:
        selection_text = _selection_expr(selection) or "all"
    elif segid_list:
        selection_text = " .or. ".join(f"segid {segid}" for segid in segid_list)
    else:
        selection_text = "all"
    mode = "byres" if byres else "byseg"
    script = f"image {mode} xcen {xcen} ycen {ycen} zcen {zcen} " f"sele {selection_text} end"
    try:
        _pycharmm_lingo().charmm_script(script)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "setting up image centering") from exc


def define_molecule(name: str, selection: AtomSelection) -> None:
    try:
        select_atoms = _select_atoms(selection)
        if select_atoms is None:
            _pycharmm_lingo().charmm_script(f"define {name} {_script_selection(selection)}")
        else:
            _pycharmm_select_module().store_selection(name, select_atoms.get_selection())
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"defining molecule selection {name}") from exc


def cons_harm_force(force: float, selection: AtomSelection) -> None:
    try:
        select_atoms = _select_atoms(selection)
        if select_atoms is None:
            _pycharmm_lingo().charmm_script(
                f"cons harm force {force} {_script_selection(selection)}"
            )
        else:
            ok = _pycharmm_restraints().harmonic_absolute(
                selection=select_atoms,
                force_const=force,
            )
            if ok is False:
                raise SystemLoadError("pyCHARMM restraints.harmonic_absolute returned failure")
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "setting harmonic constraints") from exc


def cons_harm_clear() -> None:
    try:
        ok = _pycharmm_restraints().harmonic_turn_off()
        if ok is False:
            raise SystemLoadError("pyCHARMM restraints.harmonic_turn_off returned failure")
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "clearing harmonic constraints") from exc


def noe_reset() -> None:
    try:
        ok = _pycharmm_restraints().noe_reset()
        if ok is False:
            raise SystemLoadError("pyCHARMM restraints.noe_reset returned failure")
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "resetting NOE restraints") from exc


def nbonds_setup(
    *,
    cutnb: float,
    ctofnb: float,
    ctonnb: float,
    cutim: float | None = None,
    eps: float = 1.0,
    e14fac: float = 1.0,
    wmin: float = 1.5,
    elec: bool | None = None,
    atom: bool = True,
    vatom: bool = True,
    cdie: bool = True,
    switch: bool = True,
    vswitch: bool = True,
    fshift: bool = False,
    fswitch: bool = False,
    vfswitch: bool = False,
    ewald: bool = False,
    pmewald: bool = False,
    kappa: float | None = None,
    order: int | None = None,
    fftx: int | None = None,
    ffty: int | None = None,
    fftz: int | None = None,
    bycc: bool = False,
    bygr: bool = False,
    inbfrq: int = -1,
    imgfrq: int = -1,
    nbxmod: int | None = None,
) -> None:
    params: dict[str, Any] = {
        "cutnb": cutnb,
        "ctofnb": ctofnb,
        "ctonnb": ctonnb,
        "eps": eps,
        "e14fac": e14fac,
        "wmin": wmin,
        "atom": atom,
        "vatom": vatom,
        "cdie": cdie,
        "switch": switch,
        "vswitch": vswitch,
        "fshift": fshift,
        "fswitch": fswitch,
        "vfswitch": vfswitch,
        "ewald": ewald,
        "pmewald": pmewald,
        "bycc": bycc,
        "bygr": bygr,
        "inbfrq": inbfrq,
        "imgfrq": imgfrq,
    }
    if cutim is not None:
        params["cutim"] = cutim
    if elec is not None:
        params["elec"] = elec
    if nbxmod is not None:
        params["nbxmod"] = nbxmod
    if kappa is not None:
        params["kappa"] = kappa
    if order is not None:
        params["order"] = order
    if fftx is not None:
        params["fftx"] = fftx
    if ffty is not None:
        params["ffty"] = ffty
    if fftz is not None:
        params["fftz"] = fftz
    try:
        _pycharmm_root().NonBondedScript(**params).run()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "configuring nonbonded interactions") from exc


def shake_on(
    *,
    tol: float = 1.0e-6,
    params: bool = True,
    bonh: bool = True,
    fast: bool = True,
) -> None:
    try:
        _pycharmm_shake().on(tol=tol, param=params, bonh=bonh, fast=fast)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "enabling SHAKE") from exc


def shake_off() -> None:
    try:
        _pycharmm_shake().off()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "disabling SHAKE") from exc


def blade_on(
    *,
    gpu_id: int | None = None,
    gpu_ids: int | str | Sequence[int] | None = None,
    faster: bool = False,
) -> None:
    if gpu_id is not None and gpu_ids is not None:
        raise ValueError("blade_on accepts either gpu_id or gpu_ids, not both")
    resolved_gpu_ids = gpu_ids if gpu_ids is not None else gpu_id
    try:
        if faster:
            _pycharmm_miscom().faster(True)
        if resolved_gpu_ids is None:
            _pycharmm_blade().enable()
        else:
            _pycharmm_blade().enable(gpu_ids=resolved_gpu_ids)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "enabling BLaDE") from exc


def blade_off() -> None:
    try:
        _pycharmm_blade().disable()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "disabling BLaDE") from exc


def blade_energy(*, show: bool = False, abic: bool = False) -> None:
    try:
        _pycharmm_blade().energy(show=show, abic=abic)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "running BLaDE energy") from exc


def clear_block() -> None:
    try:
        ok = _pycharmm_block().clear()
        if ok is False:
            raise SystemLoadError("pyCHARMM block.clear returned failure")
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "clearing BLOCK state") from exc


def _resolve_stream_path(
    path: Path | str,
    *,
    search_path: Sequence[Path | str] | None,
) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        raise SystemLoadError(f"stream_file: could not resolve {path} against {search_path}")

    for directory in search_path or ():
        resolved = Path(directory) / candidate
        if resolved.exists():
            return resolved

    env_toppar = os.environ.get("CPHMD_TOPPAR_DIR")
    if env_toppar:
        for resolved in (Path(env_toppar) / candidate, Path(env_toppar) / "my_files" / candidate):
            if resolved.exists():
                return resolved

    for resolved in (
        TOPPAR_DIR / candidate,
        TOPPAR_DIR / "my_files" / candidate,
        Path.cwd() / candidate,
    ):
        if resolved.exists():
            return resolved

    raise SystemLoadError(f"stream_file: could not resolve {path} against {search_path}")


def stream_file(path: Path | str, *, search_path: Sequence[Path | str] | None = None) -> None:
    resolved = _resolve_stream_path(path, search_path=search_path)
    try:
        _pycharmm_read().stream(qpath(resolved))
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"streaming CHARMM file {resolved}") from exc


def disable_autogen() -> None:
    try:
        ok = _pycharmm_generate().autogenerate(patch_mode=False)
        if ok is False:
            raise SystemLoadError("pyCHARMM generate.autogenerate returned failure")
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "disabling autogen patching") from exc


def set_output_unit(unit: int) -> None:
    try:
        _pycharmm_settings().set_output_unit(unit)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"setting output unit {unit}") from exc


def reset_io_unit(unit: int = 91) -> None:
    try:
        lingo = _pycharmm_lingo()
        lingo.charmm_script(f"open unit {unit} write form name /dev/null")
        lingo.charmm_script(f"close unit {unit}")
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"resetting I/O unit {unit}") from exc


def set_prnlev(level: int) -> None:
    try:
        _pycharmm_settings().set_verbosity(level)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"setting PRNLEV {level}") from exc


def set_warn_level(level: int) -> None:
    try:
        _pycharmm_settings().set_warn_level(level)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"setting warning level {level}") from exc


def set_bomb_level(level: int) -> None:
    try:
        _pycharmm_settings().set_bomb_level(level)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, f"setting bomb level {level}") from exc


def set_iofmt(*, extended: bool = True) -> None:
    try:
        _pycharmm_miscom().set_io_format(extended=extended)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "setting CHARMM IO format") from exc


def minimize_sd(
    *,
    nsteps: int,
    nprint: int | None = None,
    step: float | None = None,
    tolenr: float | None = None,
    tolgrd: float = 0.01,
) -> None:
    kwargs: dict[str, Any] = {"nstep": nsteps, "tolgrd": tolgrd}
    if nprint is not None:
        kwargs["nprint"] = nprint
    if step is not None:
        kwargs["step"] = step
    if tolenr is not None:
        kwargs["tolenr"] = tolenr
    try:
        _pycharmm_minimize().run_sd(**kwargs)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "running SD minimization") from exc


def minimize_abnr(*, nsteps: int, tolgrd: float = 0.0001) -> None:
    try:
        _pycharmm_minimize().run_abnr(nstep=nsteps, tolgrd=tolgrd)
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "running ABNR minimization") from exc


def energy_show() -> None:
    try:
        _pycharmm_energy().show()
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "showing energy") from exc


def energy_get_total() -> float:
    try:
        return float(_pycharmm_energy().get_total())
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "querying total energy") from exc


def energy_get_gradient_rms() -> float:
    try:
        return float(_pycharmm_energy().get_grms())
    except Exception as exc:
        raise wrap_exception(exc, SystemLoadError, "querying gradient RMS") from exc
