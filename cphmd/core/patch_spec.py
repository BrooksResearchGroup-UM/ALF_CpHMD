"""Patch specification data models and in-memory registry.

import DAG invariant: this module is pure Python data. It must not import pyCHARMM,
must not import cphmd.native, and must not import cphmd.core.patching.
Runtime CHARMM effects belong in cphmd.core.patch_applier.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Callable, Literal

AtomOpKind = Literal["type", "delete", "rename"]
BondOpKind = Literal["add", "delete"]
ImproperOpKind = Literal["add", "delete"]
BondDouble = tuple[str, ...]


class PatchSpecValidationError(ValueError):
    """Raised when patch token validation fails."""


def _is_numeric(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _require_value(value: str, field_name: str) -> str:
    if not value:
        raise ValueError(f"{field_name} is required")
    return value


def _validate_charmm_token(token: str, field_name: str, allow_star_prefix: bool = False) -> str:
    if not token:
        raise PatchSpecValidationError(f"{field_name} token is required")
    if allow_star_prefix and token.startswith("*"):
        base = token[1:]
        if not base or len(base) > 4:
            raise PatchSpecValidationError(
                f"{field_name} token '{token}' exceeds CHARMM length constraints"
            )
        return token
    if len(token) > 4:
        raise PatchSpecValidationError(
            f"{field_name} token '{token}' exceeds CHARMM length constraints"
        )
    return token


@dataclass(frozen=True)
class AtomOp:
    kind: AtomOpKind
    atom_name: str
    atom_type: str | None = None
    charge: float | None = None
    new_name: str | None = None

    def __post_init__(self) -> None:
        _validate_charmm_token(self.atom_name, "atom_name")
        if self.kind == "type":
            if self.atom_type is None or self.charge is None:
                raise ValueError("AtomOp kind='type' requires atom_type and charge")
            _validate_charmm_token(self.atom_type, "atom_type")
            if not _is_numeric(self.charge):
                raise ValueError("AtomOp kind='type' charge must be numeric")
        if self.kind == "rename":
            if not self.new_name:
                raise ValueError("AtomOp kind='rename' requires new_name")
            _validate_charmm_token(self.new_name, "new_name")


@dataclass(frozen=True)
class BondOp:
    kind: BondOpKind
    atoms: tuple[str, str]

    def __post_init__(self) -> None:
        if len(self.atoms) != 2:
            raise ValueError("BondOp requires exactly two atoms")
        _validate_charmm_token(self.atoms[0], "atoms[0]")
        _validate_charmm_token(self.atoms[1], "atoms[1]")


@dataclass(frozen=True)
class ImproperOp:
    kind: ImproperOpKind
    atoms: tuple[str, str, str, str]

    def __post_init__(self) -> None:
        if len(self.atoms) != 4:
            raise ValueError("ImproperOp requires exactly four atoms")
        for idx, atom in enumerate(self.atoms):
            _validate_charmm_token(atom, f"atoms[{idx}]")


@dataclass(frozen=True)
class ICRecord:
    atoms: tuple[str, str, str, str]
    params: tuple[float, float, float, float, float]

    def __post_init__(self) -> None:
        if len(self.atoms) != 4:
            raise ValueError("ICRecord requires exactly four atoms")
        for idx, atom in enumerate(self.atoms):
            _validate_charmm_token(atom, f"atoms[{idx}]", allow_star_prefix=True)

        if len(self.params) != 5:
            raise ValueError("ICRecord requires exactly five numeric values")
        if not all(_is_numeric(value) for value in self.params):
            raise ValueError("ICRecord params must be numeric")

    @property
    def values(self) -> tuple[float, float, float, float, float]:
        return self.params


@dataclass(frozen=True)
class PatchPreCondition:
    applies_to_resnames: frozenset[str]
    custom: Callable[..., bool] | None = None

    def __post_init__(self) -> None:
        normalized = frozenset(self.applies_to_resnames)
        if not normalized:
            raise ValueError("PatchPreCondition.applies_to_resnames is required")
        for resname in normalized:
            _validate_charmm_token(resname, "applies_to_resnames")
        object.__setattr__(self, "applies_to_resnames", normalized)

    @classmethod
    def for_resnames(cls, *resnames: str) -> PatchPreCondition:
        return cls(applies_to_resnames=frozenset(resnames))

    def matches(self, resname: str, *args, **kwargs) -> bool:
        if resname not in self.applies_to_resnames:
            return False
        if self.custom is None:
            return True
        return bool(self.custom(resname, *args, **kwargs))


@dataclass(frozen=True)
class PatchSpec:
    name: str
    source_resname: str
    target_resname: str
    charge: float = 0.0
    comment: str = ""
    atom_types: tuple[AtomOp, ...] = ()
    atom_deletes: tuple[AtomOp, ...] = ()
    atom_renames: tuple[AtomOp, ...] = ()
    bond_adds: tuple[BondOp, ...] = ()
    bond_deletes: tuple[BondOp, ...] = ()
    bond_doubles: tuple[BondDouble, ...] = ()
    improper_adds: tuple[ImproperOp, ...] = ()
    improper_deletes: tuple[ImproperOp, ...] = ()
    ic_records: tuple[ICRecord, ...] = ()
    donors: tuple[tuple[str, str], ...] = ()
    acceptors: tuple[str, ...] = ()
    atom_group_starts: tuple[str, ...] = ()
    pre_conditions: PatchPreCondition | None = None
    priority: int = 100
    ordering_priority: int | None = None
    rename_order: Literal["before_patch", "after_patch"] = "after_patch"
    source: str = ""

    def __post_init__(self) -> None:
        _require_value(self.name, "name")
        _validate_charmm_token(self.name, "name")
        _validate_charmm_token(self.source_resname, "source_resname")
        _validate_charmm_token(self.target_resname, "target_resname")
        if not _is_numeric(self.charge):
            raise ValueError("PatchSpec.charge must be numeric")

        if self.rename_order not in {"before_patch", "after_patch"}:
            raise ValueError("rename_order must be either 'before_patch' or 'after_patch'")

        atom_types = tuple(self.atom_types)
        atom_deletes = tuple(self.atom_deletes)
        atom_renames = tuple(self.atom_renames)
        bond_adds = tuple(self.bond_adds)
        bond_deletes = tuple(self.bond_deletes)
        bond_doubles = tuple(tuple(atoms) for atoms in self.bond_doubles)
        improper_adds = tuple(self.improper_adds)
        improper_deletes = tuple(self.improper_deletes)
        ic_records = tuple(self.ic_records)
        donors = tuple(self.donors)
        acceptors = tuple(self.acceptors)
        atom_group_starts = tuple(self.atom_group_starts)
        pre_conditions = self.pre_conditions or PatchPreCondition.for_resnames(self.source_resname)

        if self.source_resname not in pre_conditions.applies_to_resnames:
            raise ValueError(
                "source_resname must be included in pre_conditions.applies_to_resnames"
            )

        priority = self.priority
        ordering_priority = self.ordering_priority
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise ValueError("PatchSpec.priority must be an integer")
        if ordering_priority is not None:
            if not isinstance(ordering_priority, int) or isinstance(ordering_priority, bool):
                raise ValueError("PatchSpec.ordering_priority must be an integer")
            if self.priority != 100 and self.priority != ordering_priority:
                raise ValueError(
                    "priority and ordering_priority conflict; provide one value or matching values"
                )
            if self.priority == 100:
                priority = ordering_priority
        ordering_priority = priority

        for atom_op in atom_types:
            if atom_op.kind != "type":
                raise ValueError("atom_types may only contain AtomOp kind='type'")
        for atom_op in atom_deletes:
            if atom_op.kind != "delete":
                raise ValueError("atom_deletes may only contain AtomOp kind='delete'")
        for atom_op in atom_renames:
            if atom_op.kind != "rename":
                raise ValueError("atom_renames may only contain AtomOp kind='rename'")
        for bond_op in bond_adds:
            if bond_op.kind != "add":
                raise ValueError("bond_adds may only contain BondOp kind='add'")
        for bond_op in bond_deletes:
            if bond_op.kind != "delete":
                raise ValueError("bond_deletes may only contain BondOp kind='delete'")
        for double_idx, atoms in enumerate(bond_doubles):
            if len(atoms) < 2 or len(atoms) % 2 != 0:
                raise ValueError("bond_doubles entries must contain atom pairs")
            for atom_idx, atom in enumerate(atoms):
                _validate_charmm_token(atom, f"bond_doubles[{double_idx}][{atom_idx}]")
        for improper_op in improper_adds:
            if improper_op.kind != "add":
                raise ValueError("improper_adds may only contain ImproperOp kind='add'")
        for improper_op in improper_deletes:
            if improper_op.kind != "delete":
                raise ValueError("improper_deletes may only contain ImproperOp kind='delete'")

        for donor in donors:
            if len(donor) != 2:
                raise ValueError("donors entries must have exactly two atom tokens")
            _validate_charmm_token(donor[0], "donors[0]")
            _validate_charmm_token(donor[1], "donors[1]")
        for idx, acceptor in enumerate(acceptors):
            _validate_charmm_token(acceptor, f"acceptors[{idx}]")
        for idx, atom_name in enumerate(atom_group_starts):
            _validate_charmm_token(atom_name, f"atom_group_starts[{idx}]")
        atom_type_names = {atom_op.atom_name for atom_op in atom_types}
        missing_group_starts = [
            atom_name for atom_name in atom_group_starts if atom_name not in atom_type_names
        ]
        if missing_group_starts:
            raise ValueError(
                "atom_group_starts entries must reference atom_types: "
                f"{', '.join(missing_group_starts)}"
            )

        object.__setattr__(self, "atom_types", atom_types)
        object.__setattr__(self, "atom_deletes", atom_deletes)
        object.__setattr__(self, "atom_renames", atom_renames)
        object.__setattr__(self, "bond_adds", bond_adds)
        object.__setattr__(self, "bond_deletes", bond_deletes)
        object.__setattr__(self, "bond_doubles", bond_doubles)
        object.__setattr__(self, "improper_adds", improper_adds)
        object.__setattr__(self, "improper_deletes", improper_deletes)
        object.__setattr__(self, "ic_records", ic_records)
        object.__setattr__(self, "donors", donors)
        object.__setattr__(self, "acceptors", acceptors)
        object.__setattr__(self, "atom_group_starts", atom_group_starts)
        object.__setattr__(self, "pre_conditions", pre_conditions)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "ordering_priority", ordering_priority)

    def matches(self, resname: str, *args, **kwargs) -> bool:
        return self.pre_conditions is not None and self.pre_conditions.matches(
            resname, *args, **kwargs
        )


try:
    PATCH_REGISTRY
except NameError:
    PATCH_REGISTRY: dict[str, PatchSpec] = {}
else:
    PATCH_REGISTRY.clear()


def register_patch(patch: PatchSpec) -> None:
    if patch.name in PATCH_REGISTRY:
        raise ValueError(f"Patch '{patch.name}' is already registered")
    PATCH_REGISTRY[patch.name] = patch


def lookup(name: str) -> PatchSpec:
    try:
        return PATCH_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Patch '{name}' not found in PATCH_REGISTRY") from exc


def patches_for_resname(resname: str, *args, **kwargs) -> tuple[PatchSpec, ...]:
    indexed_matches = [
        (index, patch)
        for index, patch in enumerate(PATCH_REGISTRY.values())
        if patch.matches(resname, *args, **kwargs)
    ]
    indexed_matches.sort(
        key=lambda item: (
            item[1].priority,
            item[0],
        )
    )
    return tuple(patch for _, patch in indexed_matches)


HP1 = PatchSpec(
    name="HP1",
    source_resname="HSP",
    target_resname="HSD",
    charge=0.0,
    comment="Protonated His(HSP) to HSD",
    atom_types=(
        AtomOp(kind="type", atom_name="CB", atom_type="CT2", charge=-0.09),
        AtomOp(kind="type", atom_name="CD2", atom_type="CPH1", charge=0.22),
        AtomOp(kind="type", atom_name="HD2", atom_type="HR3", charge=0.10),
        AtomOp(kind="type", atom_name="CG", atom_type="CPH1", charge=-0.05),
        AtomOp(kind="type", atom_name="NE2", atom_type="NR2", charge=-0.70),
        AtomOp(kind="type", atom_name="ND1", atom_type="NR1", charge=-0.36),
        AtomOp(kind="type", atom_name="HD1", atom_type="H", charge=0.32),
        AtomOp(kind="type", atom_name="CE1", atom_type="CPH2", charge=0.25),
        AtomOp(kind="type", atom_name="HE1", atom_type="HR1", charge=0.13),
    ),
    atom_deletes=(AtomOp(kind="delete", atom_name="HE2"),),
    bond_doubles=(("O", "C", "CD2", "CG", "NE2", "CE1"),),
    improper_adds=(
        ImproperOp(kind="add", atoms=("ND1", "CG", "CE1", "HD1")),
        ImproperOp(kind="add", atoms=("CD2", "CG", "NE2", "HD2")),
        ImproperOp(kind="add", atoms=("CE1", "ND1", "NE2", "HE1")),
        ImproperOp(kind="add", atoms=("ND1", "CE1", "CG", "HD1")),
        ImproperOp(kind="add", atoms=("CD2", "NE2", "CG", "HD2")),
        ImproperOp(kind="add", atoms=("CE1", "NE2", "ND1", "HE1")),
    ),
    donors=(("HN", "N"), ("HD1", "ND1")),
    acceptors=("NE2",),
    atom_group_starts=("NE2",),
    rename_order="after_patch",
    source="his_patches.str PRES HP1",
)

Hp2 = PatchSpec(
    name="Hp2",
    source_resname="HSP",
    target_resname="HSE",
    charge=0.0,
    comment="Protonated His (HSP) to HSE",
    atom_types=(
        AtomOp(kind="type", atom_name="CB", atom_type="CT2", charge=-0.08),
        AtomOp(kind="type", atom_name="HB1", atom_type="HA2", charge=0.09),
        AtomOp(kind="type", atom_name="HB2", atom_type="HA2", charge=0.09),
        AtomOp(kind="type", atom_name="CD2", atom_type="CPH1", charge=-0.05),
        AtomOp(kind="type", atom_name="HD2", atom_type="HR3", charge=0.09),
        AtomOp(kind="type", atom_name="CG", atom_type="CPH1", charge=0.22),
        AtomOp(kind="type", atom_name="NE2", atom_type="NR1", charge=-0.36),
        AtomOp(kind="type", atom_name="HE2", atom_type="H", charge=0.32),
        AtomOp(kind="type", atom_name="ND1", atom_type="NR2", charge=-0.70),
        AtomOp(kind="type", atom_name="CE1", atom_type="CPH2", charge=0.25),
        AtomOp(kind="type", atom_name="HE1", atom_type="HR1", charge=0.13),
    ),
    atom_deletes=(AtomOp(kind="delete", atom_name="HD1"),),
    improper_adds=(
        ImproperOp(kind="add", atoms=("CD2", "CG", "NE2", "HD2")),
        ImproperOp(kind="add", atoms=("CE1", "ND1", "NE2", "HE1")),
        ImproperOp(kind="add", atoms=("CD2", "NE2", "CG", "HD2")),
        ImproperOp(kind="add", atoms=("CE1", "NE2", "ND1", "HE1")),
    ),
    donors=(),
    acceptors=("ND1",),
    atom_group_starts=("NE2",),
    rename_order="after_patch",
    source="his_patches.str PRES Hp2",
)

HSDP = PatchSpec(
    name="HSDP",
    source_resname="HSD",
    target_resname="HSP",
    charge=1.0,
    comment="neutral HIS, proton on ND1 ro HSP",
    atom_types=(
        AtomOp(kind="type", atom_name="CB", atom_type="CT2A", charge=-0.05),
        AtomOp(kind="type", atom_name="ND1", atom_type="NR3", charge=-0.51),
        AtomOp(kind="type", atom_name="HD1", atom_type="H", charge=0.44),
        AtomOp(kind="type", atom_name="CG", atom_type="CPH1", charge=0.19),
        AtomOp(kind="type", atom_name="CE1", atom_type="CPH2", charge=0.32),
        AtomOp(kind="type", atom_name="HE1", atom_type="HR2", charge=0.18),
        AtomOp(kind="type", atom_name="NE2", atom_type="NR3", charge=-0.51),
        AtomOp(kind="type", atom_name="HE2", atom_type="H", charge=0.44),
        AtomOp(kind="type", atom_name="CD2", atom_type="CPH1", charge=0.19),
        AtomOp(kind="type", atom_name="HD2", atom_type="HR1", charge=0.13),
    ),
    bond_adds=(BondOp(kind="add", atoms=("NE2", "HE2")),),
    improper_adds=(
        ImproperOp(kind="add", atoms=("NE2", "CD2", "CE1", "HE2")),
        ImproperOp(kind="add", atoms=("NE2", "CE1", "CD2", "HE2")),
    ),
    improper_deletes=(
        ImproperOp(kind="delete", atoms=("CD2", "CG", "NE2", "HD2")),
        ImproperOp(kind="delete", atoms=("CE1", "ND1", "NE2", "HE1")),
        ImproperOp(kind="delete", atoms=("CD2", "NE2", "CG", "HD2")),
        ImproperOp(kind="delete", atoms=("CE1", "NE2", "ND1", "HE1")),
    ),
    ic_records=(
        ICRecord(
            atoms=("CE1", "CD2", "*NE2", "HE2"),
            params=(1.3256, 108.8200, -172.9400, 125.5200, 1.0020),
        ),
    ),
    # Strict parity: source has misspelled "donoe he2 ne2", so no donor is registered
    # until the explicit Task 7.5 donor-repair step.
    donors=(),
    acceptors=(),
    atom_group_starts=("CE1",),
    rename_order="before_patch",
    source="his_patches.str PRES HSDP",
)

HSEP = PatchSpec(
    name="HSEP",
    source_resname="HSE",
    target_resname="HSP",
    charge=1.0,
    comment="neutral His, proton on NE2 to HSP",
    atom_types=(
        AtomOp(kind="type", atom_name="CB", atom_type="CT2A", charge=-0.05),
        AtomOp(kind="type", atom_name="ND1", atom_type="NR3", charge=-0.51),
        AtomOp(kind="type", atom_name="HD1", atom_type="H", charge=0.44),
        AtomOp(kind="type", atom_name="CG", atom_type="CPH1", charge=0.19),
        AtomOp(kind="type", atom_name="CE1", atom_type="CPH2", charge=0.32),
        AtomOp(kind="type", atom_name="HE1", atom_type="HR2", charge=0.18),
        AtomOp(kind="type", atom_name="NE2", atom_type="NR3", charge=-0.51),
        AtomOp(kind="type", atom_name="HE2", atom_type="H", charge=0.44),
        AtomOp(kind="type", atom_name="CD2", atom_type="CPH1", charge=0.19),
        AtomOp(kind="type", atom_name="HD2", atom_type="HR1", charge=0.13),
    ),
    bond_adds=(BondOp(kind="add", atoms=("ND1", "HD1")),),
    improper_adds=(
        ImproperOp(kind="add", atoms=("ND1", "CG", "CE1", "HD1")),
        ImproperOp(kind="add", atoms=("ND1", "CE1", "CG", "HD1")),
        ImproperOp(kind="add", atoms=("NE2", "CD2", "CE1", "HE2")),
        ImproperOp(kind="add", atoms=("NE2", "CE1", "CD2", "HE2")),
    ),
    improper_deletes=(
        ImproperOp(kind="delete", atoms=("CD2", "CG", "NE2", "HD2")),
        ImproperOp(kind="delete", atoms=("CE1", "ND1", "NE2", "HE1")),
        ImproperOp(kind="delete", atoms=("CD2", "NE2", "CG", "HD2")),
        ImproperOp(kind="delete", atoms=("CE1", "NE2", "ND1", "HE1")),
    ),
    ic_records=(
        ICRecord(
            atoms=("CE1", "CG", "*ND1", "HD1"),
            params=(1.3262, 108.9000, 171.4900, 126.0900, 1.0018),
        ),
    ),
    donors=(("HE2", "NE2"), ("HD1", "ND1")),
    acceptors=(),
    atom_group_starts=("NE2",),
    rename_order="before_patch",
    source="his_patches.str PRES HSEP",
)


def _register_builtin_patches() -> None:
    for patch in (HP1, Hp2, HSDP, HSEP):
        existing = PATCH_REGISTRY.get(patch.name)
        if existing is patch:
            continue
        register_patch(patch)


_register_builtin_patches()


__all__ = [
    "HP1",
    "Hp2",
    "HSDP",
    "HSEP",
    "AtomOp",
    "BondDouble",
    "BondOp",
    "ICRecord",
    "ImproperOp",
    "PATCH_REGISTRY",
    "PatchPreCondition",
    "PatchSpec",
    "PatchSpecValidationError",
    "lookup",
    "patches_for_resname",
    "register_patch",
]
