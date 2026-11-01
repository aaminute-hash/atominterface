from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from atominterface._charge._constants import (
    _MAX_THICKNESS_PASSES,
    ANALYSIS_CHARGE,
    CHARGE_KNOWN,
    CHARGE_SOURCE,
)
from atominterface._charge.analyze import analyze_charge_polarity
from atominterface._charge.orient import _orient_slab
from atominterface._charge.policy import tasker_classification
from atominterface._charge.resolve import _real_number
from atominterface.thermo.surface import slab_formula_units


@dataclass
class SymmetricSlab:
    """A mirror-symmetric slab plus the diagnostics needed to judge it.

    ``structure`` is a pymatgen ``Structure`` carrying vacuum, matching the
    type returned by the builders' termination enumeration.  A Tasker
    Type 3 surface necessarily yields a non-stoichiometric board (e.g.
    rocksalt (111) gives Na3Cl4 with a nominal charge of -1 e per board);
    that is reported in ``warnings`` instead of being silently "fixed".
    """

    structure: Any
    area: float
    n_formula_units: float
    nominal_charge: float | None
    formula: str
    tasker_type: int | None
    warnings: list[str]


def finite_slab_geometry(structure: Any) -> dict[str, Any]:
    """Declare a generated slab as a finite board for the charge analysis.

    Returns the ``normal_axis`` / ``finite_slab`` / ``vacuum_reference``
    triple accepted by :func:`analyze_charge_polarity`.  Structures without
    slab provenance (for example a bare bulk cell) are reported as *not*
    finite rather than certified by guesswork.
    """
    matrix = np.asarray(structure.lattice.matrix, dtype=np.float64)
    normal = np.cross(matrix[0], matrix[1])
    area = float(np.linalg.norm(normal))
    if not np.isfinite(area) or area <= 0:
        return {"normal_axis": 2, "finite_slab": False, "vacuum_reference": None}
    normal = normal / area
    height = float(matrix[2] @ normal)
    projected = np.asarray(structure.cart_coords, dtype=np.float64) @ normal
    bottom, top = float(np.min(projected)), float(np.max(projected))
    finite = bool(
        structure.properties.get("ai_slab_generated", False)
        and height > 0
        and top - bottom < height
    )
    return {
        "normal_axis": 2,
        "finite_slab": finite,
        "vacuum_reference": [top / height, 1.0 + bottom / height] if finite else None,
    }


def _select_sites(slab: Any, indices: Any, lattice: Any = None, coords: Any = None) -> Any:
    """Copy ``slab`` keeping the given site indices, with all site properties.

    ``coords`` optionally replaces the Cartesian coordinates (used when the
    selected sites have been mirrored); it must have one row per index.
    """
    from pymatgen.core import Structure

    chosen = np.asarray(indices, dtype=int)
    positions = (
        np.asarray(slab.cart_coords, dtype=np.float64)[chosen]
        if coords is None
        else np.asarray(coords, dtype=np.float64)
    )
    site_properties = {
        key: [values[index] for index in chosen] for key, values in slab.site_properties.items()
    }
    selected = Structure(
        slab.lattice if lattice is None else lattice,
        [slab.species[index] for index in chosen],
        positions,
        coords_are_cartesian=True,
        site_properties=site_properties,
    )
    selected.properties.update(slab.properties)
    return selected


def mirror_slab(slab: Any) -> Any:
    """Mirror a slab about its top plane into a same-termination board.

    The top layer stays where it is and every layer below it is reflected
    to the same distance above it, so the stacking sequence becomes a
    palindrome (NaCl (111) Na-Cl-Na becomes Na-Cl-Na-Cl-Na) with identical
    terminations on both surfaces.  Site properties, including the charge
    bookkeeping arrays, are duplicated alongside the atoms, and the vacuum
    is extended so the doubled board still fits its cell.  The input slab
    is not modified.
    """
    matrix = np.asarray(slab.lattice.matrix, dtype=np.float64)
    normal = np.cross(matrix[0], matrix[1])
    length = float(np.linalg.norm(normal))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("cannot mirror a slab with a degenerate in-plane cell")
    normal = normal / length

    coords = np.asarray(slab.cart_coords, dtype=np.float64)
    projected = coords @ normal
    top = float(np.max(projected))
    below = projected < top - 1e-8
    if not below.any():
        # A single-plane slab already exposes the same plane on both sides.
        return slab.copy()

    height = float(matrix[2] @ normal)
    vacuum = height - top
    new_top = 2.0 * top - float(np.min(projected[below]))
    lattice = slab.lattice
    if new_top + vacuum > height + 1e-9:
        from pymatgen.core import Lattice

        scaled = matrix.copy()
        scaled[2] = matrix[2] * ((new_top + vacuum) / height)
        lattice = Lattice(scaled)

    mirrored = coords[below] + (2.0 * (top - projected[below]))[:, None] * normal
    indices = np.concatenate([np.arange(len(slab)), np.flatnonzero(below)])
    new_coords = np.vstack([coords, mirrored])
    return _select_sites(slab, indices, lattice, coords=new_coords)


def _truncate_layers(slab: Any, max_layers: int) -> Any:
    """Keep only the ``max_layers`` lowest atomic planes of a slab."""
    matrix = np.asarray(slab.lattice.matrix, dtype=np.float64)
    normal = np.cross(matrix[0], matrix[1])
    length = float(np.linalg.norm(normal))
    if length <= 0:
        raise ValueError("cannot truncate a slab with a degenerate in-plane cell")
    projected = np.asarray(slab.cart_coords, dtype=np.float64) @ (normal / length)
    levels = np.unique(np.round(projected, 6))
    plane_of_atom = np.searchsorted(levels, np.round(projected, 6), side="right") - 1
    return _select_sites(slab, np.flatnonzero(plane_of_atom < max_layers))


def _as_structure_copy(structure: Any) -> Any:
    """Deep-copy an ASE Atoms or pymatgen Structure into a pymatgen Structure."""
    if hasattr(structure, "lattice") and hasattr(structure, "sites"):
        return structure.copy()
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_structure(structure)


def _normal_axis(structure: Any) -> np.ndarray:
    """Unit surface normal of a slab cell (cross product of the in-plane vectors)."""
    matrix = np.asarray(structure.lattice.matrix, dtype=np.float64)
    normal = np.cross(matrix[0], matrix[1])
    length = float(np.linalg.norm(normal))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("cannot handle a slab with a degenerate in-plane cell")
    return normal / length


def _normal_extent(structure: Any) -> float:
    """Thickness of a slab along its own surface normal, in Angstrom."""
    projected = np.asarray(structure.cart_coords, dtype=np.float64) @ _normal_axis(structure)
    return float(np.max(projected) - np.min(projected))


def _mark_generated(slab: Any) -> Any:
    """Flag a structure as produced by a real slab cut, not a bulk fallback."""
    slab.properties["ai_slab_generated"] = True
    return slab


def _site_dependent_charge(structure: Any) -> bool:
    """Whether species-only symmetry could merge different charge assignments.

    A structure whose same-species sites carry different resolved charges
    (or different known/source states) must be enumerated with
    ``primitive=False`` and charge-aware cuts, otherwise pymatgen's
    species-only symmetry merges terminations that the charge model
    distinguishes.
    """
    seen: dict[str, tuple] = {}
    for site in structure:
        signature = (
            site.properties.get(ANALYSIS_CHARGE),
            site.properties.get(CHARGE_KNOWN),
            site.properties.get(CHARGE_SOURCE),
        )
        species = str(site.species)
        if species in seen and seen[species] != signature:
            return True
        seen[species] = signature
    return False


def termination_cuts(generator: Any) -> list:
    """Enumerate every layer cut of a slab generator, charge-aware.

    Unlike ``generator.get_slabs()`` this keeps terminations that differ
    only by their per-site charges apart, because the enumeration is
    driven by the oriented-unit-cell layer levels rather than by
    species-only symmetry.  A single failing cut is skipped with a
    warning; the caller decides what an empty enumeration means.
    """
    levels = np.unique(np.round(generator.oriented_unit_cell.frac_coords[:, 2] % 1, 10))
    following = np.r_[levels[1:], levels[0] + 1]
    slabs = []
    for a, b in zip(levels, following, strict=True):
        try:
            slabs.append(generator.get_slab(shift=float((a + b) / 2 % 1)))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"skipping a termination cut: {exc}", stacklevel=2)
    return slabs


def _oriented_repeat_height(generator: Any) -> float:
    """Height of one oriented-unit-cell repeat along the surface normal."""
    cell = generator.oriented_unit_cell
    matrix = np.asarray(cell.lattice.matrix, dtype=np.float64)
    normal = np.cross(matrix[0], matrix[1])
    length = float(np.linalg.norm(normal))
    if length <= 0:
        raise ValueError("cannot handle an oriented unit cell with a degenerate in-plane cell")
    return abs(float(matrix[2] @ (normal / length)))


def _prefer_stoichiometric_termination(slabs: list) -> int:
    """Index of the first charge-neutral termination, or 0 when none is."""
    for index, slab in enumerate(slabs):
        marked = _mark_generated(slab)
        analysis = analyze_charge_polarity(marked, **finite_slab_geometry(marked))
        if analysis["groups"]["total"]["neutrality"] == "neutral":
            return index
    return 0


def _symmetric_options(
    min_thickness: Any, vacuum: Any, max_layers: Any
) -> tuple[float, float, int | None]:
    """Validate and normalize the symmetric-slab geometry options."""
    thickness = _real_number(min_thickness, "min_thickness")
    if thickness <= 0:
        raise ValueError("min_thickness must be positive")
    padding = _real_number(vacuum, "vacuum")
    if padding < 0:
        raise ValueError("vacuum must be non-negative")
    layers: int | None = None
    if max_layers is not None:
        if isinstance(max_layers, bool) or int(max_layers) < 1:
            raise ValueError("max_layers must be a positive integer")
        layers = int(max_layers)
    return thickness, padding, layers


def _select_termination_index(slabs: list, termination: int | None, miller) -> int:
    """Index of the requested cut, or the preferred one when unspecified."""
    if termination is None:
        return _prefer_stoichiometric_termination(slabs)
    index = int(termination)
    if not 0 <= index < len(slabs):
        raise ValueError(
            f"termination {index} is out of range for Miller indices "
            f"{tuple(miller)}: {len(slabs)} termination(s) available"
        )
    return index


def _symmetric_board_notes(
    nominal_charge: float | None, tasker_type: int | None, polar: bool
) -> list[str]:
    """Honest notes about a symmetric board that is still not clean."""
    notes: list[str] = []
    if nominal_charge is not None and abs(nominal_charge) > 1e-9:
        notes.append(
            f"non-stoichiometric symmetric board: nominal charge {nominal_charge:+.3f} e "
            "per board; its surface energy depends on the bulk-reference assumption"
        )
    if polar:
        notes.append("symmetric board still carries a nonzero normal dipole")
    if tasker_type is None:
        notes.append("tasker type unknown: the supplied charge model is incomplete")
    return notes


def make_symmetric_slab(
    structure: Any,
    miller: tuple[int, int, int],
    *,
    termination: int | None = None,
    min_thickness: float = 8.0,
    vacuum: float = 15.0,
    max_layers: int | None = None,
) -> SymmetricSlab:
    """Build a mirror-symmetric slab with the same termination on both surfaces.

    A half-board is cut from ``structure`` along ``miller`` and mirrored
    about its top plane, so the finished board is at least ``min_thickness``
    thick, exposes one termination on both sides, and carries the requested
    vacuum.  ``termination`` selects the cut (``None`` prefers a
    charge-neutral one, else the first).  For a Tasker Type 3 surface the
    symmetric board is necessarily non-stoichiometric; its nominal charge
    and a warning are reported, never silently compensated.  The input
    structure is never modified.
    """
    from pymatgen.core.surface import SlabGenerator

    thickness, padding, layers = _symmetric_options(min_thickness, vacuum, max_layers)
    bulk = _as_structure_copy(structure)
    # SlabGenerator counts a half-board in whole oriented-unit-cell repeats
    # (ceil(min_slab_size / repeat_height)); start at half the requested
    # board thickness and add repeats until the mirrored board is thick
    # enough, since mirroring shares the top plane with the appended half.
    # Charge-distinct terminations must not be merged by species-only
    # symmetry, exactly as in the builder's own enumeration.
    preserve_sites = _site_dependent_charge(bulk)
    min_half = thickness / 2.0
    symmetric = None
    for _ in range(_MAX_THICKNESS_PASSES):
        generator = SlabGenerator(
            bulk,
            miller_index=tuple(int(index) for index in miller),
            min_slab_size=min_half,
            min_vacuum_size=padding,
            in_unit_planes=False,
            center_slab=True,
            primitive=not preserve_sites,
        )
        slabs = termination_cuts(generator) if preserve_sites else generator.get_slabs()
        if not slabs:
            raise ValueError(f"no slab could be generated for Miller indices {tuple(miller)}")
        slab = _mark_generated(slabs[_select_termination_index(slabs, termination, miller)])
        # Level the half-board before it is mirrored.  For anything but
        # (001) the generator's in-plane vectors carry a z component, and
        # the finished board inherits that skew -- which is what made the
        # builder read a projected cell further downstream.  Mirroring,
        # layer truncation and the extent check all project onto
        # ``cross(a, b)``, so none of them cares which frame the board is
        # in, and rotating first leaves every one of their numbers alone
        # while making the board the builders expect.
        slab = _orient_slab(slab)
        if layers is not None:
            slab = _truncate_layers(slab, layers)
        symmetric = mirror_slab(slab)
        if layers is not None or _normal_extent(symmetric) >= thickness - 1e-9:
            break
        min_half += _oriented_repeat_height(generator)
    else:  # pragma: no cover - bounded safety net
        raise ValueError(
            f"could not reach a symmetric slab of {thickness} A for Miller indices {tuple(miller)}"
        )

    analysis = analyze_charge_polarity(symmetric, **finite_slab_geometry(symmetric))
    nominal_charge = analysis["groups"]["total"]["net_charge_e"]
    tasker_type = tasker_classification(analysis)
    polar = any(group["polarity"] == "polar" for group in analysis["groups"].values())

    return SymmetricSlab(
        structure=symmetric,
        area=float(
            np.linalg.norm(np.cross(symmetric.lattice.matrix[0], symmetric.lattice.matrix[1]))
        ),
        n_formula_units=slab_formula_units(symmetric, bulk),
        nominal_charge=nominal_charge,
        formula=str(symmetric.composition.reduced_formula),
        tasker_type=tasker_type,
        warnings=_symmetric_board_notes(nominal_charge, tasker_type, polar),
    )
