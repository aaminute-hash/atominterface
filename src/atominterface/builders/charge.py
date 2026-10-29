"""Explicit atomic-charge bookkeeping and finite-slab dipole diagnostics.

Unknown charges are not zero charges.  No oxidation states are guessed, and no
calculator is consulted or modified.  These diagnostics describe the supplied
charge model, not electronic polarization, interface stability, or a prescription
for compensating atoms or DFT dipole-correction settings.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum
from numbers import Real
from typing import Any

import numpy as np
from ase.data import atomic_numbers

from atominterface.thermo.surface import slab_formula_units

ANALYSIS_CHARGE = "ai_charge"
CHARGE_KNOWN = "ai_charge_known"
CHARGE_SOURCE = "ai_charge_source"
SLAB_ID = "ai_slab_id"

_POLICIES = {"report", "require-neutral", "require-nonpolar"}
_SOURCES = {
    "explicit_charges",
    "oxidation_mapping",
    "initial_charges",
    "oxidation_states",
    "unknown",
}
_GROUPS = ("total", "substrate", "film")

# Bound on the symmetric-slab thickening loop; each pass adds one
# oriented-unit-cell repeat to the half-board.
_MAX_THICKNESS_PASSES = 16


class _GeometryDefault:
    """Distinguish omitted geometry options from explicit default-valued options."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:
        return repr(self.value)


_DEFAULT_AXIS = _GeometryDefault(2)
_DEFAULT_FINITE = _GeometryDefault(False)
_DEFAULT_VACUUM = _GeometryDefault(None)


def _real_number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, not {value!r}")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _vector(data: Any, size: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(data, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional array of length {size}") from exc
    if result.ndim != 1 or len(result) != size:
        raise ValueError(f"{name} must be a one-dimensional array of length {size}")
    return result


def _numeric_data(data: Any, size: int, source: str, *, allow_unknown: bool) -> dict[str, Any]:
    raw = _vector(data, size, source)
    known = np.asarray([value is not None for value in raw], dtype=bool)
    if not allow_unknown and not known.all():
        raise ValueError(f"{source} must contain a finite real charge for every atom")
    values = np.asarray(
        [
            _real_number(value, source) if present else 0.0
            for value, present in zip(raw, known, strict=True)
        ],
        dtype=float,
    )
    sources = np.where(known, source, "unknown")
    return {"values": values, "known": known, "sources": sources}


def _validate_symbols(symbols: Any) -> list[str]:
    result = list(symbols)
    for symbol in result:
        if (
            not isinstance(symbol, str)
            or symbol not in atomic_numbers
            or atomic_numbers[symbol] == 0
        ):
            raise ValueError(f"Invalid element symbol: {symbol!r}")
    return result


def _mapping_data(symbols: list[str], mapping: Any) -> dict[str, Any]:
    if not isinstance(mapping, Mapping):
        raise ValueError("oxidation_states must be an element-to-oxidation-state mapping")
    _validate_symbols(mapping)
    values = {
        symbol: _real_number(value, f"oxidation state for {symbol}")
        for symbol, value in mapping.items()
    }
    return _numeric_data(
        [values.get(symbol) for symbol in symbols],
        len(symbols),
        "oxidation_mapping",
        allow_unknown=True,
    )


def _stored_data(stored: Any, size: int) -> dict[str, Any]:
    if not isinstance(stored, Mapping) or not {"values", "known", "sources"} <= stored.keys():
        raise ValueError("stored charge data must contain values, known and sources")
    raw_known = _vector(stored["known"], size, "stored known mask")
    if not all(isinstance(value, (bool, np.bool_)) for value in raw_known):
        raise ValueError("stored known mask must contain booleans")
    known = np.asarray(raw_known, dtype=bool)
    result = _numeric_data(stored["values"], size, "stored charges", allow_unknown=False)
    sources = _vector(stored["sources"], size, "stored sources")
    if not all(isinstance(value, str) and value in _SOURCES for value in sources):
        raise ValueError("stored sources must contain recognized charge-source names")
    if np.any(known != (sources != "unknown")):
        raise ValueError("stored known mask and sources must agree on unknown charges")
    result.update(
        values=np.where(known, result["values"], 0.0), known=known, sources=sources.astype(str)
    )
    return result


def resolve_charge_data(
    symbols: Any,
    *,
    charges: Any = None,
    oxidation_states: Any = None,
    stored: Any = None,
    initial_charges: Any = None,
    oxi_states: Any = None,
) -> dict[str, Any]:
    """Resolve one source, never fill missing entries from a lower-priority source.

    Priority is explicit ``charges``, element ``oxidation_states``, dedicated
    ``stored`` arrays, actually present ``initial_charges``, then site
    ``oxi_states``.  ``None`` means absent; an empty mapping means explicitly
    unknown.  Partial mappings and ``None`` site oxidation states stay unknown.
    Returned arrays are independent copies; unknown values have a zero storage
    placeholder and a false known mask.  Only the selected source is validated.
    """
    elements = _validate_symbols(symbols)
    supplied = [
        ("explicit_charges", charges),
        ("oxidation_mapping", oxidation_states),
        ("stored", stored),
        ("initial_charges", initial_charges),
        ("oxidation_states", oxi_states),
    ]
    available = [(source, data) for source, data in supplied if data is not None]
    if not available:
        result = _numeric_data([None] * len(elements), len(elements), "unknown", allow_unknown=True)
    else:
        source, data = available[0]
        if source == "oxidation_mapping":
            result = _mapping_data(elements, data)
        elif source == "stored":
            result = _stored_data(data, len(elements))
        else:
            result = _numeric_data(
                data, len(elements), source, allow_unknown=source == "oxidation_states"
            )
    result["overridden_sources"] = [source for source, _ in available[1:]]
    return result


def validate_charge_options(policy: str, charge_tolerance: float, dipole_tolerance: float) -> None:
    """Raise ``ValueError`` for an unknown policy or nonfinite/negative tolerance."""
    if not isinstance(policy, str) or policy not in _POLICIES:
        raise ValueError("charge policy must be report, require-neutral or require-nonpolar")
    for name, value in (
        ("charge_tolerance", charge_tolerance),
        ("dipole_tolerance", dipole_tolerance),
    ):
        if _real_number(value, name) < 0:
            raise ValueError(f"{name} must be nonnegative")


def _structure_data(atoms: Any) -> dict[str, Any]:
    """Adapt ASE or ordered pymatgen-like properties without importing pymatgen."""
    if hasattr(atoms, "get_chemical_symbols"):
        return {
            "symbols": atoms.get_chemical_symbols(),
            "positions": atoms.positions,
            "cell": np.asarray(atoms.cell),
            "pbc": atoms.pbc,
            "arrays": atoms.arrays,
            "info": atoms.info,
        }
    try:
        sites = list(atoms.sites)
        arrays = dict(atoms.site_properties)
        if "charge" in arrays and "initial_charges" not in arrays:
            arrays["initial_charges"] = arrays["charge"]
        if "oxi_states" not in arrays:
            states = [getattr(site.specie, "oxi_state", None) for site in sites]
            if any(state is not None for state in states):
                arrays["oxi_states"] = states
        return {
            "symbols": [site.specie.symbol for site in sites],
            "positions": atoms.cart_coords,
            "cell": atoms.lattice.matrix,
            "pbc": getattr(atoms.lattice, "pbc", (True, True, True)),
            "arrays": arrays,
            "info": getattr(atoms, "properties", {}),
        }
    except (AttributeError, TypeError) as exc:
        raise ValueError("Expected ASE Atoms or an ordered structure with site properties") from exc


def _resolve_structure_charges(
    data: dict[str, Any], charges: Any, oxidation_states: Any
) -> dict[str, Any]:
    arrays = data["arrays"]
    stored = None
    if any(key in arrays for key in (ANALYSIS_CHARGE, CHARGE_KNOWN, CHARGE_SOURCE)):
        stored = {
            field: arrays[key]
            for field, key in (
                ("values", ANALYSIS_CHARGE),
                ("known", CHARGE_KNOWN),
                ("sources", CHARGE_SOURCE),
            )
            if key in arrays
        }
    return resolve_charge_data(
        data["symbols"],
        charges=charges,
        oxidation_states=oxidation_states,
        stored=stored,
        initial_charges=arrays.get("initial_charges"),
        oxi_states=arrays.get("oxi_states"),
    )


def _geometry_options(info: Any, axis: Any, finite: Any, vacuum: Any) -> tuple[int, bool, Any]:
    context = info.get("charge_geometry", {})
    if not isinstance(context, Mapping):
        raise ValueError("charge_geometry must be a mapping")
    options = {"normal_axis": axis, "finite_slab": finite, "vacuum_reference": vacuum}
    for key, value in options.items():
        if isinstance(value, _GeometryDefault):
            options[key] = context.get(key, value.value)
    axis = options["normal_axis"]
    if (
        isinstance(axis, (bool, np.bool_))
        or not isinstance(axis, (int, np.integer))
        or axis not in (0, 1, 2)
    ):
        raise ValueError("normal_axis must be 0, 1 or 2")
    finite = options["finite_slab"]
    if not isinstance(finite, (bool, np.bool_)):
        raise ValueError("finite_slab must be a boolean")
    return int(axis), bool(finite), options["vacuum_reference"]


def _vacuum_interval(reference: Any) -> tuple[list[float], float, float] | None:
    if reference is None:
        return None
    raw = _vector(reference, 2, "vacuum_reference")
    start, end = [_real_number(value, "vacuum_reference") for value in raw]
    delta = end - start
    if not 0 < abs(delta) < 1:
        raise ValueError(
            "vacuum_reference must describe a nonempty interval shorter than one period"
        )
    width = delta % 1.0
    return [start, end], start % 1.0, width


def _normal_geometry(
    data: dict[str, Any], axis: int
) -> tuple[np.ndarray, float, np.ndarray, float]:
    try:
        positions = np.asarray(data["positions"], dtype=float)
        cell = np.asarray(data["cell"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("positions and cell must contain finite real coordinates") from exc
    if positions.shape != (len(data["symbols"]), 3) or not np.isfinite(positions).all():
        raise ValueError("positions must be a finite (n_atoms, 3) array")
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        raise ValueError("cell must be a finite (3, 3) array")
    cross = np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3])
    area = float(np.linalg.norm(cross))
    if not np.isfinite(area) or area <= 0:
        raise ValueError("in-plane cell area must be positive and finite")
    normal = cross / area
    height = float(cell[axis] @ normal)
    if height < 0:
        normal, height = -normal, -height
    projected = positions @ normal
    if not np.isfinite(projected).all() or not np.isfinite(height):
        raise ValueError("projected coordinates and cell height must be finite")
    return projected, area, normal, height


def _unwrap_periodic(
    projected: np.ndarray,
    height: float,
    interval: tuple[list[float], float, float] | None,
    geometry: dict[str, Any],
) -> np.ndarray:
    if height <= 0:
        raise ValueError("periodic normal cell height must be positive")
    if interval is None:
        geometry["reasons"].append("vacuum_reference_missing")
        return projected
    _, start, width = interval
    relative = (projected / height - start) % 1.0
    # The reference is an open interval: builders may place its endpoints on
    # the outermost surface atoms.  Ignore only roundoff at those boundaries.
    occupied = (relative > 1e-12) & (relative < width - 1e-12)
    if occupied.any():
        geometry["reasons"].append("vacuum_reference_not_empty")
        return projected
    cut = (start + width / 2.0) % 1.0
    geometry["vacuum_cut_fractional"] = float(cut)
    return ((projected / height - cut) % 1.0) * height


def _prepare_geometry(
    data: dict[str, Any], axis: Any, finite: Any, vacuum: Any
) -> tuple[dict[str, Any], np.ndarray]:
    axis, finite, vacuum = _geometry_options(data["info"], axis, finite, vacuum)
    interval = _vacuum_interval(vacuum)
    projected, area, normal, height = _normal_geometry(data, axis)
    periodic = bool(data["pbc"][axis])
    geometry: dict[str, Any] = {
        "normal_axis": axis,
        "finite_slab": finite,
        "vacuum_reference": interval[0] if interval else None,
        "normal_vector": normal.tolist(),
        "area_A2": area,
        "normal_periodic": periodic,
        "normal_period_A": height if periodic else None,
        "vacuum_cut_fractional": None,
        "reasons": [],
    }
    if not finite:
        geometry["reasons"].append("finite_slab_not_declared")
    if periodic:
        projected = _unwrap_periodic(projected, height, interval, geometry)
    geometry["dipole_geometry_valid"] = not geometry["reasons"]
    return geometry, projected


def _group_masks(tags: Any, size: int) -> dict[str, np.ndarray | None]:
    masks: dict[str, np.ndarray | None] = {
        "total": np.ones(size, dtype=bool),
        "substrate": None,
        "film": None,
    }
    if tags is None:
        return masks
    raw = _vector(tags, size, "slab_tags")
    if not all(
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and value in (0, 1)
        for value in raw
    ):
        raise ValueError("slab_tags must contain integer 0 (substrate) or 1 (film)")
    masks.update(substrate=raw == 0, film=raw == 1)
    return masks


def _finite_sum(values: Any, name: str) -> float:
    try:
        result = float(fsum(values))
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} cannot be represented as a finite value") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} cannot be represented as a finite value")
    return result


def _group_analysis(
    mask: np.ndarray | None,
    data: dict[str, Any],
    geometry: dict[str, Any],
    projected: np.ndarray,
    charge_tolerance: float,
    dipole_tolerance: float,
) -> dict[str, Any]:
    group: dict[str, Any] = {
        "n_atoms": 0,
        "n_known": 0,
        "net_charge_e": None,
        "neutrality": "unknown",
        "dipole_normal_eA": None,
        "dipole_density_e_per_A": None,
        "polarity": "unknown",
        "sources": [],
        "reasons": [],
    }
    if mask is None:
        group["reasons"].append("slab_tags_missing")
        return group
    group["n_atoms"] = int(np.count_nonzero(mask))
    group["n_known"] = int(np.count_nonzero(data["known"][mask]))
    group["sources"] = sorted(set(data["sources"][mask].tolist()))
    if len(set(group["sources"]) - {"unknown"}) > 1:
        group["reasons"].append("mixed_charge_sources")
    if group["n_atoms"] == 0:
        group["reasons"].append("empty_group")
        return group
    if group["n_known"] != group["n_atoms"]:
        group["reasons"].append("charge_data_incomplete")
        return group
    charges = data["values"][mask]
    net = _finite_sum(charges, "net charge")
    group["net_charge_e"] = net
    group["neutrality"] = "neutral" if abs(net) <= charge_tolerance else "charged"
    if group["neutrality"] != "neutral":
        group["reasons"].append("nonneutral_group")
    elif geometry["dipole_geometry_valid"]:
        coordinates = projected[mask]
        center = _finite_sum(coordinates / len(coordinates), "geometric center")
        dipole = _finite_sum(charges * (coordinates - center), "normal dipole")
        density = dipole / geometry["area_A2"]
        if not np.isfinite(density):
            raise ValueError("dipole density must be finite")
        group.update(dipole_normal_eA=dipole, dipole_density_e_per_A=density)
        group["polarity"] = "polar" if abs(density) > dipole_tolerance else "nonpolar"
    else:
        group["reasons"].extend(geometry["reasons"])
    return group


def analyze_charge_polarity(
    atoms: Any,
    slab_tags: Any = None,
    *,
    charges: Any = None,
    oxidation_states: Any = None,
    normal_axis: Any = _DEFAULT_AXIS,
    finite_slab: Any = _DEFAULT_FINITE,
    vacuum_reference: Any = _DEFAULT_VACUUM,
    charge_tolerance: float = 1e-6,
    dipole_tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Return JSON-safe charge and finite-slab normal-dipole diagnostics.

    Geometry defaults are axis 2, no finite-slab declaration, and no vacuum
    reference; omitted options may instead come from ``info['charge_geometry']``.
    Explicit options (including False/None) override that context.  Old analysis
    snapshots are never read.  A periodic normal requires an explicitly empty
    normalized vacuum interval, such as (0.9, 1.1) or (0.9, 0.1), whose midpoint
    is the common unwrapping cut for every group.  Nonperiodic normals use raw
    coordinates.  Neutral groups use their geometric centers as dipole origins,
    making even charge-tolerance residuals invariant under rigid translation.
    """
    validate_charge_options("report", charge_tolerance, dipole_tolerance)
    structure = _structure_data(atoms)
    charge_data = _resolve_structure_charges(structure, charges, oxidation_states)
    geometry, projected = _prepare_geometry(structure, normal_axis, finite_slab, vacuum_reference)
    tags = structure["arrays"].get(SLAB_ID) if slab_tags is None else slab_tags
    masks = _group_masks(tags, len(structure["symbols"]))
    groups = {
        name: _group_analysis(
            mask, charge_data, geometry, projected, charge_tolerance, dipole_tolerance
        )
        for name, mask in masks.items()
    }
    return {
        "schema_version": 1,
        "model": "supplied_atomic_charge_bookkeeping",
        "groups": groups,
        "thresholds": {
            "charge_tolerance_e": float(charge_tolerance),
            "dipole_tolerance_e_per_A": float(dipole_tolerance),
        },
        "geometry": geometry,
        "overridden_sources": charge_data["overridden_sources"],
        "snapshot": "input_geometry",
    }


def charge_rejection_reasons(analysis: dict[str, Any], policy: str) -> list[str]:
    """Explain strict-policy failures for total, substrate AND film groups."""
    validate_charge_options(policy, 0.0, 0.0)
    if policy == "report":
        return []
    reasons = []
    groups = analysis.get("groups", {})
    for name in _GROUPS:
        group = groups.get(name, {})
        neutrality = group.get("neutrality", "unknown")
        if neutrality != "neutral":
            reasons.append(f"{name}: neutrality {neutrality}")
        polarity = group.get("polarity", "unknown")
        if policy == "require-nonpolar" and polarity != "nonpolar":
            reasons.append(f"{name}: polarity {polarity}")
    return reasons


def is_potentially_polar(structure: Any) -> tuple[bool, str]:
    """Composition-only hint that a polar termination is possible.

    Only multi-element compositions containing a polar-bond-forming anion
    (F, Cl, Br, I, O, S, N) are flagged; a single-element or anion-free
    composition returns ``(False, "")``.  The empty reason is not a
    nonpolarity determination -- it only means the composition carries no
    polar-termination warning.  Exact element symbols are matched, so Co,
    Ni and Fe are never flagged by substring accidents.
    """
    try:
        if hasattr(structure, "get_chemical_symbols"):
            elements = set(structure.get_chemical_symbols())
        else:
            elements = {element.symbol for element in structure.composition.elements}
    except (AttributeError, TypeError, ValueError):
        return False, "polarity unknown: cannot determine element composition"
    if len(elements) > 1 and elements.intersection({"F", "Cl", "Br", "I", "O", "S", "N"}):
        return (
            True,
            "composition-only hint: a polar termination is possible; actual polarity unknown",
        )
    return False, ""


def check_neutral_slabs(atoms: Any, slab_tags: list[int]) -> list[str]:
    """Return charged/unknown group problems without guessing oxidation states."""
    return charge_rejection_reasons(analyze_charge_polarity(atoms, slab_tags), "require-neutral")


def suggest_dipole_correction(atoms: Any) -> dict[str, Any]:
    """Suggest True/False/None only from the total-slab model; never change DFT settings."""
    total = analyze_charge_polarity(atoms)["groups"]["total"]
    polarity = total["polarity"]
    if polarity == "unknown":
        return {
            "dipole_correction_recommended": None,
            "reason": "polarity unknown: " + ", ".join(total["reasons"]),
        }
    return {
        "dipole_correction_recommended": polarity == "polar",
        "reason": f"total slab is {polarity} under the supplied charge model; calculator setup is unchanged",
    }


def tasker_classification(analysis: dict[str, Any]) -> int | None:
    """Tasker-type proxy derived from a supplied-charge analysis.

    Returns 3 when any group is polar or charged (an asymmetric board's
    normal dipole grows with thickness), 1 when every group is neutral and
    nonpolar, and ``None`` when the supplied charge model does not
    determine it.  This is bookkeeping over the supplied charges, not the
    full plane-resolved Tasker analysis, and it never guesses charges.
    """
    groups = analysis.get("groups", {})
    if not groups:
        return None
    if any(
        group.get("polarity") == "polar" or group.get("neutrality") == "charged"
        for group in groups.values()
    ):
        return 3
    if all(
        group.get("polarity") == "nonpolar" and group.get("neutrality") == "neutral"
        for group in groups.values()
    ):
        return 1
    return None


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


# Absolute tolerance (Å) below which a cell component counts as nothing
# when deciding whether a slab already satisfies the "+z normal" contract
# enforced by :func:`_orient_slab`.  Cell lengths are a few Å, so this is
# ~1e-10 relative -- far above float64 round-off, far below any physical
# or numerical quantity downstream.
_ORIENT_TOL = 1e-9


def _orient_slab(slab):
    """Return a slab whose surface normal is exactly ``+z``, ``a``/``b`` in ``xy``.

    ``SlabGenerator(primitive=True)`` returns cells whose in-plane vectors
    carry a z component for every surface but (001): fcc (111) comes back
    as ``a = (1.81, 0, -1.81)``, ``b = (0, 1.81, -1.81)``.  Every
    downstream ``[:2]`` slice then projects those vectors onto the xy
    plane, which silently shrinks the interface area by ``sqrt(3)``
    (3.2671 instead of 5.6587 Å²) and lets ``_assemble_interface`` rebuild
    a hexagonal cell as a square one.  Rotating once here turns that
    assumption into a fact for every consumer further down the chain
    instead of patching six separate slices.

    The rotation is applied to the orthogonal-c form so that ``c`` really
    is parallel to ``cross(a, b)`` -- clearing the in-plane part of a skew
    ``c`` would otherwise discard ~40 Å of real cell vector.

    A slab that already satisfies the contract skips the rotation and comes
    back with its coordinates untouched, which is what keeps the published
    (001) reference numbers pinned bit for bit.
    """
    from pymatgen.core import Structure
    from pymatgen.core.lattice import Lattice as _Lattice

    matrix = np.asarray(slab.lattice.matrix, dtype=np.float64)
    already_aligned = max(
        abs(float(matrix[0, 2])),
        abs(float(matrix[1, 2])),
        abs(float(matrix[2, 0])),
        abs(float(matrix[2, 1])),
    )
    # This tests the invariant itself -- whether a and b are level and c
    # carries no in-plane part -- not the Miller indices, and not any
    # guess about which surface is being built.  When the test passes the
    # rotation below is the identity, so skipping it changes nothing.  Do
    # not read this as a "(001) fast path": it fires for any already-level
    # cell from any source, and it must not be reused as an
    # orientation-recognition heuristic anywhere else.
    if already_aligned <= _ORIENT_TOL:
        # Copied rather than handed back as-is, so a caller cannot mutate
        # the generator's own slab through this result.  ``properties`` is
        # re-applied explicitly because pymatgen's ``copy()`` drops it --
        # which would silently strip the finite-slab provenance flag the
        # charge layer reads.
        aligned = slab.copy()
        aligned.properties.update(getattr(slab, "properties", {}) or {})
        return aligned

    orth = slab.get_orthogonal_c_slab()  # c perpendicular to a, b
    L = np.asarray(orth.lattice.matrix, dtype=np.float64)
    n = np.cross(L[0], L[1])
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    e1 = L[0] / np.linalg.norm(L[0])
    e2 = np.cross(n, e1)
    e2 = e2 / np.linalg.norm(e2)
    rot = np.vstack([e1, e2, n])  # rows are the new basis vectors, in old coordinates
    new_L = L @ rot.T
    new_L[2, :2] = 0.0  # defensive: drop any residual in-plane part of c
    coords = np.asarray(orth.cart_coords, dtype=np.float64) @ rot.T
    out = Structure(
        _Lattice(new_L),
        list(orth.species),
        coords,
        coords_are_cartesian=True,
        to_unit_cell=False,
        site_properties={key: list(values) for key, values in orth.site_properties.items()},
    )
    # Provenance comes from the *input*, not from ``orth``:
    # ``get_orthogonal_c_slab`` rebuilds its own structure and drops
    # ``properties`` on the floor, so reading them off it would silently
    # strip the ``ai_slab_generated`` flag the charge layer certifies
    # finite boards by.
    out.properties.update(getattr(slab, "properties", {}) or {})
    return out


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
