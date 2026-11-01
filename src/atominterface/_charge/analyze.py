from __future__ import annotations

from collections.abc import Mapping
from math import fsum
from typing import Any

import numpy as np

from atominterface._charge._constants import (
    _DEFAULT_AXIS,
    _DEFAULT_FINITE,
    _DEFAULT_VACUUM,
    ANALYSIS_CHARGE,
    CHARGE_KNOWN,
    CHARGE_SOURCE,
    SLAB_ID,
    _GeometryDefault,
)
from atominterface._charge.resolve import (
    _real_number,
    _vector,
    resolve_charge_data,
    validate_charge_options,
)


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
