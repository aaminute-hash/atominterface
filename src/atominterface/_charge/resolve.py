from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
from ase.data import atomic_numbers

from atominterface._charge._constants import _POLICIES, _SOURCES


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
