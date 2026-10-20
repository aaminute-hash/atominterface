"""Termination thermodynamics: bulk-referenced energy and deterministic ranking.

The vacuum is a zero-energy reference: it carries no matter, so a slab's
termination energy is measured against the bulk it was cut from,

    E_term = (E_slab - n * E_bulk) / (n_surfaces * A)

with ``n`` the stoichiometry-weighted number of bulk formula units inside
the slab and ``n_surfaces`` the number of distinct surfaces the slab
exposes (2 for a symmetric board, 1 for an asymmetric one).  No chemical
potential scanning happens here; a non-stoichiometric board is reported
with a warning rather than silently referenced to the bulk.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from numbers import Real
from typing import Any

from atominterface.physics import EV_A2_TO_J_M2

__all__ = [
    "TerminationRecord",
    "rank_terminations",
    "slab_formula_units",
    "termination_energy",
]

# Unit -> multiplicative factor from eV/A^2, mirroring ``interface_energy``.
_EV_A2_TO_UNIT = {"eV_A2": 1.0, "J_m2": EV_A2_TO_J_M2}


@dataclass(frozen=True)
class TerminationRecord:
    """One termination of one surface, ranked once its energies are known.

    Energy fields are optional: geometry-only records rank after every
    record that carries a termination energy, in their original order.
    """

    label: str
    formula: str
    stoichiometric: bool
    tasker_type: int | None
    nominal_charge: float | None
    area: float
    n_atoms: int
    slab_energy: float | None = None
    bulk_energy: float | None = None
    n_formula_units: float | None = None
    n_surfaces: int = 2
    stretch: float = 0.0
    shear: float = 0.0
    symmetric: bool = False

    @property
    def termination_energy_ev_a2(self) -> float | None:
        """Termination energy in eV/A^2, or ``None`` while energies are missing."""
        if self.slab_energy is None or self.bulk_energy is None or self.n_formula_units is None:
            return None
        return termination_energy(
            slab_energy=self.slab_energy,
            bulk_energy=self.bulk_energy,
            n_formula_units=self.n_formula_units,
            area=self.area,
            n_surfaces=self.n_surfaces,
        )


def _finite_real(value: Any, name: str) -> float:
    """Reject booleans, non-numbers and non-finite values with one message."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, not {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def termination_energy(
    *,
    slab_energy: float,
    bulk_energy: float,
    n_formula_units: float,
    area: float,
    n_surfaces: int = 2,
    unit: str = "eV_A2",
) -> float:
    """Termination energy per unit area against the bulk reference.

    ``E_term = (E_slab - n_formula_units * E_bulk) / (n_surfaces * area)``.
    A fractional ``n_formula_units`` (non-stoichiometric board) warns but is
    still referenced to the bulk; a non-positive ``area`` or an unknown
    ``unit`` raises ``ValueError``.
    """
    if unit not in _EV_A2_TO_UNIT:
        raise ValueError(f"unsupported unit {unit!r}")
    if isinstance(n_surfaces, bool) or not isinstance(n_surfaces, int) or n_surfaces < 1:
        raise ValueError("n_surfaces must be a positive integer")
    slab = _finite_real(slab_energy, "slab_energy")
    bulk = _finite_real(bulk_energy, "bulk_energy")
    count = _finite_real(n_formula_units, "n_formula_units")
    surface_area = _finite_real(area, "area")
    if surface_area <= 0:
        raise ValueError("termination area must be positive")
    if abs(count - round(count)) > 1e-9:
        warnings.warn(
            f"non-stoichiometric slab: {count} formula units is fractional, so the bulk "
            "reference is an approximation",
            stacklevel=2,
        )
    energy = (slab - count * bulk) / (n_surfaces * surface_area)
    return energy * _EV_A2_TO_UNIT[unit]


def _element_counts(structure: Any) -> dict[str, int]:
    """Element -> atom count for an ASE Atoms or a pymatgen Structure."""
    if hasattr(structure, "get_chemical_symbols"):
        symbols = structure.get_chemical_symbols()
    else:
        symbols = [site.specie.symbol for site in structure]
    counts: dict[str, int] = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def slab_formula_units(structure: Any, bulk_structure: Any) -> float:
    """Stoichiometry-weighted bulk formula units contained in ``structure``.

    Each element contributes ``n_slab(alpha) / n_bulk_fu(alpha)`` and the
    contributions are averaged, so a stoichiometric multiple of the bulk
    cell returns an integer while a non-stoichiometric board (e.g. a
    Na-terminated symmetric rocksalt (111) slab) returns a fraction.
    """
    slab_counts = _element_counts(structure)
    bulk_counts = _element_counts(bulk_structure)
    if not bulk_counts:
        raise ValueError("bulk structure contains no atoms")
    missing = sorted(set(slab_counts) - set(bulk_counts))
    if missing:
        raise ValueError(
            "slab contains elements absent from the bulk reference: " + ", ".join(missing)
        )
    shares = [slab_counts[symbol] / count for symbol, count in bulk_counts.items()]
    return float(sum(shares) / len(shares))


def rank_terminations(records: list[TerminationRecord]) -> list[TerminationRecord]:
    """Order terminations by energy, then geometry, then label.

    Records carrying a termination energy come first, ascending, with
    ``(stretch, shear, -area, label)`` as deterministic tie-breaks.  Records
    without energies keep their input order at the end.
    """
    ranked: list[tuple[float, TerminationRecord]] = []
    pending: list[TerminationRecord] = []
    for record in records:
        energy = record.termination_energy_ev_a2
        if energy is None:
            pending.append(record)
        else:
            ranked.append((energy, record))
    ranked.sort(
        key=lambda item: (item[0], item[1].stretch, item[1].shear, -item[1].area, item[1].label)
    )
    return [record for _, record in ranked] + pending
