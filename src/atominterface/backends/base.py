"""Backend protocol and capability declaration.

A backend wraps an ASE-compatible calculator and declares what it
supports (elements, stress, units, license).  This is the contract
that lets the guard layer stay model-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Capability:
    """What a calculator backend claims to support."""

    name: str
    elements: list[str] = field(default_factory=list)
    supports_stress: bool = False
    supports_forces: bool = True
    supports_uncertainty: bool = False
    energy_unit: str = "eV"
    force_unit: str = "eV/A"
    stress_unit: str = "eV/A^3"
    license: str = "unknown"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "elements": sorted(set(self.elements)),
            "supports_stress": self.supports_stress,
            "supports_forces": self.supports_forces,
            "supports_uncertainty": self.supports_uncertainty,
            "energy_unit": self.energy_unit,
            "force_unit": self.force_unit,
            "stress_unit": self.stress_unit,
            "license": self.license,
            "notes": self.notes,
        }


@runtime_checkable
class CalculatorLike(Protocol):
    """ASE calculator duck-type (subset we rely on)."""

    def get_potential_energy(self, atoms) -> float:  # noqa: D401
        ...

    def get_forces(self, atoms):  # noqa: D401
        ...

    def get_stress(self, atoms):  # noqa: D401
        ...
