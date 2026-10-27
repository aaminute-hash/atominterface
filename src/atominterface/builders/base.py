"""Builder protocol and the Candidate data model.

A Candidate is the single currency of the builder layer: every builder
takes two bulk structures plus a config, and yields a list of
Candidate objects.  Relaxers, thermo routines and exporters all
consume Candidates -- never raw ASE Atoms.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class BuilderConfig:
    """Shared knobs for every interface builder.

    Parameters
    ----------
    max_strain
        Absolute strain tolerance.  Candidates whose lattice mismatch
        exceeds this are dropped **before** structure generation, so
        the user is never handed a geometrically nonsense interface.
    max_area_ratio
        ``film_area / substrate_area`` window.  Values outside are
        rejected.  Defaults to (0.8, 1.25).
    gap
        Vacuum / separation along the interface normal in Angstrom.
    min_thickness
        Minimum slab thickness in Angstrom for **both** sides.  Thin
        slabs are the #1 source of fake interface energies.
    max_candidates
        Hard cap on returned candidates.  Builder enumerations grow
        combinatorially with terminations and matching cells.
    vacuum_dir
        Which cell vector is normal to the interface (0/1/2).  The
        builder will reorient the cell so this axis is the third one.
    extra
        Builder-specific free-form settings.
    """

    max_strain: float = 0.05
    max_area_ratio: tuple[float, float] = (0.8, 1.25)
    gap: float = 2.0
    min_thickness: float = 6.0
    max_candidates: int = 20
    vacuum_dir: int = 2
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Cross-field validation deferred to runtime.

        We deliberately do **not** raise in ``__post_init__``: a config
        object may be constructed programmatically with invalid values
        and then inspected via :meth:`validate` without forcing the
        caller to catch exceptions.  Builders call this method before
        doing any real work.
        """
        errors: list[str] = []
        if not 0.0 <= self.max_strain <= 1.0:
            errors.append("max_strain must be in [0, 1]")
        lo, hi = self.max_area_ratio
        if not (0 < lo <= hi):
            errors.append("max_area_ratio must satisfy 0 < lo <= hi")
        if self.gap < 0:
            errors.append("gap must be non-negative")
        if self.min_thickness < 0:
            errors.append("min_thickness must be non-negative")
        if self.max_candidates < 1:
            errors.append("max_candidates must be >= 1")
        return errors


@dataclass
class Candidate:
    """One interface candidate plus the provenance needed to judge it.

    ``atoms`` is guaranteed to be the ASE view, with ``atoms.info``
    carrying the scalar fields below.  Expensive properties (interface
    energy, work of separation) are **not** computed here: builders
    build, relaxers relax, thermo computes.
    """

    atoms: Any  # ase.Atoms; type left loose to avoid hard ASE dep in stubs
    interface_id: str
    builder_name: str
    substrate_formula: str
    film_formula: str
    substrate_miller: tuple[int, int, int]
    film_miller: tuple[int, int, int]
    substrate_termination: str = "bulk"
    film_termination: str = "bulk"
    strain_substrate: float = 0.0
    strain_film: float = 0.0
    area: float = 0.0
    n_substrate: int = 0
    n_film: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    # Bilateral area bookkeeping.  ``area`` stays the single scalar the
    # rest of the stack already consumes; these record *which* side that
    # scalar came from and whether the two sides actually agree.  NaN is
    # the "not supplied" sentinel resolved in ``__post_init__``: an
    # unknown side falls back to ``area`` rather than to 0, which would
    # silently claim a degenerate cell.
    area_substrate: float = float("nan")
    area_film: float = float("nan")
    area_consistent: bool = True
    area_policy: str = "strict"

    def __post_init__(self) -> None:
        """Fall back to ``area`` for either side when it was not given.

        Callers that build a Candidate by hand still pass only ``area``;
        they must not be handed a NaN (or a misleading 0.0) for a side
        whose area was never measured separately.
        """
        if math.isnan(self.area_substrate):
            self.area_substrate = float(self.area)
        if math.isnan(self.area_film):
            self.area_film = float(self.area)

    def to_atoms(self) -> Any:
        """Return the ASE Atoms view, with info dict populated."""
        from ase.atoms import Atoms

        if not isinstance(self.atoms, Atoms):
            raise TypeError(f"Candidate {self.interface_id} atoms is not an ASE Atoms instance")
        info = dict(self.atoms.info)
        info.update(
            {
                "interface_id": self.interface_id,
                "builder": self.builder_name,
                "substrate": self.substrate_formula,
                "film": self.film_formula,
                "sub_miller": list(self.substrate_miller),
                "film_miller": list(self.film_miller),
                "sub_termination": self.substrate_termination,
                "film_termination": self.film_termination,
                "strain_sub": self.strain_substrate,
                "strain_film": self.strain_film,
                "interface_area": self.area,
            },
        )
        # Charge diagnostics are a construction-time snapshot, not a live
        # validation of coordinates changed by a subsequent relaxation.
        for key in ("charge_analysis", "charge_geometry"):
            if key in self.metadata:
                info[key] = deepcopy(self.metadata[key])
        info["n_substrate"] = self.n_substrate
        info["n_film"] = self.n_film
        self.atoms.info = info
        return self.atoms

    @property
    def total_atoms(self) -> int:
        return int(len(self.atoms))

    @property
    def mean_strain(self) -> float:
        return float(np.mean([self.strain_substrate, self.strain_film]))


@runtime_checkable
class InterfaceBuilder(Protocol):
    """Minimal contract every builder must satisfy."""

    name: str

    def build(self, config: BuilderConfig | None = None) -> list[Candidate]: ...
