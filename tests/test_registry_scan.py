"""``registry="scan"``: one candidate per distinguishable site, capped.

These tests pin the second stage of the ``registry`` field: the explicit
enumeration of every high-symmetry in-plane site a surface resolves, the
one-candidate-per-site build it drives, the ``max_registry_sites`` cap,
and the fact that only an explicit scan retires the non-(001) warning.

No MLIP is involved: everything here is decided during construction, the
same way ``test_registry_field.py`` keeps its placement claims cheap.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from atominterface._coherent.placement import _wrap_frac, enumerate_registry_sites
from atominterface.builders.coherent import (
    CoherentInterfaceBuilder,
    CoherentInterfaceConfig,
    _get_terminations,
)


def _fcc(species: str, a: float) -> Structure:
    """Conventional fcc cell, as a bulk input for slab generation."""
    coords: list[list[float]] = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    return Structure(Lattice.cubic(a), [species] * 4, coords)


def _surface(miller: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(sub_coords, sub_2d) of the first Cu termination at ``miller``.

    ``_get_terminations`` already routes the generated slab through
    ``_orient_slab``, so the in-plane 2x2 block is exact and the surface
    normal really is the cartesian z axis -- the same board the builder
    hands :func:`enumerate_registry_sites`.
    """
    terms = _get_terminations(_fcc("Cu", 3.615), miller, 14.4, 15.0)
    slab = terms[0][1]
    sub_2d = np.asarray(slab.lattice.matrix, dtype=np.float64)[:2, :2]
    sub_coords = np.asarray(slab.cart_coords, dtype=np.float64)
    return sub_coords, sub_2d


def _builder(miller: tuple[int, int, int], **overrides) -> CoherentInterfaceBuilder:
    """Cu/Ni builder at ``miller`` with a single termination and match."""
    config = CoherentInterfaceConfig(
        substrate_miller=miller,
        film_miller=miller,
        min_thickness=14.4,
        max_area=12.0,
        relax_terminations=False,
        **overrides,
    )
    return CoherentInterfaceBuilder(
        substrate=_fcc("Cu", 3.615), film=_fcc("Ni", 3.524), config=config
    )


def test_enumerate_111_gives_three_sites():
    """A close-packed surface resolves exactly three high-symmetry sites."""
    sub_coords, sub_2d = _surface((1, 1, 1))
    sites = enumerate_registry_sites(sub_coords, sub_2d)
    assert len(sites) == 3
    assert len(set(sites)) == 3
    for u, v in sites:
        assert 0.0 <= u < 1.0
        assert 0.0 <= v < 1.0


def test_enumerate_110_gives_four_sites():
    """(110) resolves the four half-cell sites of a two-sublattice surface."""
    sub_coords, sub_2d = _surface((1, 1, 0))
    sites = enumerate_registry_sites(sub_coords, sub_2d)
    assert len(sites) == 4
    normalized = {
        tuple(np.round(_wrap_frac(np.asarray(site, dtype=np.float64)), 6)) for site in sites
    }
    assert normalized == {(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)}


def test_scan_builds_one_candidate_per_site():
    """``registry="scan"`` emits one candidate for each (001) site."""
    candidates = _builder((0, 0, 1), registry="scan").build_all()
    assert len(candidates) == 4
    assert all(candidate.metadata["registry_requested"] == "scan" for candidate in candidates)
    targets = [tuple(candidate.metadata["registry_target_frac"]) for candidate in candidates]
    assert len(set(targets)) == 4


def test_max_registry_sites_caps_count():
    """The cap truncates the site enumeration before any candidate is built."""
    candidates = _builder((0, 0, 1), registry="scan", max_registry_sites=2).build_all()
    assert 0 < len(candidates) <= 2


def test_scan_suppresses_non_001_warning():
    """Only ``registry="scan"`` retires the non-(001) orientation warning."""

    def _orientation_warnings(
        miller: tuple[int, int, int], registry: str
    ) -> list[warnings.WarningMessage]:
        builder = _builder(miller, registry=registry)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.build_all()
        return [item for item in caught if "[orientation]" in str(item.message)]

    assert _orientation_warnings((1, 1, 0), "scan") == []
    assert _orientation_warnings((1, 1, 0), "auto") != []


@pytest.mark.parametrize("value", [0, -1])
def test_max_registry_sites_validation(value):
    """A non-positive cap is a validation error, not a silent no-scan."""
    errors = CoherentInterfaceConfig(max_registry_sites=value).validate()
    assert errors
