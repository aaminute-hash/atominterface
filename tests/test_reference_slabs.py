"""Tests for the isolated reference slabs behind an interface energy."""

from __future__ import annotations

import numpy as np
import pytest

from atominterface.builders.charge import SLAB_ID
from atominterface.builders.coherent import (
    CoherentInterfaceBuilder,
    CoherentInterfaceConfig,
    _get_terminations,
    _MatchInfo,
    reference_slabs,
)
from atominterface.thermo import slab_formula_units


def _fcc(species: str, a: float):
    """Conventional fcc cell, as a bulk input for the coherent builder."""
    try:
        from pymatgen.core import Lattice, Structure  # noqa: WPS433
    except ImportError:
        pytest.skip("pymatgen not installed")

    return Structure(
        Lattice.cubic(a),
        [species] * 4,
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    )


@pytest.fixture
def cu_bulk():
    return _fcc("Cu", 3.61)


@pytest.fixture
def ni_bulk():
    return _fcc("Ni", 3.52)


@pytest.fixture
def mgo_bulk():
    """Rocksalt bulk whose (001) terminations differ by species."""
    try:
        from pymatgen.core import Lattice, Structure  # noqa: WPS433
    except ImportError:
        pytest.skip("pymatgen not installed")

    return Structure(Lattice.cubic(4.21), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _config(**overrides):
    values = dict(
        substrate_miller=(0, 0, 1),
        film_miller=(0, 0, 1),
        max_strain=0.05,
        max_area=100.0,
        gap=2.0,
        min_thickness=6.0,
        vacuum_thickness=15.0,
        relax_terminations=False,
    )
    values.update(overrides)
    return CoherentInterfaceConfig(**values)


def _in_plane_area(atoms) -> float:
    return abs(float(np.linalg.det(np.asarray(atoms.cell.array)[:2, :2])))


def _in_plane_cell(atoms) -> np.ndarray:
    return np.asarray(atoms.cell.array)[:2, :2]


def _layer_count(atoms) -> int:
    return len(np.unique(np.round(atoms.get_positions()[:, 2], 6)))


def _frac_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Fractional-coordinate distance modulo one lattice vector."""
    return np.abs(((left - right + 0.5) % 1.0) - 0.5)


def test_reference_slabs_without_match_are_plain_terminations(cu_bulk, ni_bulk):
    """match=None is the legacy stacking: 1x1 boards, film left unstrained."""
    config = _config()
    sub_ref, film_ref = reference_slabs(cu_bulk, ni_bulk, config)

    # Both boards are ASE Atoms, cut by the builder's own termination path.
    expected_sub = _get_terminations(
        cu_bulk, config.substrate_miller, config.min_thickness, config.vacuum_thickness
    )[0][1]
    expected_film = _get_terminations(
        ni_bulk, config.film_miller, config.min_thickness, config.vacuum_thickness
    )[0][1]

    assert hasattr(sub_ref, "get_positions") and hasattr(film_ref, "get_positions")
    assert np.allclose(_in_plane_cell(sub_ref), np.asarray(expected_sub.lattice.matrix)[:2, :2])
    assert np.allclose(_in_plane_cell(film_ref), np.asarray(expected_film.lattice.matrix)[:2, :2])

    # SlabGenerator semantics for a conventional fcc cell cut at (001):
    # the primitive surface cell (area a^2/2) with one atom per layer,
    # min_thickness=6.0 buying 4 layers at d_001 = a/2 = 1.805 A.
    assert _layer_count(sub_ref) == 4
    assert len(sub_ref) == 4
    assert _in_plane_area(sub_ref) == pytest.approx(3.61**2 / 2)
    assert _in_plane_area(film_ref) == pytest.approx(3.52**2 / 2)
    # The film is NOT deformed onto the substrate without a match.
    assert not np.allclose(_in_plane_cell(sub_ref), _in_plane_cell(film_ref))


@pytest.mark.parametrize(
    ("sub_supercell", "film_supercell"),
    [
        (((2, 0), (0, 1)), ((2, 0), (0, 1))),
        (((2, 0), (0, 1)), ((1, 0), (0, 2))),
    ],
)
def test_reference_slabs_replicate_and_stay_coherent(
    cu_bulk, ni_bulk, sub_supercell, film_supercell
):
    """Same supercell, same thickness: the film is strained onto the substrate cell."""
    config = _config()
    plain_sub, plain_film = reference_slabs(cu_bulk, ni_bulk, config)
    match = _MatchInfo(
        match_area=2 * _in_plane_area(plain_sub),
        substrate_supercell=sub_supercell,
        film_supercell=film_supercell,
        strain=0.01,
        shear_strain=0.0,
    )

    sub_ref, film_ref = reference_slabs(cu_bulk, ni_bulk, config, match)

    # The substrate board carries the supercell and is never strained.
    assert _in_plane_area(sub_ref) == pytest.approx(2 * _in_plane_area(plain_sub))
    assert len(sub_ref) == 2 * len(plain_sub)

    # The film board is replicated by its own supercell and then strained,
    # so its in-plane cell equals the substrate cell exactly -- that is
    # what "coherent" means, even for a cross-shaped 2x1 / 1x2 pair.
    assert len(film_ref) == 2 * len(plain_film)
    assert np.allclose(_in_plane_cell(film_ref), _in_plane_cell(sub_ref))
    assert _in_plane_area(film_ref) == pytest.approx(_in_plane_area(sub_ref))


def test_reference_slabs_match_the_assembled_interface(cu_bulk, ni_bulk):
    """The film side of a real interface IS film_ref, up to the rigid stacking."""
    config = _config()
    builder = CoherentInterfaceBuilder(substrate=cu_bulk, film=ni_bulk, config=config)
    candidates = builder.build_all()
    assert candidates
    candidate = candidates[0]

    metadata = candidate.metadata
    match = _MatchInfo(
        match_area=metadata["match_area_ang2"],
        substrate_supercell=metadata["substrate_supercell"],
        film_supercell=metadata["film_supercell"],
        strain=metadata["mismatch_stretch"],
        shear_strain=metadata["mismatch_shear"],
    )
    sub_ref, film_ref = reference_slabs(
        cu_bulk,
        ni_bulk,
        config,
        match,
        substrate_termination=int(candidate.substrate_termination),
        film_termination=int(candidate.film_termination),
    )

    tags = candidate.atoms.arrays[SLAB_ID]
    film_side = candidate.atoms[tags == 1]
    sub_side = candidate.atoms[tags == 0]

    assert len(film_side) == len(film_ref)
    assert film_side.get_chemical_symbols() == film_ref.get_chemical_symbols()

    # Stacking sequence preserved: fractional in-plane coordinates agree
    # (modulo one lattice vector, since an atom may sit on the boundary),
    # and the strained cartesian in-plane coordinates agree exactly.
    film_frac = film_side.get_scaled_positions()
    ref_frac = film_ref.get_scaled_positions()
    assert np.allclose(_frac_delta(film_frac[:, :2], ref_frac[:, :2]), 0.0, atol=1e-8)
    film_xy = film_side.get_positions()[:, :2]
    ref_xy = film_ref.get_positions()[:, :2]
    assert np.allclose(film_xy, ref_xy, atol=1e-8)

    # Same in-plane lattice, same z ordering up to the rigid stacking shift.
    assert np.allclose(_in_plane_cell(film_side), _in_plane_cell(film_ref), atol=1e-8)
    film_z = film_side.get_positions()[:, 2]
    ref_z = film_ref.get_positions()[:, 2]
    assert np.allclose(film_z - film_z.min(), ref_z - ref_z.min(), atol=1e-8)

    # The substrate side is the unstrained reference itself, only stacked.
    assert len(sub_side) == len(sub_ref)
    assert np.allclose(sub_side.get_positions(), sub_ref.get_positions(), atol=1e-8)


def test_reference_slabs_symmetric_are_mirror_boards(mgo_bulk):
    """symmetric_slabs=True references a board with one termination per surface."""
    config = _config(symmetric_slabs=True, min_thickness=8.0)
    sub_ref, film_ref = reference_slabs(mgo_bulk, mgo_bulk, config)

    for board in (sub_ref, film_ref):
        symbols = [
            symbol
            for _, symbol in sorted(
                zip(board.get_positions()[:, 2], board.get_chemical_symbols(), strict=True)
            )
        ]
        # Same termination on both surfaces: the species sequence read
        # along z is a palindrome, and so are the layer positions.
        assert symbols == symbols[::-1]
        z = np.sort(board.get_positions()[:, 2])
        center = 0.5 * (z[0] + z[-1])
        assert np.allclose(z - center, -(z[::-1] - center))


@pytest.mark.parametrize("sub_supercell", [None, ((2, 0), (0, 1))])
def test_reference_slabs_formula_units_track_the_thickness(cu_bulk, ni_bulk, sub_supercell):
    """An elemental fcc board is a whole number of bulk formula units."""
    config = _config()
    match = None
    if sub_supercell is not None:
        match = _MatchInfo(
            match_area=0.0,
            substrate_supercell=sub_supercell,
            film_supercell=sub_supercell,
            strain=0.0,
            shear_strain=0.0,
        )
    sub_ref, film_ref = reference_slabs(cu_bulk, ni_bulk, config, match)

    # The bulk reference is the supplied conventional cell, so an
    # elemental fcc board is a whole number of bulk formula units that
    # scales exactly with the replicated atom count.
    assert slab_formula_units(sub_ref, cu_bulk) == pytest.approx(len(sub_ref) / len(cu_bulk))
    assert slab_formula_units(film_ref, ni_bulk) == pytest.approx(len(film_ref) / len(ni_bulk))
    assert float(slab_formula_units(sub_ref, cu_bulk)).is_integer()


@pytest.mark.parametrize("side", ["substrate_termination", "film_termination"])
def test_reference_slabs_rejects_unknown_termination(cu_bulk, ni_bulk, side):
    config = _config()
    with pytest.raises(ValueError, match="available indices"):
        reference_slabs(cu_bulk, ni_bulk, config, None, **{side: 5})
