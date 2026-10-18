"""Tests for the coherent interface builder wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from atominterface.builders.coherent import (
    CoherentInterfaceBuilder,
    CoherentInterfaceConfig,
    _assemble_interface,
    _fallback_matches,
    _lattice_area,
    _make_slabs,
    _match_lattices,
    _MatchInfo,
    _multiplicity,
    _scale_in_plane,
    _supercell_transform,
    _to_candidate,
)
from atominterface.builders.registry import builder_registry


@pytest.fixture
def cubic_structure():
    """Primitive cubic structure (e.g. a generic rocksalt)."""
    try:
        from pymatgen.core import Lattice, Structure  # noqa: WPS433
    except ImportError:
        pytest.skip("pymatgen not installed")

    lattice = Lattice.cubic(4.0)
    return Structure(lattice, ["Li", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.fixture
def fcc_structure():
    """FCC structure (e.g. Al)."""
    try:
        from pymatgen.core import Lattice, Structure  # noqa: WPS433
    except ImportError:
        pytest.skip("pymatgen not installed")

    lattice = Lattice.cubic(4.05)
    return Structure(
        lattice,
        ["Al", "Al", "Al", "Al"],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )


@pytest.fixture
def bcc_structure():
    """BCC structure (e.g. Li metal)."""
    try:
        from pymatgen.core import Lattice, Structure  # noqa: WPS433
    except ImportError:
        pytest.skip("pymatgen not installed")

    lattice = Lattice.cubic(3.51)
    return Structure(lattice, ["Li", "Li"], [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.fixture
def dummy_match():
    return _MatchInfo(
        match_area=64.0,
        substrate_supercell=(2, 2),
        film_supercell=(2, 2),
        strain=0.012,
        shear_strain=0.008,
    )


def test_lattice_area():
    lattice = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 10.0]])
    assert abs(_lattice_area(lattice) - 16.0) < 1e-12


def test_supercell_transform_keeps_off_diagonal_entries():
    """A rotated supercell must not be flattened to its diagonal."""
    mat = np.array([[1.0, -1.0], [1.0, 1.0]])
    assert _supercell_transform(mat) == ((1, -1), (1, 1))
    assert _multiplicity(((1, -1), (1, 1))) == 2


def test_supercell_transform_rounds_float_noise():
    mat = np.array([[2.0000001, 0.0], [0.0, 2.9999998]])
    assert _supercell_transform(mat) == ((2, 0), (0, 3))


def test_assemble_interface_manual_fallback(cubic_structure):
    """Manual stacking must preserve both slabs and shift the film."""
    sub = cubic_structure
    film = cubic_structure.copy()

    match = _MatchInfo(
        match_area=16.0,
        substrate_supercell=(1, 1),
        film_supercell=(1, 1),
        strain=0.0,
        shear_strain=0.0,
    )

    with patch(
        "atominterface.builders.coherent._PYMTG_COHERENT_BUILDER",
        side_effect=RuntimeError("disabled"),
    ):
        interface = _assemble_interface(sub, film, match, CoherentInterfaceConfig(gap=2.5))

    assert len(interface) == len(sub) + len(film)

    sub_z = np.asarray([s.coords[2] for s in sub.sites])
    interface_z = np.asarray([s.coords[2] for s in interface.sites])

    # Film atoms must sit above the substrate atoms.
    assert interface_z.max() > sub_z.max() + 2.5 - 1e-6


def test_fallback_matches_respects_strain_cutoff(cubic_structure):
    matches = _fallback_matches(
        cubic_structure.lattice.matrix,
        cubic_structure.lattice.matrix,
        CoherentInterfaceConfig(max_strain=0.01, max_area=1000.0),
    )
    assert matches
    assert all(m.strain <= 0.01 for m in matches)
    assert all(m.shear_strain <= 0.01 for m in matches)


def test_fallback_matches_rejects_large_area(cubic_structure):
    matches = _fallback_matches(
        cubic_structure.lattice.matrix,
        cubic_structure.lattice.matrix,
        CoherentInterfaceConfig(max_strain=1.0, max_area=30.0),
    )
    assert all(m.match_area <= 30.0 for m in matches)


def test_to_candidate_attaches_metadata(dummy_match, cubic_structure):
    candidate = _to_candidate(
        interface=cubic_structure,
        substrate_slab=cubic_structure,
        film_slab=cubic_structure,
        match=dummy_match,
        sub_term_idx=1,
        film_term_idx=2,
        config=CoherentInterfaceConfig(
            substrate_miller=(0, 0, 1),
            film_miller=(1, 1, 1),
            gap=2.0,
            vacuum_thickness=15.0,
        ),
    )

    assert candidate.builder_name == "coherent"
    assert candidate.substrate_miller == (0, 0, 1)
    assert candidate.film_miller == (1, 1, 1)
    assert candidate.substrate_termination == "1"
    assert candidate.film_termination == "2"
    assert candidate.metadata["substrate_supercell"] == (2, 2)
    assert candidate.metadata["film_supercell"] == (2, 2)
    assert abs(candidate.metadata["stretch_strain"] - 0.012) < 1e-12
    assert candidate.metadata["gap_ang"] == 2.0
    assert candidate.metadata["vacuum_ang"] == 15.0


def test_to_candidate_interface_id_encoding(dummy_match, cubic_structure):
    candidate = _to_candidate(
        interface=cubic_structure,
        substrate_slab=cubic_structure,
        film_slab=cubic_structure,
        match=dummy_match,
        sub_term_idx=3,
        film_term_idx=7,
        config=CoherentInterfaceConfig(
            substrate_miller=(1, 0, 0),
            film_miller=(0, 1, 1),
        ),
    )
    assert candidate.interface_id.startswith("coherent:")
    assert "(1, 0, 0)-(0, 1, 1):3-7" in candidate.interface_id


def test_to_candidate_energy_per_area_conversion(cubic_structure):
    lattice = np.asarray(cubic_structure.lattice.matrix)
    area = float(np.linalg.norm(np.cross(lattice[0], lattice[1])))

    # Patch a fake energy onto the structure so the conversion path
    # is exercised without invoking a calculator.
    interface = cubic_structure.copy()
    interface.energy = 3.2

    candidate = _to_candidate(
        interface=interface,
        substrate_slab=cubic_structure,
        film_slab=cubic_structure,
        match=_MatchInfo(
            match_area=area,
            substrate_supercell=(1, 1),
            film_supercell=(1, 1),
            strain=0.0,
            shear_strain=0.0,
        ),
        sub_term_idx=0,
        film_term_idx=0,
        config=CoherentInterfaceConfig(),
    )

    expected = 3.2 / area
    assert abs(candidate.metadata["energy_per_area_ev_a2"] - expected) < 1e-12
    assert candidate.metadata["energy_per_area_jm2"] == pytest.approx(
        expected * 16.02176634, rel=1e-6
    )


def test_validate_rejects_missing_structures():
    builder = CoherentInterfaceBuilder(substrate=None, film=None)
    errors = builder.validate()
    assert any("substrate" in e for e in errors)


def test_validate_rejects_zero_miller():
    builder = CoherentInterfaceBuilder(
        substrate=MagicMock(lattice=MagicMock(matrix=np.eye(3))),
        film=MagicMock(lattice=MagicMock(matrix=np.eye(3))),
        config=CoherentInterfaceConfig(substrate_miller=(0, 0, 0)),
    )
    errors = builder.validate()
    assert any("Miller" in e for e in errors)


def test_validate_rejects_negative_strain():
    builder = CoherentInterfaceBuilder(
        substrate=MagicMock(lattice=MagicMock(matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        film=MagicMock(lattice=MagicMock(matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        config=CoherentInterfaceConfig(max_strain=-0.1),
    )
    errors = builder.validate()
    assert any("max_strain" in e for e in errors)


def test_validate_rejects_strain_above_one():
    builder = CoherentInterfaceBuilder(
        substrate=MagicMock(lattice=MagicMock(matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        film=MagicMock(lattice=MagicMock(matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        config=CoherentInterfaceConfig(max_strain=1.2),
    )
    errors = builder.validate()
    assert any("max_strain" in e for e in errors)


def test_validate_rejects_non_positive_volume():
    bad = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    builder = CoherentInterfaceBuilder(
        substrate=MagicMock(lattice=MagicMock(matrix=bad)),
        film=MagicMock(lattice=MagicMock(matrix=np.eye(3))),
    )
    errors = builder.validate()
    assert any("volume" in e for e in errors)


def test_builder_name():
    builder = CoherentInterfaceBuilder(substrate=MagicMock(), film=MagicMock())
    assert builder.name == "coherent"


def test_build_all_returns_candidates():
    """Two fcc cells 1.2 % apart form a genuine coherent interface."""
    builder = CoherentInterfaceBuilder(
        substrate=_fcc("Al", 4.05),
        film=_fcc("Al", 4.00),
        config=CoherentInterfaceConfig(
            substrate_miller=(0, 0, 1),
            film_miller=(0, 0, 1),
            max_strain=0.05,
            gap=2.0,
            relax_terminations=False,
        ),
    )

    candidates = builder.build_all()
    assert candidates
    assert all(c.area > 0 for c in candidates)
    assert all(c.builder_name == "coherent" for c in candidates)
    assert all(c.total_atoms > 0 for c in candidates)
    # Candidate.atoms must be the ASE view, not the pymatgen Structure.
    assert all(hasattr(c.atoms, "get_positions") for c in candidates)


def test_build_all_rejects_an_incoherent_pair(fcc_structure, bcc_structure):
    """fcc(001) is a 60-degree rhombus, bcc(001) a square: no coherent match.

    Asserted at the default 5 % tolerance, which no supercell of this pair
    can satisfy -- the in-plane shear alone is ~30 degrees.  (At a loose
    30 % tolerance the diagonal fallback *will* hand back something, and
    which path gets there depends on the pymatgen build, so the cut-off is
    the part worth pinning.)
    """
    builder = CoherentInterfaceBuilder(
        substrate=fcc_structure,
        film=bcc_structure,
        config=CoherentInterfaceConfig(relax_terminations=False),
    )

    assert builder.build_all() == []


def test_build_all_filters_by_strain(fcc_structure, bcc_structure):
    """A tight strain cutoff should produce an empty or small list."""
    config = CoherentInterfaceConfig(
        substrate_miller=(0, 0, 1),
        film_miller=(0, 0, 1),
        max_strain=0.001,
        max_area=100.0,
        gap=2.0,
        relax_terminations=False,
    )
    builder = CoherentInterfaceBuilder(
        substrate=fcc_structure,
        film=bcc_structure,
        config=config,
    )

    candidates = builder.build_all()
    assert all(max(c.strain_substrate, c.strain_film) <= 0.001 for c in candidates)


def test_build_all_relax_terminations_off(fcc_structure):
    config = CoherentInterfaceConfig(
        substrate_miller=(0, 0, 1),
        film_miller=(0, 0, 1),
        max_strain=0.30,
        max_area=100.0,
        relax_terminations=False,
    )
    builder = CoherentInterfaceBuilder(
        substrate=fcc_structure,
        film=fcc_structure,
        config=config,
    )

    # With termination relaxation off we should only see the first
    # termination, so the candidate set is driven by matches alone.
    candidates = builder.build_all()
    term_pairs = {(c.substrate_termination, c.film_termination) for c in candidates}
    assert ("0", "0") in term_pairs


def test_list_terminations(fcc_structure):
    builder = CoherentInterfaceBuilder(
        substrate=fcc_structure,
        film=fcc_structure,
    )
    terms = builder.list_terminations()
    assert "substrate" in terms
    assert "film" in terms
    assert all(isinstance(idx, int) for idx, _ in terms["substrate"])
    assert all(isinstance(idx, int) for idx, _ in terms["film"])


def test_export_candidates_writes_files(fcc_structure, tmp_path):
    candidate = _to_candidate(
        interface=fcc_structure,
        substrate_slab=fcc_structure,
        film_slab=fcc_structure,
        match=_MatchInfo(
            match_area=16.0,
            substrate_supercell=(1, 1),
            film_supercell=(1, 1),
            strain=0.01,
            shear_strain=0.005,
        ),
        sub_term_idx=0,
        film_term_idx=0,
        config=CoherentInterfaceConfig(),
    )

    builder = CoherentInterfaceBuilder(
        substrate=fcc_structure,
        film=fcc_structure,
    )
    paths = builder.export_candidates(
        candidates=[candidate],
        out_dir=str(tmp_path),
        formats=("json", "csv", "md"),
    )

    assert any(p.suffix == ".json" for p in paths)
    assert any(p.suffix == ".csv" for p in paths)
    assert any(p.suffix == ".md" for p in paths)
    assert all(p.exists() for p in paths)


def test_export_candidates_empty(tmp_path):
    builder = CoherentInterfaceBuilder(substrate=MagicMock(), film=MagicMock())
    paths = builder.export_candidates(candidates=[], out_dir=str(tmp_path))
    # Empty input still writes an empty JSON/MD skeleton so downstream
    # scripts have a predictable file to read.
    assert paths
    json_path = next(p for p in paths if p.suffix == ".json")
    assert json_path.read_text() == "[]"


def test_make_slabs_relax_terminations_flag(fcc_structure):
    sub_terms, film_terms = _make_slabs(
        fcc_structure,
        fcc_structure,
        CoherentInterfaceConfig(relax_terminations=False),
    )
    assert len(sub_terms) == 1
    assert len(film_terms) == 1

    sub_terms, film_terms = _make_slabs(
        fcc_structure,
        fcc_structure,
        CoherentInterfaceConfig(relax_terminations=True),
    )
    # Termination enumeration may collapse to one; the flag must not
    # raise and the returned list must be indexable.
    assert sub_terms
    assert film_terms


def test_match_lattices_uses_zsl_when_available(fcc_structure, bcc_structure):
    config = CoherentInterfaceConfig(
        max_strain=0.30,
        max_area=100.0,
        zsl_max_area_ratio=4.0,
        zsl_max_angle=2.0,
    )
    matches = _match_lattices(
        fcc_structure.lattice.matrix,
        bcc_structure.lattice.matrix,
        config,
    )

    assert matches
    assert all(m.strain <= config.max_strain for m in matches)
    assert all(m.shear_strain <= config.max_strain for m in matches)
    assert all(m.match_area <= config.max_area for m in matches)


def test_match_lattices_fallback_when_zsl_fails(fcc_structure):
    config = CoherentInterfaceConfig(
        max_strain=0.30,
        max_area=100.0,
    )

    with patch(
        "atominterface.builders.coherent.ZSLGenerator",
        side_effect=RuntimeError("ZSL unavailable"),
        create=True,
    ):
        matches = _match_lattices(
            fcc_structure.lattice.matrix,
            fcc_structure.lattice.matrix,
            config,
        )

    assert matches
    assert all(m.match_area <= config.max_area for m in matches)


def test_rotation_is_recorded_when_present(fcc_structure):
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)

    candidate = _to_candidate(
        interface=fcc_structure,
        substrate_slab=fcc_structure,
        film_slab=fcc_structure,
        match=_MatchInfo(
            match_area=16.0,
            substrate_supercell=(1, 1),
            film_supercell=(1, 1),
            strain=0.0,
            shear_strain=0.0,
            rotation=rotation,
        ),
        sub_term_idx=0,
        film_term_idx=0,
        config=CoherentInterfaceConfig(),
    )

    recorded = np.asarray(candidate.metadata["rotation_matrix"])
    assert recorded.shape == (3, 3)
    assert np.allclose(recorded, rotation)


def test_build_is_alias_for_build_all(fcc_structure):
    builder = CoherentInterfaceBuilder(
        substrate=fcc_structure,
        film=fcc_structure,
        config=CoherentInterfaceConfig(relax_terminations=False),
    )
    assert builder.build() == builder.build_all()


def test_registration_on_import():
    assert "coherent" in builder_registry.list()
    cls = builder_registry.get("coherent")
    assert cls is CoherentInterfaceBuilder


def _fcc(species: str, a: float):
    """Conventional fcc cell, as a bulk input for the coherent builder."""
    from pymatgen.core import Lattice, Structure  # noqa: WPS433

    return Structure(
        Lattice.cubic(a),
        [species] * 4,
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    )


def test_scale_in_plane_does_not_mutate_the_source_slab():
    """Newer pymatgen defaults ``make_supercell`` to in_place=True.

    Without a defensive copy the candidate loop compounds: the first
    candidate is 13x, the second 169x, the third 2537x.
    """
    slab = _fcc("Cu", 3.615)
    before = abs(np.linalg.det(np.asarray(slab.lattice.matrix)[:2, :2]))

    first = _scale_in_plane(slab, ((2, 0), (0, 1)))
    second = _scale_in_plane(slab, ((2, 0), (0, 1)))

    after = abs(np.linalg.det(np.asarray(slab.lattice.matrix)[:2, :2]))
    assert after == pytest.approx(before)
    assert len(first) == len(second) == 2 * len(slab)
    assert abs(np.linalg.det(np.asarray(first.lattice.matrix)[:2, :2])) == pytest.approx(2 * before)


def test_supercell_can_be_rotated_and_anisotropic():
    slab = _fcc("Cu", 3.615)
    scaled = _scale_in_plane(slab, ((1, 0), (1, 2)))
    assert len(scaled) == 2 * len(slab)


def test_build_prefers_the_smallest_equally_good_cell():
    """Identical cell shapes tie on strain; the smallest cell must win.

    Getting this wrong produced 1.8-million-atom "candidates" whose
    strain was identical to the 4-atom cell, because the ranking key was
    dominated by float noise in the strain.
    """
    builder = CoherentInterfaceBuilder(
        substrate=_fcc("Cu", 3.615),
        film=_fcc("Ni", 3.524),
        config=CoherentInterfaceConfig(max_candidates=1),
    )

    candidates = builder.build()

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.total_atoms == 4
    # fcc(001) primitive 2D cell of Cu: a^2 / 2.
    assert candidate.area == pytest.approx(3.615**2 / 2, rel=1e-6)
    # Strain applied *to the film*: (a_Cu - a_Ni) / a_Ni.
    assert candidate.metadata["mismatch"] == pytest.approx(3.615 / 3.524 - 1, rel=1e-3)


def test_build_geometry_is_coherent():
    """The substrate and the film must end up in the same in-plane cell."""
    builder = CoherentInterfaceBuilder(
        substrate=_fcc("Cu", 3.615),
        film=_fcc("Ni", 3.524),
        config=CoherentInterfaceConfig(max_candidates=3, gap=2.0),
    )

    for candidate in builder.build():
        cell = np.asarray(candidate.atoms.get_cell()[:2], dtype=float)
        # Coherent by construction: one shared cell for both slabs.
        assert candidate.area == pytest.approx(abs(np.linalg.det(cell[:2, :2])))
        # The two slabs must not overlap: film z > substrate z + gap.
        z = candidate.atoms.get_positions()[:, 2]
        symbols = candidate.atoms.get_chemical_symbols()
        sub_z = z[[s == "Cu" for s in symbols]]
        film_z = z[[s == "Ni" for s in symbols]]
        assert film_z.min() >= sub_z.max() + 2.0 - 1e-6


def test_preview_matches_ignores_the_strain_cutoff():
    builder = CoherentInterfaceBuilder(
        substrate=_fcc("Cu", 3.615),
        film=_fcc("Ni", 3.524),
        config=CoherentInterfaceConfig(max_strain=0.001),
    )

    assert builder.build() == []

    preview = builder.preview_matches(limit=3)
    assert preview
    assert len(preview) <= 3
    # The closest match is the 2.6 % isotropic Cu/Ni mismatch.
    assert preview[0].strain == pytest.approx(3.615 / 3.524 - 1, rel=1e-3)
    assert preview[0].shear_strain == pytest.approx(0.0, abs=1e-6)
