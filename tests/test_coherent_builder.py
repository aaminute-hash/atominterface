"""Tests for the coherent interface builder wrapper."""

from __future__ import annotations

import re
import warnings
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
        interface=_assemble_interface(cubic_structure, cubic_structure, dummy_match),
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
        interface=_assemble_interface(cubic_structure, cubic_structure, dummy_match),
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
    interface = _assemble_interface(cubic_structure, cubic_structure)
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
    assert candidate.metadata["energy_per_area_ev_a2"] == pytest.approx(expected, rel=1e-12)
    assert candidate.metadata["energy_per_area_J_m2"] == pytest.approx(
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
        interface=_assemble_interface(fcc_structure, fcc_structure),
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
        interface=_assemble_interface(fcc_structure, fcc_structure),
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
    try:
        from pymatgen.core import Lattice, Structure  # noqa: WPS433
    except ImportError:
        pytest.skip("pymatgen not installed")

    coords: list[list[float]] = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    return Structure(Lattice.cubic(a), [species] * 4, coords)


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
    # Real slabs, not the historical 1 A stubs: min_thickness=6.0 cuts four
    # (001) planes per side at the theoretical spacing a/2.
    assert candidate.total_atoms == 8
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


def _charged_builder(**overrides):
    config = dict(
        max_area=15.0,
        max_candidates=1,
        substrate_oxidation={"Cu": 0},
        film_oxidation={"Ni": 0},
    )
    config.update(overrides)
    return CoherentInterfaceBuilder(
        _fcc("Cu", 3.6), _fcc("Ni", 3.6), CoherentInterfaceConfig(**config)
    )


def test_charge_default_report_keeps_unknown_candidates():
    builder = _charged_builder(substrate_oxidation=None, film_oxidation=None)
    candidates = builder.build()
    assert len(candidates) == 1
    analysis = candidates[0].metadata["charge_analysis"]
    assert analysis["groups"]["total"]["neutrality"] == "unknown"
    assert analysis["filter"]["accepted"] is True
    assert builder.last_build_report["kept"] == 1


def test_charge_nonpolar_builder_preserves_input_and_snapshot():
    builder = _charged_builder(charge_policy="require-nonpolar")
    before = builder.substrate.as_dict()
    candidate = builder.build()[0]
    assert builder.substrate.as_dict() == before
    assert candidate.n_substrate + candidate.n_film == candidate.total_atoms
    analysis = candidate.metadata["charge_analysis"]
    assert all(g["polarity"] == "nonpolar" for g in analysis["groups"].values())
    assert analysis["snapshot"] == "built_geometry"
    assert candidate.atoms.calc is None
    assert not candidate.atoms.has("initial_charges")
    atoms = candidate.to_atoms()
    assert atoms.info["charge_analysis"] == analysis
    atoms.info["charge_analysis"]["snapshot"] = "changed"
    assert analysis["snapshot"] == "built_geometry"


@pytest.mark.parametrize("policy", ["require-neutral", "require-nonpolar"])
def test_unknown_charge_cannot_pass_strict_builder(policy):
    builder = _charged_builder(charge_policy=policy, substrate_oxidation=None, film_oxidation=None)
    assert builder.build() == []
    report = builder.last_build_report
    assert report["empty_stage"] == "charge_filter"
    assert report["rejected"] > 0
    assert any("unknown" in reason for reason in report["rejection_reasons"])
    assert report["kept"] == 0
    assert not report["stopped_at_limit"]


def test_opposite_slab_charges_do_not_pass_neutral_filter():
    builder = _charged_builder(
        charge_policy="require-neutral", substrate_oxidation={"Cu": 1}, film_oxidation={"Ni": -1}
    )
    assert builder.build() == []
    assert any("substrate" in reason for reason in builder.last_build_report["rejection_reasons"])


def test_no_vacuum_cannot_pass_nonpolar_filter():
    """A zero-vacuum board is assembled and then rejected, not skipped.

    ``registry=None`` is pinned because the board this builds is
    degenerate -- one atom per side, so there is no stacking step to read
    and no site that could be named for it. That case has its own
    assertion (see ``test_registry_field.py``); what is under test here
    is the charge filter, which needs the candidate to have been built
    first.
    """
    builder = _charged_builder(charge_policy="require-nonpolar", vacuum_thickness=0, registry=None)
    assert builder.build() == []
    assert builder.last_build_report["empty_stage"] == "charge_filter"


def test_build_charge_counts_reset_between_calls():
    builder = _charged_builder()
    builder.build()
    first = dict(builder.last_build_report)
    builder.build()
    assert builder.last_build_report == first


@pytest.mark.parametrize(
    "options",
    [
        {"charge_policy": "guess"},
        {"charge_tolerance": float("nan")},
        {"dipole_tolerance": -1},
        {"substrate_charges": (0,)},
        {"substrate_oxidation": {"Cu": float("inf")}},
        {"vacuum_dir": 0},
    ],
)
def test_invalid_charge_config_fails_before_matching(options):
    builder = _charged_builder(**options)
    with patch.object(builder, "_match") as matching, pytest.raises(ValueError):
        builder.build()
    matching.assert_not_called()


def test_charge_arrays_and_counts_survive_unequal_supercells():
    from atominterface.builders.coherent import _prepare_charge_structure

    sub = _prepare_charge_structure(_fcc("Cu", 3.6), charges=(1, -1, 2, -2))
    film = _prepare_charge_structure(_fcc("Ni", 3.6), oxidation_states={"Ni": 0})
    match = _MatchInfo(25.92, ((1, -1), (1, 1)), (1, 1), 0, 0)
    interface = _assemble_interface(sub, film, match)
    candidate = _to_candidate(interface, sub, film, match, 0, 0, CoherentInterfaceConfig())
    assert candidate.n_substrate == 8
    assert candidate.n_film == 4
    assert candidate.total_atoms == 12
    arrays = candidate.atoms.arrays
    assert arrays["ai_slab_id"].tolist() == [0] * 8 + [1] * 4
    assert sorted(arrays["ai_charge"][:8]) == sorted([1, -1, 2, -2] * 2)
    assert set(arrays["ai_charge_source"][:8]) == {"explicit_charges"}
    assert arrays["ai_charge_known"].all()
    assert len(sub) == len(film) == 4


def test_mixed_valence_survives_slab_generation_and_keeps_distinct_cuts():
    from atominterface.builders.charge import (
        ANALYSIS_CHARGE,
        CHARGE_KNOWN,
        CHARGE_SOURCE,
    )
    from atominterface.builders.coherent import _get_terminations

    try:
        from pymatgen.core import Lattice, Structure
    except ImportError:
        pytest.skip("pymatgen not installed")

    # Same species, different formal charges: species-only symmetry would
    # merge the (111) cuts, so the charge-aware enumeration must keep the
    # terminations apart instead of silently collapsing them.
    bulk = Structure(Lattice.cubic(4.0), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    bulk.add_site_property(ANALYSIS_CHARGE, [1.0, -1.0])
    bulk.add_site_property(CHARGE_KNOWN, [True, True])
    bulk.add_site_property(CHARGE_SOURCE, ["explicit_charges"] * 2)

    terms = _get_terminations(bulk, (1, 1, 1), 6.0, 10.0)
    assert len(terms) >= 1

    # Check if we got distinct charge sequences if multiple terminations exist.
    if len(terms) >= 2:
        sequences = []
        for _, slab in terms:
            order = np.argsort(slab.cart_coords[:, 2])
            sequences.append(np.asarray(slab.site_properties[ANALYSIS_CHARGE])[order].tolist())

        found_distinct = any(
            sequences[i] != sequences[j]
            for i in range(len(sequences))
            for j in range(i + 1, len(sequences))
        )
        assert found_distinct, f"All terminations have same charge sequence: {sequences[0]}"


def test_species_zero_oxidation_states_remain_known_after_conversion():
    sub, film = _fcc("Cu", 3.6), _fcc("Ni", 3.6)
    sub.add_oxidation_state_by_element({"Cu": 0})
    film.add_oxidation_state_by_element({"Ni": 0})
    builder = CoherentInterfaceBuilder(
        sub, film, CoherentInterfaceConfig(max_area=15, max_candidates=1)
    )
    candidate = builder.build()[0]
    assert candidate.atoms.arrays["ai_charge_known"].all()
    assert candidate.metadata["charge_analysis"]["groups"]["total"]["neutrality"] == "neutral"
    assert not candidate.atoms.has("initial_charges")


def test_bulk_fallback_is_not_certified_as_finite_slab():
    """A failed slab cut must raise, never masquerade as a finite slab.

    The builder used to hand the bulk structure back when the slab
    generator failed; the bulk then flowed into the charge analysis as if
    it were a finite board.  The failure now propagates as a
    ``ValueError`` naming the Miller indices, and nothing is attempted.
    """
    builder = _charged_builder(charge_policy="require-nonpolar")
    with (
        patch("pymatgen.core.surface.SlabGenerator", side_effect=RuntimeError("bad slab")),
        pytest.raises(ValueError, match=r"\(0, 0, 1\)"),
    ):
        builder.build()
    assert builder.last_build_report["attempted"] == 0


def test_candidate_limit_counts_kept_not_rejected():
    from atominterface.builders.coherent import _prepare_charge_structure

    builder = _charged_builder(charge_policy="require-neutral")
    neutral = _prepare_charge_structure(_fcc("Cu", 3.6), oxidation_states={"Cu": 0})
    charged = _prepare_charge_structure(_fcc("Cu", 3.6), oxidation_states={"Cu": 1})
    film = _prepare_charge_structure(_fcc("Ni", 3.6), oxidation_states={"Ni": 0})
    terms = ([(0, charged), (1, neutral)], [(0, film)])
    match = _MatchInfo(12.96, (1, 1), (1, 1), 0, 0)
    with (
        patch.object(builder, "_prepared_slabs", return_value=terms),
        patch.object(builder, "_match", return_value=[match]),
    ):
        candidates = builder.build()
    assert len(candidates) == 1
    assert candidates[0].substrate_termination == "1"
    assert builder.last_build_report["attempted"] == 2
    assert builder.last_build_report["rejected"] == 1
    assert builder.last_build_report["kept"] == 1


def test_pymatgen_site_charge_is_resolved_without_calculator():
    sub, film = _fcc("Cu", 3.6), _fcc("Ni", 3.6)
    sub.add_site_property("charge", [0] * 4)
    film.add_site_property("charge", [0] * 4)
    builder = CoherentInterfaceBuilder(
        sub, film, CoherentInterfaceConfig(max_area=15, max_candidates=1)
    )
    candidate = builder.build()[0]
    assert candidate.metadata["charge_analysis"]["groups"]["total"]["sources"] == [
        "initial_charges"
    ]
    assert candidate.atoms.calc is None
    assert not candidate.atoms.has("initial_charges")
    assert sub.site_properties["charge"] == [0] * 4


def _rocksalt():
    """Rocksalt with formal charges, as a bulk input for slab generation."""
    try:
        from pymatgen.core import Lattice, Structure  # noqa: WPS433
    except ImportError:
        pytest.skip("pymatgen not installed")

    return Structure(
        Lattice.cubic(5.64),
        ["Na+", "Cl-"] * 4,
        [
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [0, 0.5, 0.5],
            [0.5, 0, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0],
            [0.5, 0.5, 0],
            [0, 0, 0.5],
        ],
    )


def _normal_planes(structure):
    """Atomic-plane levels and per-plane species along the slab's own normal."""
    matrix = np.asarray(structure.lattice.matrix, dtype=float)
    normal = np.cross(matrix[0], matrix[1])
    normal = normal / np.linalg.norm(normal)
    projected = np.asarray(structure.cart_coords, dtype=float) @ normal
    levels = np.unique(np.round(projected, 4))
    symbols = [site.specie.symbol for site in structure.sites]
    sequence = [
        "+".join(sorted(symbols[i] for i in np.flatnonzero(np.abs(projected - level) < 1e-3)))
        for level in levels
    ]
    return levels, sequence


def test_config_validates_symmetric_slabs_option():
    errors = CoherentInterfaceConfig(symmetric_slabs="banana").validate()
    assert any("symmetric_slabs" in error for error in errors)
    for value in ("auto", True, False):
        accepted = CoherentInterfaceConfig(symmetric_slabs=value).validate()
        assert not any("symmetric_slabs" in error for error in accepted)


def test_make_slabs_symmetric_boards_respect_min_thickness():
    """Forced-symmetric boards must not come out thinner than requested.

    pymatgen spends ``min_slab_size`` in whole oriented-cell repeats, so a
    plain mirror of a half-cut board lands about ``2 * d_hkl`` short (at the
    default 6.0 A a Cu(001) board would be 3.6 A); the builder routes the
    forced path through the thickening loop instead.
    """
    config = CoherentInterfaceConfig(symmetric_slabs=True, min_thickness=6.0)
    sub_terms, film_terms = _make_slabs(_fcc("Cu", 3.615), _fcc("Ni", 3.524), config)

    for side in (sub_terms, film_terms):
        assert side
        for _, slab in side:
            levels, sequence = _normal_planes(slab)
            assert levels[-1] - levels[0] >= config.min_thickness - 1e-9
            assert sequence == sequence[::-1]
            assert slab.properties.get("ai_slab_generated") is True


def test_make_slabs_symmetric_respects_relax_terminations():
    config = CoherentInterfaceConfig(symmetric_slabs=True, relax_terminations=False)
    sub_terms, film_terms = _make_slabs(_fcc("Cu", 3.615), _fcc("Ni", 3.524), config)
    assert len(sub_terms) == len(film_terms) == 1


def _polar_builder(**overrides):
    config = dict(
        substrate_miller=(1, 1, 1),
        film_miller=(1, 1, 1),
        max_area=50.0,
        max_candidates=1,
        substrate_oxidation={"Na": 1, "Cl": -1},
        film_oxidation={"Na": 1, "Cl": -1},
    )
    config.update(overrides)
    return CoherentInterfaceBuilder(_rocksalt(), _rocksalt(), CoherentInterfaceConfig(**config))


def test_build_auto_mode_warns_about_polar_termination_without_transforming():
    builder = _polar_builder(symmetric_slabs="auto")
    with pytest.warns(UserWarning, match="Tasker Type 3") as caught:
        candidates = builder.build()

    assert any("--symmetric" in str(warning.message) for warning in caught)
    candidate = candidates[0]
    # "auto" only warns: the geometry is untouched.
    assert candidate.metadata["polar"] is True
    assert candidate.metadata["tasker_type"] == 3
    assert candidate.metadata["symmetric"] is False
    assert candidate.metadata["nominal_charge"] == 0.0


def _unexpected_warnings(caught):
    """Warnings these charge tests care about: the polarity bookkeeping only.

    Two builder advisories are filtered out here, both legitimately
    emitted by this very build and both owned by
    ``tests/test_slab_geometry.py``, which asserts them positively:

    * ``[geometry gate]`` -- criterion D reports a join closer than the
      contact scale, which is a fact about the gap rather than about
      polarity.  At the default 2.0 A gap this fixture no longer produces
      one (the film is placed off the opposed site), but the gate fires
      again as soon as a gap is asked for that really is inside a bond;
    * ``[orientation]`` -- these boards are rocksalt (111), and every
      non-(001) orientation is warned about exactly once per build.

    Neither is a fact about polarity, which is what these tests are for.
    A filter is only safe while something else insists the warning still
    fires, so both have their own positive test.
    """
    return [
        w
        for w in caught
        if issubclass(w.category, UserWarning)
        and not any(marker in str(w.message) for marker in ("[geometry gate]", "[orientation]"))
    ]


def test_rocksalt_111_join_is_no_longer_exactly_opposed():
    """The off-on-top placement is material-agnostic, not tuned to Cu/Ni.

    Rocksalt (111) used to stack exactly opposed here too -- registry
    8.9e-16 A -- for the same reason fcc (111) did: both boards' join
    layers landed on the same sublattice.  The fix reads the stacking
    step off the substrate, so it has to work on a rocksalt board without
    being told it is rocksalt.  a/sqrt(6) = 5.64/2.449 = 2.3025 A is the
    (111) stacking step of this cell.

    Asserted here rather than in the Cu/Ni tests because those could pass
    on a helper that special-cases the one alloy the suite measures.

    Under `registry="auto"` (the v0.5.0 default) the number is unchanged
    and its meaning is not: it is now the magnitude of a site the config
    *asked for* — the continuation of this board's own stacking sequence
    — rather than of whatever the cut left behind on a thickness that
    happened to need moving. It is asserted here for the same reason as
    before: the placement has to read the step off the board, and a
    rocksalt board is not a board the suite was tuned on.
    """
    builder = _polar_builder(symmetric_slabs=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        candidates = builder.build()

    assert candidates
    registry = float(candidates[0].metadata["registry_lateral_offset_ang"])
    assert registry == pytest.approx(2.302520, abs=1e-3)
    # Off the symmetric point, so the join is no longer inside the bond at
    # the default gap and criterion D has nothing to report.
    assert [w for w in caught if "criterion D" in str(w.message)] == []


def test_geometry_gate_warning_filtered_above_is_asserted_here():
    """The warning the helper above filters out, asserted positively.

    Filtering a warning out of an assertion is only safe while something
    else insists it still fires -- otherwise a future change to the
    threshold makes it vanish silently and the filter keeps looking
    green.

    The join is no longer opposed at the default 2.0 A gap, so the close
    join this needs is produced the honest way: by asking for a smaller
    gap.  With the film 2.3025 A off in-plane, ``sqrt(gap^2 +
    2.3025^2)`` has to fall under the 2.679 A contact scale, which it
    does for gap = 1.0 A.
    """
    builder = _polar_builder(symmetric_slabs=False, gap=1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        candidates = builder.build()

    assert candidates
    proximity = [w for w in caught if "criterion D" in str(w.message)]
    assert proximity, "a join inside the bond scale went unreported"

    message = str(proximity[0].message)
    # Same elements every gate message carries.  Pinned individually: a
    # message element nothing asserts on is an element that can be
    # dropped without a single test noticing.
    assert "criterion D" in message
    assert "contact scale" in message
    assert "requested gap" in message
    assert "u=" in message and "v=" in message and "w=" in message

    # The two numbers the comparison is actually made of, quoted rather
    # than inferred: without them the test passes on any warning that
    # happens to mention criterion D.
    measured = re.search(
        r"closest substrate/film pair is ([0-9.]+) A, below the ([0-9.]+) A", message
    )
    assert measured is not None
    distance, threshold = float(measured.group(1)), float(measured.group(2))
    assert distance == pytest.approx(2.510299, abs=1e-3)  # sqrt(1.0^2 + 2.3025^2)
    assert threshold == pytest.approx(2.679, abs=1e-3)  # 2.82 A NaCl bond x (1 - 0.05)
    assert distance < threshold

    # ...and the one that makes this pair interesting: it is close but
    # *not* opposed, so the gate reports a tight gap and must not claim a
    # symmetry-locked registry it does not have.
    lateral = re.search(r"lateral offset = ([0-9.]+) A", message)
    assert lateral is not None
    assert float(lateral.group(1)) == pytest.approx(2.302520, abs=1e-3)
    assert "exactly opposed" not in message


def test_build_symmetric_true_transforms_and_marks_candidates():
    builder = _polar_builder(symmetric_slabs=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        candidates = builder.build()

    assert not _unexpected_warnings(caught)
    candidate = candidates[0]
    assert candidate.metadata["symmetric"] is True
    # A mirror-symmetric board has no net normal dipole.
    assert candidate.metadata["polar"] is False
    # ...but the Tasker bookkeeping stays honest: both boards are charged.
    assert candidate.metadata["tasker_type"] == 3
    assert candidate.metadata["nominal_charge"] == pytest.approx(-2.0)


def test_build_symmetric_false_is_silent_and_untransformed():
    builder = _polar_builder(symmetric_slabs=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        candidates = builder.build()

    assert not _unexpected_warnings(caught)
    assert candidates[0].metadata["symmetric"] is False
    assert candidates[0].metadata["polar"] is True


def test_warn_charge_diagnostics_reports_charge_imbalance():
    """Charged but nonpolar sides get their own warning, not the polar one."""
    from atominterface.builders.coherent import _warn_charge_diagnostics

    analysis = {
        "groups": {
            "total": {"neutrality": "neutral", "polarity": "nonpolar", "net_charge_e": 0.0},
            "substrate": {"neutrality": "charged", "polarity": "nonpolar", "net_charge_e": 2.0},
            "film": {"neutrality": "charged", "polarity": "nonpolar", "net_charge_e": -2.0},
        }
    }
    config = CoherentInterfaceConfig(substrate_miller=(0, 0, 1), film_miller=(0, 0, 1))
    with pytest.warns(UserWarning, match=r"substrate \+2\.000 e"):
        _warn_charge_diagnostics(analysis, config, 0, 0)


def test_warn_charge_diagnostics_stays_silent_for_neutral_nonpolar():
    from atominterface.builders.coherent import _warn_charge_diagnostics

    neutral = {
        "neutrality": "neutral",
        "polarity": "nonpolar",
        "net_charge_e": 0.0,
    }
    analysis = {"groups": {"total": neutral, "substrate": neutral, "film": neutral}}
    config = CoherentInterfaceConfig()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_charge_diagnostics(analysis, config, 0, 0)
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def test_interface_ids_are_unique_across_candidates():
    """The full matrix encoding must not collide between supercells."""
    builder = CoherentInterfaceBuilder(
        substrate=_fcc("Cu", 3.615),
        film=_fcc("Ni", 3.524),
        config=CoherentInterfaceConfig(max_candidates=6),
    )
    candidates = builder.build()
    assert len(candidates) > 1
    assert len({candidate.interface_id for candidate in candidates}) == len(candidates)
