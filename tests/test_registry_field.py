"""A requested lateral registry is placed, measured and recorded.

Before ``registry`` existed, assembly had one in-plane step: if the two
slab cuts happened to meet exactly opposed, the film was nudged off by
the substrate's stacking step. That is an *incremental* rule, and it is
only right where the cut leaves the film on top -- on fcc (111) it
produces the fcc hollow, but on fcc (110) it produces the short bridge,
1.5189 J/m^2 against a scanned 0.0881, 2.55x. So ``registry`` is
*targeted*: name the site, then solve for the displacement that reaches
it.

These tests pin, in this order:

* the field's vocabulary and its rejections, every one of them naming
  what could have been asked instead;
* that a requested site is the site that was built, and that a
  placement that misses is refused rather than shipped;
* the three orientations as measured, including the (001) identity that
  the whole rule depends on -- (001) is the one orientation with a DFT
  cross-check, and it is untouched;
* that what was asked for, where the join was built and where it
  stopped are three separate recorded facts.

No MLIP is involved: everything here is decided during construction and
the one relaxation is faked.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from typer.testing import CliRunner

from atominterface import cli_build
from atominterface.builders.coherent import (
    REGISTRY_MODES,
    SLAB_ID,
    CoherentInterfaceBuilder,
    CoherentInterfaceConfig,
    _check_registry_placement,
    _registry_shift,
    _registry_target_frac,
    _stacking_period,
    _stacking_step,
    measure_registry,
)
from atominterface.cli import app
from atominterface.relax import RelaxResult

runner = CliRunner()

#: The four thicknesses the validation matrix documents.
THICKNESSES = (10.8, 14.4, 18.0, 21.6)

#: Lateral offset of a film atom on a hollow site of the Cu (111) surface:
#: a_nn / sqrt(3) = (3.615 / sqrt(2)) / sqrt(3).
HOLLOW_LATERAL_111 = 1.4758175700268625

#: The as-built (001) and (110) registry: half an in-plane lattice vector.
AS_BUILT_LATERAL_001_110 = 1.8075

#: The (110) trough, |(1/2, 1/2)| on the Cu (110) cell: half the diagonal
#: of the a/2 x a/sqrt(2) rectangle.  This is where ``auto`` puts (110),
#: and it is the minimum of the archive's own four-site scan (0.0881).
TROUGH_LATERAL_110 = 2.2137263550402930

#: What moving (110) from the ridge to the trough costs in displacement:
#: a / (2 sqrt(2)) -- half the short in-plane vector of the (110) cell.
RIDGE_TO_TROUGH_110 = 1.2780955069945400

#: The (111) thicknesses whose as-built join is already off the symmetric
#: point, so the v0.4.1 nudge never fires there.
QUIESCENT_111 = (10.8, 18.0)

#: The one (111) thickness whose cut leaves the join exactly opposed, so
#: the nudge is what it sits on -- and the one whose cut leaves it on the
#: hcp hollow, which ``auto`` is supposed to leave.
ON_TOP_111 = 14.4
HCP_111 = 21.6


def _fcc(species: str, a: float) -> Structure:
    """Conventional fcc cell, as a bulk input for slab generation."""
    coords: list[list[float]] = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    return Structure(Lattice.cubic(a), [species] * 4, coords)


def _build(miller, min_thickness, **overrides):
    """First candidate of a Cu/Ni interface at ``miller`` and ``min_thickness``.

    ``max_area`` and ``max_candidates`` are kept at the smallest window
    that still contains the 1x1 match on all three orientations, because
    every case here is a statement about the 1x1 candidate and the
    enumeration beyond it is pure cost: the same geometry, a tenth of
    the runtime.
    """
    config = CoherentInterfaceConfig(
        substrate_miller=miller,
        film_miller=miller,
        min_thickness=min_thickness,
        max_area=15.0,
        max_candidates=1,
        **overrides,
    )
    builder = CoherentInterfaceBuilder(
        substrate=_fcc("Cu", 3.615), film=_fcc("Ni", 3.524), config=config
    )
    candidates = builder.build()
    assert candidates
    return candidates[0]


def _lateral(candidate) -> float:
    return float(candidate.metadata["registry_lateral_offset_ang"])


def _shift_between(reference, other) -> float:
    """Minimum-image in-plane displacement of the film between two builds.

    A displacement of one lattice vector is no displacement at all, so
    the difference is reduced before it is measured -- otherwise this
    would report a move wherever the two builds took the shortest path
    to the same site by different representatives.
    """
    n = reference.n_substrate
    delta = other.atoms.get_positions()[n:] - reference.atoms.get_positions()[n:]
    cell = np.asarray(reference.atoms.cell, dtype=np.float64)[:2, :2]
    frac = delta[:, :2] @ np.linalg.inv(cell)
    wrapped = (frac - np.round(frac)) @ cell
    return float(np.max(np.linalg.norm(wrapped, axis=1)))


def _minimal_boards() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A substrate and a film with one atomic plane each, and a square cell."""
    return (
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        np.asarray([[3.0, 0.0], [0.0, 3.0]], dtype=np.float64),
    )


# ── the vocabulary ────────────────────────────────────────────────────


def test_default_registry_is_auto():
    """The default asks for a site instead of accepting one by inheritance.

    The old default produced, on (110), a number the archive's own scan
    puts 6.8x above its minimum. A field that could fix that and did not
    would be half a measure.
    """
    assert CoherentInterfaceConfig().registry == "auto"
    assert set(REGISTRY_MODES) == {"auto", "on-top", "hcp"}


def test_none_keeps_the_v041_path_available():
    """``None`` is a legal request: the as-built behaviour, bit for bit."""
    assert CoherentInterfaceConfig(registry=None).validate() == []


@pytest.mark.parametrize(
    "mode", ["fcc-hollow", "trough", "ridge", "bridge", "short-bridge", "AUTO", "on_top", ""]
)
def test_unknown_registry_names_are_rejected_with_the_vocabulary(mode):
    """A name this vocabulary cannot resolve is answered, not guessed at.

    Every one of these is a name a surface scientist would reach for,
    and every one of them means a different fractional offset on (111)
    than on (110) -- which is why none of them is in the vocabulary. The
    error has to carry the alternatives plus ``(u, v)``, or the user is
    left to guess again.
    """
    errors = CoherentInterfaceConfig(registry=mode).validate()
    assert errors
    text = " ".join(errors)
    for available in REGISTRY_MODES:
        assert available in text
    assert "(u, v)" in text
    assert mode in text


def test_scan_is_accepted_as_a_valid_registry_request():
    """``"scan"`` is the explicit enumeration; it must pass validation.

    It builds one candidate per distinguishable site and is the only
    thing that can retire the non-(001) warning, so it is now a legal
    request rather than a reserved name.
    """
    errors = CoherentInterfaceConfig(registry="scan").validate()
    assert errors == []


@pytest.mark.parametrize("value", [(0.5,), (0.5, 0.5, 0.5), ()])
def test_uv_must_have_exactly_two_components(value):
    errors = CoherentInterfaceConfig(registry=value).validate()
    assert errors
    assert "exactly 2 components" in errors[0]
    assert str(len(value)) in errors[0]


@pytest.mark.parametrize(
    "value", [(float("nan"), 0.0), (float("inf"), 0.0), (0.0, float("-inf")), ("a", "b")]
)
def test_uv_must_be_finite_numbers(value):
    errors = CoherentInterfaceConfig(registry=value).validate()
    assert errors
    assert "finite" in errors[0] or "pair of" in errors[0]


@pytest.mark.parametrize("value", ["auto", "on-top", "hcp", (0.5, 0.5), [0.5, 0.5], None])
def test_every_legal_registry_validates(value):
    """The accepted spellings, including the list a YAML replay produces.

    A recorded run comes back from YAML with ``u,v`` as a list, so a list
    has to validate exactly like the tuple the command line builds.
    """
    assert CoherentInterfaceConfig(registry=value).validate() == []


# ── the placement ─────────────────────────────────────────────────────


@pytest.mark.parametrize("min_thickness", THICKNESSES)
def test_auto_leaves_001_untouched(min_thickness):
    """(001) is bit-identical: the one orientation with a DFT cross-check.

    Not a tolerance. The rule is a target and not an increment precisely
    so that (001) -- where the as-built site is half a lattice vector and
    the sequence continues to the same atom -- resolves to no movement at
    all. The published (001) numbers have a cross-check the (110) and
    (111) ones do not, and they are the reason the rule was written this
    way round.
    """
    as_built = _build((0, 0, 1), min_thickness, registry=None)
    placed = _build((0, 0, 1), min_thickness, registry="auto")
    assert np.array_equal(as_built.atoms.get_positions(), placed.atoms.get_positions())
    assert np.array_equal(np.asarray(as_built.atoms.cell), np.asarray(placed.atoms.cell))
    assert _lateral(placed) == pytest.approx(AS_BUILT_LATERAL_001_110, abs=1e-6)
    assert _shift_between(as_built, placed) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("min_thickness", QUIESCENT_111)
def test_auto_leaves_111_where_the_fix_already_put_it(min_thickness):
    """On the thicknesses the old nudge never fired, ``auto`` fires neither."""
    as_built = _build((1, 1, 1), min_thickness, registry=None)
    placed = _build((1, 1, 1), min_thickness, registry="auto")
    assert np.array_equal(as_built.atoms.get_positions(), placed.atoms.get_positions())
    assert _lateral(placed) == pytest.approx(HOLLOW_LATERAL_111, abs=1e-6)


def test_auto_leaves_111_at_14_4_on_the_same_site_as_the_fix():
    """At 14.4 A the old nudge fires; the target reaches the same hollow.

    Both routes put the film on the fcc hollow, which is what makes this
    release a change of mechanism rather than of result on (111): the
    site is the one the 0.4.1 changelog records as confirmed.
    """
    as_built = _build((1, 1, 1), ON_TOP_111, registry=None)
    placed = _build((1, 1, 1), ON_TOP_111, registry="auto")
    assert np.array_equal(as_built.atoms.get_positions(), placed.atoms.get_positions())
    assert _lateral(placed) == pytest.approx(HOLLOW_LATERAL_111, abs=1e-6)


def test_auto_moves_111_at_21_6_from_hcp_to_fcc():
    """The one (111) thickness whose cut leaves the join on the hcp hollow.

    The point of the whole thickness column: with ``auto`` the four
    thicknesses all start from the fcc hollow, so a (111) thickness scan
    varies one thing. The displacement is one hollow-to-hollow step,
    1.4758 A, and the lateral distance does not change -- both hollows
    are the same distance from on-top, which is exactly why the lateral
    offset alone cannot tell them apart and the site has to be reported
    as fractional coordinates.
    """
    as_built = _build((1, 1, 1), HCP_111, registry=None)
    placed = _build((1, 1, 1), HCP_111, registry="auto")
    assert _shift_between(as_built, placed) == pytest.approx(HOLLOW_LATERAL_111, abs=1e-6)
    assert _lateral(placed) == pytest.approx(HOLLOW_LATERAL_111, abs=1e-6)

    sub_coords, film_coords, cell = _boards_of(as_built)
    step = _stacking_step(sub_coords, cell)
    assert _stacking_period(step) == 3
    # The two hollows, named by the stacking sequence and not by a label:
    # the one the sequence continues to is a step further on, and it is
    # the one the thicknesses above already sit on.
    requested = np.asarray(placed.metadata["registry_target_frac"], dtype=np.float64)
    assert np.allclose(requested, step, atol=1e-6)
    assert np.allclose(
        np.asarray(placed.metadata["registry_as_built_frac"], dtype=np.float64),
        step,
        atol=1e-6,
    )
    # The as-built join this thickness was handed is the other hollow --
    # one step back along the same sequence.
    assert np.allclose(
        np.asarray(as_built.metadata["registry_as_built_frac"], dtype=np.float64),
        -step,
        atol=1e-6,
    )


@pytest.mark.parametrize("min_thickness", THICKNESSES)
def test_auto_puts_110_on_the_trough(min_thickness):
    """(110) moves from the ridge to the trough, 1.2781 A.

    This is the case the feature exists for. The as-built ridge reads
    1.8075 A and gamma_int 0.5952 J/m^2; the incremental reading -- add
    the stacking step to whatever the cut produced -- arrives at the
    short bridge, 1.5189, 2.55x the trough's 0.0881. The trough is the
    minimum of the archive's own four-site scan, so a default that lands
    there is a default that can be cited.
    """
    as_built = _build((1, 1, 0), min_thickness, registry=None)
    placed = _build((1, 1, 0), min_thickness, registry="auto")
    assert _lateral(placed) == pytest.approx(TROUGH_LATERAL_110, abs=1e-6)
    assert _lateral(as_built) == pytest.approx(AS_BUILT_LATERAL_001_110, abs=1e-6)
    assert _shift_between(as_built, placed) == pytest.approx(RIDGE_TO_TROUGH_110, abs=1e-6)


def test_auto_is_not_the_unfixed_path_on_110(monkeypatch):
    """``auto`` has to move (110) even where the old nudge never fired.

    Arm A is ``registry=None`` with ``_off_on_top_shift`` still in place;
    arm B is the same build with the nudge replaced by zeros, which is
    v0.4.1 with its own fix taken back out. On (110) those two arms are
    bit-identical -- the nudge never fired there, which is exactly why
    the (110) numbers were wrong. So the arms agreeing is not evidence
    that the new default works; only the third build is.
    """
    from atominterface._coherent import placement

    reference = _build((1, 1, 0), 14.4, registry=None)
    monkeypatch.setattr(placement, "_off_on_top_shift", lambda *a, **k: np.zeros(2))
    unfixed = _build((1, 1, 0), 14.4, registry=None)
    monkeypatch.undo()

    assert np.array_equal(reference.atoms.get_positions(), unfixed.atoms.get_positions())
    placed = _build((1, 1, 0), 14.4, registry="auto")
    assert not np.array_equal(reference.atoms.get_positions(), placed.atoms.get_positions())
    assert _lateral(placed) == pytest.approx(TROUGH_LATERAL_110, abs=1e-6)


@pytest.mark.parametrize(
    ("miller", "target", "expected"),
    [
        ((1, 1, 0), (0.0, 0.0), 0.0),  # on-top
        ((1, 1, 0), (0.5, 0.5), TROUGH_LATERAL_110),  # trough
        ((1, 1, 0), (0.0, 0.5), RIDGE_TO_TROUGH_110),  # short bridge
        ((1, 1, 1), (0.0, 0.0), 0.0),  # on-top
        ((1, 1, 1), (1 / 3, 1 / 3), HOLLOW_LATERAL_111),  # one hollow
        ((1, 1, 1), (2 / 3, 2 / 3), HOLLOW_LATERAL_111),  # the other hollow
    ],
)
def test_explicit_uv_lands_on_the_site_it_names(miller, target, expected):
    """An explicit offset is honoured, and the landing site is recorded.

    The two (111) hollows are the same distance from on-top, so the
    lateral number is checked *and* the fractional one: a placement that
    landed on the other hollow would pass the first and fail the second.
    """
    candidate = _build(miller, 14.4, registry=target)
    built = np.asarray(candidate.metadata["registry_as_built_frac"], dtype=np.float64)
    requested = np.asarray(candidate.metadata["registry_target_frac"], dtype=np.float64)
    # Same site, compared in fractional coordinates and wrapped, so the
    # representative cannot make a hit look like a miss.
    deviation = built - requested
    assert np.allclose(deviation - np.round(deviation), 0.0, atol=1e-6)
    assert _lateral(candidate) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    ("miller", "min_thickness"),
    [((0, 0, 1), 14.4), ((1, 1, 0), 14.4), ((1, 1, 1), 10.8), ((1, 1, 1), 18.0)],
)
def test_requesting_the_site_the_cut_already_gives_moves_nothing(miller, min_thickness):
    """Asking for what you were going to get costs no displacement.

    ``array_equal``, not ``allclose``: the claim is that the builder
    wrote nothing at all, and a tolerance would turn a small unintended
    move into a passing test.
    """
    as_built = _build(miller, min_thickness, registry=None)
    site = tuple(as_built.metadata["registry_as_built_frac"])
    placed = _build(miller, min_thickness, registry=site)
    assert np.array_equal(as_built.atoms.get_positions(), placed.atoms.get_positions())


@pytest.mark.parametrize("miller", [(0, 0, 1), (1, 1, 1), (1, 1, 0)])
def test_on_top_is_the_origin_everywhere(miller):
    """``"on-top"`` is (0, 0) whatever the surface, and it lands there.

    Worth its own case because it is the site the criterion-D warning
    calls the symmetry-locked one: a user who asks for it has to be able
    to get it, and a user who did not ask for it has to be able to see
    it in the record.
    """
    candidate = _build(miller, 14.4, registry="on-top")
    assert np.allclose(candidate.metadata["registry_as_built_frac"], (0.0, 0.0), atol=1e-6)
    assert _lateral(candidate) == pytest.approx(0.0, abs=1e-6)


def test_hcp_is_the_other_hollow_on_111():
    """On a three-sublattice surface ``hcp`` is a different atom.

    The (111) stacking sequence resolves three sublattices, so the two
    hollows are distinguishable and the name means something: it is the
    one directly above a second-layer atom, where the sequence would
    continue as hcp rather than fcc.
    """
    fcc_site = _build((1, 1, 1), 14.4, registry="auto")
    hcp_site = _build((1, 1, 1), 14.4, registry="hcp")
    built_fcc = np.asarray(fcc_site.metadata["registry_as_built_frac"], dtype=np.float64)
    built_hcp = np.asarray(hcp_site.metadata["registry_as_built_frac"], dtype=np.float64)
    assert np.allclose(built_fcc, -built_hcp, atol=1e-6)
    assert not np.allclose(built_fcc, built_hcp, atol=1e-6)
    # ...and it is genuinely a different site, not a different
    # representative of one: a third of a cell apart, not a whole one.
    assert _shift_between(fcc_site, hcp_site) == pytest.approx(HOLLOW_LATERAL_111, abs=1e-6)


@pytest.mark.parametrize("miller", [(0, 0, 1), (1, 1, 0)])
def test_hcp_is_refused_where_the_sequence_repeats_every_two_layers(miller):
    """``hcp`` on a two-sublattice surface would be naming nothing.

    On (110) the stacking step is half a lattice vector, so its negative
    is the same atom and "the other hollow" is the hollow just asked for.
    A name that silently resolved to ``auto`` would be a request that
    looked honoured while meaning something else, so it raises -- and it
    says which sites the surface does resolve, because that is the
    answer the user actually needs.
    """
    candidate = _build(miller, 14.4, registry="auto")
    sub_coords, film_coords, cell = _boards_of(candidate)
    step = _stacking_step(sub_coords, cell)
    assert _stacking_period(step) == 2
    with pytest.raises(ValueError) as caught:
        _registry_target_frac(sub_coords, cell, "hcp")
    message = str(caught.value)
    assert "three sublattices" in message
    assert "repeats after 2 layer(s)" in message
    assert "0x step" in message and "1x step" in message
    assert "registry='auto'" in message and "explicit (u, v)" in message


def test_a_surface_that_cannot_name_a_site_produces_no_candidate():
    """A request that cannot be honoured builds nothing, and says so.

    Built structures are the only thing a downstream reader sees, so the
    failure has to be visible in the report rather than lost inside the
    candidate loop.
    """
    builder = CoherentInterfaceBuilder(
        substrate=_fcc("Cu", 3.615),
        film=_fcc("Ni", 3.524),
        config=CoherentInterfaceConfig(
            substrate_miller=(1, 1, 0), film_miller=(1, 1, 0), min_thickness=14.4, registry="hcp"
        ),
    )
    assert builder.build() == []
    report = builder.last_build_report
    assert report["assembly_failed"] == report["attempted"]
    assert report["attempted"] > 0
    assert report["empty_stage"] == "assembly"


def test_a_refused_site_also_says_why():
    """Zero candidates is a count, not an explanation.

    Eighty-seven matches that all die of the same unnameable site is one
    cause. The report carries the cause so the CLI can print it, and caps
    it so the cause is not buried under the enumeration that repeated it.
    """
    builder = CoherentInterfaceBuilder(
        substrate=_fcc("Cu", 3.615),
        film=_fcc("Ni", 3.524),
        config=CoherentInterfaceConfig(
            substrate_miller=(1, 1, 0), film_miller=(1, 1, 0), min_thickness=14.4, registry="hcp"
        ),
    )
    assert builder.build() == []
    report = builder.last_build_report
    detail = report["assembly_detail"]
    assert detail, "a refused site must leave a reason, not just a count"
    assert any("three sublattices" in line for line in detail)
    assert sum(report["assembly_reasons"].values()) == report["assembly_failed"]
    # One cause, many cells: the detail is capped, not proportional to the
    # number of matches the ZSL enumeration handed over.
    assert len(detail) <= 4
    assert report["attempted"] > len(detail)


def test_a_board_without_a_stacking_sequence_is_refused_by_name():
    """One atomic plane cannot say which way the sequence is going.

    The refusal names the reason instead of falling back to the cut's
    position: a join built at a site nobody can name is exactly the
    trustworthy-looking output this package refuses to produce.
    """
    sub_coords, film_coords, cell = _minimal_boards()
    with pytest.raises(ValueError) as caught:
        _registry_shift(sub_coords, film_coords, cell, "auto")
    assert "cannot be resolved" in str(caught.value)
    assert "two atomic planes" in str(caught.value)

    with pytest.raises(ValueError) as caught:
        _registry_shift(sub_coords, film_coords, cell, "hcp")
    assert "cannot be resolved" in str(caught.value)


def test_a_placement_that_misses_is_refused_instead_of_shipped():
    """The self-check catches a join that did not land where it was aimed.

    The failure it is looking for is not hypothetical: a wrap applied to
    a cartesian difference put a (110) film 0.31 of a cell off its
    requested site, and every downstream check stayed green while the
    energy came out belonging to no site anyone had asked for.
    """
    cell = np.asarray([[3.0, 0.0], [0.0, 3.0]], dtype=np.float64)
    target = np.asarray([0.5, 0.5], dtype=np.float64)
    landed = np.asarray([1.0, 0.5], dtype=np.float64) @ cell
    with pytest.raises(ValueError) as caught:
        _check_registry_placement(landed, target, "registry='auto'", cell)
    message = str(caught.value)
    assert "1.500000 A from it" in message
    assert "requested site" in message and "measured" in message
    assert "not being built" in message

    # ...and the same check passes when the join is where it was aimed,
    # including when the two agree by a whole cell.
    exact = np.asarray([0.5, 0.5], dtype=np.float64) @ cell
    _check_registry_placement(exact, target, "registry='auto'", cell)
    _check_registry_placement(exact + cell[0], target, "registry='auto'", cell)


def test_the_placement_check_refuses_an_unmeasurable_join():
    """A join that cannot be re-measured is reported, not assumed.

    "I could not tell" is not evidence that the film is on the site it
    was aimed at, so it is an error and not a pass.
    """
    cell = np.asarray([[3.0, 0.0], [0.0, 3.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="could not be re-measured"):
        _check_registry_placement(None, np.zeros(2), "registry='auto'", cell)


# ── the record ────────────────────────────────────────────────────────


def test_every_candidate_carries_what_was_asked_and_what_was_built():
    """Three facts, kept apart: the request, the site, and the result.

    Collapsing them into the single lateral number the candidate used to
    carry would make a request that was not honoured invisible -- and the
    lateral number cannot do the job on its own, because the two (111)
    hollows and the two (110) half-cell sites share one lateral distance.
    """
    candidate = _build((1, 1, 0), 14.4, registry="auto")
    assert candidate.metadata["registry_requested"] == "auto"
    assert candidate.metadata["registry_target_frac"] is not None
    assert candidate.metadata["registry_as_built_frac"] is not None
    assert candidate.metadata["registry_as_built_lateral_A"] == pytest.approx(
        TROUGH_LATERAL_110, abs=1e-6
    )
    # The historical key downstream consumers read is untouched.
    assert candidate.metadata["registry_lateral_offset_ang"] == pytest.approx(
        TROUGH_LATERAL_110, abs=1e-6
    )


def test_as_built_under_none_is_recorded_without_a_target():
    """With ``None`` there is no requested site, and the record says so.

    A blank is not the same claim as a request: "the cut put it here" and
    "you asked for it here" have to stay distinguishable in a record that
    someone will read later.
    """
    candidate = _build((1, 1, 0), 14.4, registry=None)
    assert candidate.metadata["registry_requested"] is None
    assert candidate.metadata["registry_target_frac"] is None
    assert candidate.metadata["registry_as_built_lateral_A"] == pytest.approx(
        AS_BUILT_LATERAL_001_110, abs=1e-6
    )


@pytest.mark.parametrize("requested", [None, "auto", "on-top", (0.5, 0.5)])
def test_the_requested_registry_is_json_serializable(requested):
    """The request reaches a JSON report, so it has to survive one.

    Tuples serialise as arrays but come back as lists, so a tuple in the
    metadata is a value that stops comparing equal to the config it came
    from. It is normalised on the way in instead.
    """
    candidate = _build((1, 1, 0), 14.4, registry=requested)
    value = candidate.metadata["registry_requested"]
    assert json.loads(json.dumps(value)) == value
    if isinstance(requested, tuple):
        assert value == list(requested)
        assert isinstance(value, list)


def test_measure_registry_agrees_with_the_built_site():
    """The public measurement is the same route the builder recorded.

    Two definitions of "where the join is" would be two chances to
    disagree about what a run shipped, so the public helper used after a
    relaxation is the private one used during assembly.
    """
    candidate = _build((1, 1, 0), 14.4, registry="auto")
    measured = measure_registry(candidate.atoms)
    assert measured is not None
    assert measured["lateral"] == pytest.approx(
        candidate.metadata["registry_as_built_lateral_A"], abs=1e-9
    )
    difference = np.asarray(measured["frac"]) - np.asarray(
        candidate.metadata["registry_as_built_frac"]
    )
    assert np.allclose(difference - np.round(difference), 0.0, atol=1e-9)


def test_measure_registry_reports_none_rather_than_zero(monkeypatch):
    """An unmeasurable registry is a blank, never a plausible zero.

    A zero would be read as "on-top" -- the symmetry-locked site -- by
    anyone who did not know it meant "not measured".
    """
    candidate = _build((1, 1, 0), 14.4, registry="auto")
    atoms = candidate.atoms.copy()
    del atoms.arrays[SLAB_ID]
    assert measure_registry(atoms) is None

    single = candidate.atoms.copy()
    single.arrays[SLAB_ID] = np.ones(len(single), dtype=np.int64)
    assert measure_registry(single) is None


def test_relaxation_records_the_site_it_stopped_on(tmp_path, monkeypatch):
    """The post-relaxation site is measured, not recalled from the request.

    This is the third fact and the one that cannot be reconstructed: the
    archive holds a (110) run at 21.6 A that slid from the ridge to the
    trough on its own, so a candidate that records only where it was
    built lets a reader take that for where it stopped. The relaxation
    here is faked to slide, which is what makes the test able to tell the
    two apart.
    """

    class _SlidingRelaxer:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, atoms):
            moved = atoms.copy()
            tags = moved.arrays[SLAB_ID]
            moved.positions[tags == 1, 0] += 0.4
            return RelaxResult(
                atoms=moved,
                n_steps=3,
                converged=True,
                valid=True,
                energies_finite=True,
                forces_finite=True,
                final_energy=0.0,
                metadata={"committee_enabled": False, "ood_monitoring": "heuristics_only"},
            )

    monkeypatch.setattr("atominterface.relax.GuardedRelaxer", _SlidingRelaxer)
    monkeypatch.setattr(
        "atominterface.backends.get_calculator",
        lambda name, **kwargs: lambda atoms: None,
    )
    candidate = _build((1, 1, 0), 14.4, registry="auto")
    built = candidate.metadata["registry_as_built_lateral_A"]
    cli_build._relax_candidates(
        [candidate],
        backend="dummy",
        committee_backends=[],
        device="cpu",
        fmax=0.05,
        max_steps=5,
        ood_threshold=0.8,
        outdir=tmp_path,
    )
    final = candidate.metadata["registry_final_lateral_A"]
    assert final is not None
    assert final != pytest.approx(built, abs=1e-3)
    assert candidate.metadata["registry_final_frac"] is not None
    # The relax record itself is untouched: it is compared as a whole
    # downstream, so the two facts are written beside it and not into it.
    assert "registry_final_frac" not in candidate.metadata["relax"]


# ── the command line ──────────────────────────────────────────────────


@pytest.fixture
def fake_build(monkeypatch):
    """Exercise the real config assembly without lattice enumeration."""
    from types import SimpleNamespace

    state = SimpleNamespace(
        candidates=[],
        report={
            "charge_policy": "report",
            "attempted": 0,
            "assembled": 0,
            "assembly_failed": 0,
            "analyzed": 0,
            "kept": 0,
            "rejected": 0,
            "rejection_reasons": {},
            "stopped_at_limit": False,
            "empty_stage": "no_matches",
        },
        config=None,
    )

    class FakeBuilder:
        def __init__(self, *, substrate, film, config):
            state.config = config
            self.last_build_report = state.report

        def build(self):
            return state.candidates

        def preview_matches(self, limit):
            return []

    def load(path):
        return SimpleNamespace(composition=SimpleNamespace(reduced_formula="Cu"))

    monkeypatch.setattr(cli_build, "_resolve_builder", lambda name: FakeBuilder)
    monkeypatch.setattr(cli_build, "_load_structure", load)
    return state


def _combined(result) -> str:
    output = result.output or ""
    try:
        stderr = result.stderr or ""
    except ValueError:
        stderr = ""
    return output + stderr


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        (["--registry", "auto"], "auto"),
        (["--registry", "on-top"], "on-top"),
        (["--registry", "hcp"], "hcp"),
        (["--registry", "0.5,0.5"], (0.5, 0.5)),
        (["--registry", "0.25, -0.5"], (0.25, -0.5)),
    ],
)
def test_cli_passes_every_spelling_through(fake_build, tmp_path, flag, expected):
    """The flag reaches the config in the shape the config validates."""
    result = runner.invoke(
        app,
        ["build", "sub.cif", "film.cif", *flag, "--outdir", str(tmp_path)],
    )
    assert result.exit_code == 0, _combined(result)
    assert fake_build.config.registry == expected


def test_cli_records_the_registry_in_the_run_config(fake_build, tmp_path):
    """The request is part of the recorded run, so a replay reproduces it.

    A run whose site was chosen is only reproducible if the choice was
    recorded with it.
    """
    from atominterface.run_config import load_run_config

    outdir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "build",
            "sub.cif",
            "film.cif",
            "--registry",
            "on-top",
            "--outdir",
            str(outdir),
        ],
    )
    assert result.exit_code == 0, _combined(result)
    assert load_run_config(outdir / "run_config.yaml")["registry"] == "on-top"


def test_cli_rejects_a_site_name_it_cannot_resolve(fake_build, tmp_path):
    """A bad name fails with the vocabulary, not with a stack trace."""
    result = runner.invoke(
        app,
        [
            "build",
            "sub.cif",
            "film.cif",
            "--registry",
            "trough",
            "--outdir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    text = _combined(result)
    assert "auto" in text and "on-top" in text and "hcp" in text
    assert "0.5,0.5" in text


def test_cli_rejects_scan_as_reserved_rather_than_unknown(fake_build, tmp_path):
    """The reserved name says why it is refused, not just that it is.

    Otherwise `--registry scan` reads as a typo and a user who wanted the
    enumeration concludes it does not exist.
    """
    result = runner.invoke(
        app,
        ["build", "sub.cif", "film.cif", "--registry", "scan", "--outdir", str(tmp_path)],
    )
    assert result.exit_code == 2
    text = _combined(result)
    assert "not implemented" in text
    assert "candidate per distinguishable site" in text
    assert "auto" in text and "on-top" in text and "hcp" in text


@pytest.mark.parametrize("flag", ["0.5", "0.5,0.5,0.5", "a,b"])
def test_cli_rejects_a_malformed_uv(fake_build, tmp_path, flag):
    result = runner.invoke(
        app,
        ["build", "sub.cif", "film.cif", "--registry", flag, "--outdir", str(tmp_path)],
    )
    assert result.exit_code == 2
    assert "registry" in _combined(result)


def _boards_of(candidate):
    """(substrate coords, film coords, in-plane cell) of a built candidate."""
    atoms = candidate.atoms
    tags = np.asarray(atoms.arrays[SLAB_ID], dtype=np.int64)
    positions = np.asarray(atoms.positions, dtype=np.float64)
    cell = np.asarray(atoms.cell, dtype=np.float64)[:2, :2]
    return positions[tags == 0], positions[tags == 1], cell
