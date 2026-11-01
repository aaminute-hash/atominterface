"""Registry placement and measurement for the coherent interface builder."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from atominterface._coherent._constants import (
    _JOIN_ROUND_OFF,
    _OFF_ON_TOP_MIN_SHIFT,
    _PLANE_Z_FLOOR,
    _STACKING_STEP_TOL,
    ON_TOP_LATERAL_TOL,
    PUBLISHED_MILLER,
    REGISTRY_MODES,
    REGISTRY_PLACEMENT_TOL,
)
from atominterface._coherent._util import (
    _pair_label,
    _to_ase,
)
from atominterface.builders.charge import SLAB_ID

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


def _describe_registry(registry: str | Sequence[float] | None) -> str:
    """Name a requested registry the way an advisory message quotes it.

    A mode and an explicit ``(u, v)`` are not two kinds of thing to a
    reader who has to act on the message, so both are spelled out: the
    name alone would leave "where is the join?" unanswered, and a bare
    pair would leave "was this asked for or inherited?" unanswered.
    """
    if registry is None:
        return "registry=None (the site the two slab cuts happen to carry)"
    if isinstance(registry, str):
        return f"registry={registry!r}"
    values = tuple(float(item) for item in registry)
    if len(values) != 2:
        return f"registry={registry!r}"
    return f"registry=(u={values[0]:.6f}, v={values[1]:.6f})"


def _warn_unscanned_orientation(
    config, lateral: float | None = None, registry: str | Sequence[float] | None = None
) -> None:
    """Say out loud that a non-(001) run ships one unscanned registry.

    Three things the caller cannot see from the output otherwise.

    The registry is now *requested* -- :class:`CoherentInterfaceConfig`
    has a ``registry`` field and the join is placed on the site it names
    -- but asking for one site is not scanning the surface. The other
    sites this orientation resolves are never visited, so a gamma_int
    read off this run is a number for the site that was asked for and
    says nothing about the one a scan would have found lower. What the
    cut would have left is no longer what you get either: a film that
    would land exactly on top of a substrate atom is moved off that site
    by the request, so the offset no longer collapses to on-top as the
    thickness changes: what is left is a single site, decided by the
    request rather than by the physics.

    The geometry gate cannot catch any of this and must not be stretched
    to: a non-(001) board can be level, conserve its area and keep a
    healthy in-plane period while sitting on a high-symmetry registry,
    where the in-plane gradient vanishes by symmetry and the relaxation
    stops where it started -- converged, fmax met, OOD green, valid=True,
    and wrong. The (110) ridge site reads 0.5953 J/m^2 against a scanned
    0.088, 6.8x high, with a shortest contact of 2.27 A that no
    proximity check would flag.

    Relaxing longer does not help and is worth saying so: 900 steps left
    the (110) ridge site bit-identical. The fix is to scan the registry
    and quote that minimum, not to relax harder.

    ``lateral`` is the as-built registry in A, measured once per
    assembly. It is quoted so two runs can be checked for being the same
    run before their gamma_int is compared at all.  ``registry`` is what
    the user asked for; it is quoted because "the join sits where you
    asked" and "the join sits where the cut left it" are the two halves
    of this warning, and a reader can only tell them apart if the
    request is named.
    """
    off_001 = [
        f"{side} {tuple(int(index) for index in miller)}"
        for side, miller in (("substrate", config.substrate_miller), ("film", config.film_miller))
        if tuple(int(index) for index in miller) != PUBLISHED_MILLER
    ]
    if not off_001:
        return

    requested = _describe_registry(registry)
    as_built = (
        f"as-built lateral offset: {lateral:.6f} A (in-plane part of the closest "
        "substrate/film pair; two runs are comparable only if this matches). "
        if lateral is not None
        else ""
    )
    warnings.warn(
        f"[orientation] {' and '.join(off_001)}: this orientation is not (001), so the "
        "lateral registry the builder hands over has not been scanned: you asked for "
        f"{requested} and the join has been built on that site, but one requested "
        "site is not a scan -- every other site of this surface has not been scanned, "
        "and what the cut would have left is no longer what you get: a film that would "
        "sit exactly on top of a substrate atom is moved off that site by the request, "
        "so the registry no longer collapses to on-top as the thickness changes: it is "
        "one site, decided by the request rather than by the physics, and two runs "
        f"still have to be checked for being on it before their gamma_int can be "
        f"compared at all. {as_built}"
        "The site itself is not a geometry violation: it is level, conserves its area "
        "and keeps a healthy in-plane period, but on a high-symmetry registry the "
        "in-plane gradient vanishes by symmetry, so a relaxation stops where it "
        "started -- converged, fmax met, OOD green, valid=True, and wrong. (110) shows "
        "what a site is worth: its ridge site reads 0.5953 J/m^2 against a scanned "
        "0.088, 6.8x high, and registry='auto' (the default) asks for the trough "
        "instead -- naming a site is a choice, and this warning cannot make it for "
        "you. A longer relaxation does not fix it: 900 steps left that site "
        "bit-identical, and only a thickness step that accumulated in-plane shear "
        "moved it. Asking for a registry does not lift this warning either: the other "
        "sites are still unvisited, and only an explicit registry scan (registry="
        "'scan', not implemented in this release) would visit them. Run that scan and "
        "quote its minimum before citing gamma_int for a non-(001) orientation. (001) "
        "is the exception: the published protocol enumerated ten candidates itself, "
        "which is why its value can be quoted directly.",
        stacklevel=2,
    )


class _RegistryPair(NamedTuple):
    """The closest substrate/film pair across the join, and how it is offset.

    ``lateral`` -- the in-plane part of their minimum-image separation --
    is the registry.  It is what a registry scan varies and what two runs
    have to match before their gamma_int can be compared, so it is
    measured once per assembly and reused by both consumers rather than
    derived twice by two different routes that could disagree.
    """

    distance: float
    lateral: float
    substrate_index: int
    film_index: int
    du: float
    dv: float


def _closest_cross_pair(interface, n_substrate: int) -> _RegistryPair | None:
    """Closest substrate/film pair in ``interface``, with its lateral offset.

    ``None`` when one side is empty, so there is no join to measure.
    """
    from atominterface.relax.proximity import nearest_image_vectors, pairwise_distance_matrix

    if n_substrate <= 0 or n_substrate >= len(interface):
        return None
    atoms = _to_ase(interface)
    matrix = pairwise_distance_matrix(atoms)
    cross = matrix[:n_substrate, n_substrate:]
    flat = int(np.argmin(cross))
    i, j = divmod(flat, cross.shape[1])
    distance = float(cross[i, j])

    # Lateral part of the minimum-image separation: the pair's offset
    # within the interface plane, with the normal component removed.
    vectors, _ = nearest_image_vectors(atoms, atoms.positions[i])
    cell = np.asarray(interface.lattice.matrix, dtype=np.float64)
    normal = np.cross(cell[0], cell[1])
    normal = normal / np.linalg.norm(normal)
    separation = vectors[n_substrate + j]
    lateral_vector = separation - float(separation @ normal) * normal
    lateral = float(np.linalg.norm(lateral_vector))
    # Rounded to the precision the message prints, then ``+ 0.0`` to turn
    # the -0.0 a tiny negative component leaves behind into +0.0: an
    # exactly opposed pair has to read "du=0.000000", because "du=-0.000000"
    # invites the reader to go looking for an offset that is not there.
    fit = np.linalg.lstsq(cell[:2].T, lateral_vector, rcond=None)[0]
    du = float(np.round(fit[0], 6)) + 0.0
    dv = float(np.round(fit[1], 6)) + 0.0
    return _RegistryPair(distance, lateral, i, n_substrate + j, du, dv)


def _warn_interface_proximity(
    interface, pair: _RegistryPair | None, d_refs, max_strain: float, gap: float
) -> None:
    """Criterion D, advisory only: is the closest cross-interface pair a bond?

    Deliberately a warning and not a gate.  ``config.gap`` (2.0 A by
    default) is *shorter* than a Cu/Ni bond on purpose -- it hands the
    relaxer a slightly compressed join to resolve -- so "closer than a
    bond" is the normal state of a fresh interface and raising here would
    reject every legitimate one.  It is still worth saying out loud,
    because the same signature means two different things: the join the
    caller asked for, or two boards pushed into each other.  The distance
    cannot separate those; the fractional coordinates can, so they are
    what the message leads with.

    Nor is a small distance the worst thing this can be.  A pair that is
    both close *and* laterally undisplaced is on top of its counterpart,
    and an exactly opposed stack sits on a mirror-symmetric stationary
    point where the in-plane gradient vanishes by symmetry -- the
    relaxation converges there, every numerical check stays green, and
    gamma_int comes out 5.9x high on Cu/Ni(111) and 6.3x high on (110).
    That is why the lateral offset is measured and named rather than
    left for the reader to work out.

    The offset is compared against an absolute tolerance, not against a
    fraction of the bond scale: the gradient vanishes only at the
    symmetric point, and a pair a fraction of an angstrom off it is a
    shallow basin rather than a stationary point.  Calling both
    "symmetric" would make the message assert something false.
    """
    measured = [d for d in d_refs if math.isfinite(d)]
    if not measured or pair is None:
        return
    threshold = min(measured) * (1.0 - float(max_strain))
    if pair.distance >= threshold:
        return
    on_top = pair.lateral <= ON_TOP_LATERAL_TOL

    warnings.warn(
        f"[geometry gate] criterion D (interface proximity) on the assembled interface: "
        f"closest substrate/film pair is {pair.distance:.6f} A, below the {threshold:.6f} A "
        f"contact scale (d_ref {min(measured):.6f} A x (1 - {float(max_strain):.6f})); "
        f"the requested gap was {float(gap):.6f} A. "
        f"pair: {_pair_label(interface, pair.substrate_index, pair.film_index)}. "
        f"lateral offset = {pair.lateral:.6f} A (du={pair.du:.6f}, dv={pair.dv:.6f}). "
        "Advisory, and ambiguous on distance alone: a pair sharing u, v whose w straddles "
        "the join is the gap you asked for and will relax apart; a pair inside one board "
        "means the two sides were pushed into each other. Read w before acting."
        + (
            " This pair is exactly opposed -- zero lateral offset, i.e. on-top registry. "
            "Such a stack sits on a mirror-symmetric stationary point whose in-plane "
            "gradient vanishes by symmetry, so a relaxation can converge there and stay: "
            "it relaxes, passes OOD and looks numerically healthy while gamma_int reads "
            "5.9x high on Cu/Ni(111) (at 14.4 A: 7 Cu against 8 Ni layers leaves the join "
            "on-top and the optimizer stops after a single step) and 6.3x high on (110). "
            "Break the symmetry with any one of: shifting the initial registry; changing "
            "the thickness step so both sides get the same layer count (the 7/8 asymmetry "
            "above is what made the on-top site a stationary point); or running an "
            "explicit registry scan."
            if on_top
            else ""
        ),
        stacklevel=2,
    )


def _atomic_planes(z: np.ndarray) -> list[np.ndarray]:
    """Indices of each atomic plane in ``z``, ordered from the bottom up.

    Grouped by a tolerance read off the data itself (half the smallest
    real z step) rather than a fixed constant: a plane is flat, so every
    separation seen below :data:`_PLANE_Z_FLOOR` is scatter inside one
    plane and the rest are steps between them.
    """
    order = np.argsort(z, kind="stable")
    steps = np.diff(z[order])
    real = steps[steps > _PLANE_Z_FLOOR]
    tol = 0.5 * float(np.min(real)) if real.size else 0.0

    planes: list[list[int]] = []
    current = [int(order[0])]
    reference = float(z[current[0]])
    for index in order[1:]:
        if float(z[index]) - reference > tol:
            planes.append(current)
            current = [int(index)]
            reference = float(z[index])
        else:
            current.append(int(index))
    planes.append(current)
    return [np.asarray(plane, dtype=int) for plane in planes]


def _off_on_top_shift(
    sub_coords: np.ndarray, film_coords: np.ndarray, sub_2d: np.ndarray
) -> np.ndarray:
    """In-plane shift that lifts the film off an exactly opposed join (Å).

    Stacking the boards as they come puts the film's join layer on
    whatever sublattice pymatgen's cut left it on and the substrate's on
    the one its own layer count implies -- on fcc (111) that is sublattice
    ``(n_sub - 1) mod 3``, so every third ``--min-thickness`` step lands
    the two on top of each other.  An exactly opposed stack sits on a
    mirror-symmetric stationary point whose in-plane gradient vanishes by
    symmetry, so a relaxation converges there and stays.  Measured on
    Cu/Ni(111): 7 Cu layers at 14.4 A gives gamma_int 0.898 J/m^2 against
    0.081 one thickness step either side.

    The nudge is the substrate's own stacking step -- the in-plane
    offset from its second plane to its top one -- so the film lands on
    the site that continues the substrate's stacking sequence rather
    than on an arbitrary fraction of a cell.  That is the one choice that
    is defined by the geometry at hand and by nothing else.

    Returns zeros when the join is already off the symmetric point, or
    when either board has too few planes to say what "the next one up"
    means.  The caller skips the shift instead of applying a zero one.
    """
    empty = np.zeros(2, dtype=np.float64)
    if not len(sub_coords) or not len(film_coords):
        return empty
    sub_planes = _atomic_planes(sub_coords[:, 2])
    film_planes = _atomic_planes(film_coords[:, 2])
    if len(sub_planes) < 2 or not film_planes:
        return empty

    inv_2d = np.linalg.inv(np.asarray(sub_2d, dtype=np.float64))
    sub_top = sub_coords[sub_planes[-1], :2] @ inv_2d
    sub_below = sub_coords[sub_planes[-2], :2] @ inv_2d
    film_bottom = film_coords[film_planes[0], :2] @ inv_2d

    def _wrapped(delta: np.ndarray) -> np.ndarray:
        return delta - np.round(delta)

    # As-built registry of the join: the closest pair of in-plane offsets
    # between the two facing planes, minimum image, in Å.  This is the
    # same quantity ``_closest_cross_pair`` reports for the stack, just
    # read before the stack exists.
    lateral = min(
        float(np.min(np.linalg.norm(_wrapped(sub_top - point) @ sub_2d, axis=1)))
        for point in film_bottom
    )
    if lateral >= ON_TOP_LATERAL_TOL:
        return empty

    # Forward stacking step: from the nearest atom one plane down to an
    # atom in the top plane.  Wrapped to the shortest representative so
    # the film moves by one step and not by a lattice vector plus one.
    offsets = np.asarray(_wrapped(sub_top[0] - sub_below) @ sub_2d, dtype=np.float64)
    nearest = int(np.argmin(np.linalg.norm(offsets, axis=1)))
    return np.asarray(offsets[nearest], dtype=np.float64)


def _wrap_frac(delta: np.ndarray) -> np.ndarray:
    """Shortest representative of a fractional in-plane offset.

    One lattice vector is as good as no displacement, so every offset is
    reduced to a representative before it is measured, compared or
    applied. Two measured reasons this is load-bearing and not cosmetic:

    * wrapping a *cartesian* difference is meaningless -- the cell is
      generally not cubic, so one lattice vector is not a round number
      of Å -- and it does not raise. It was measured putting a (110)
      film 0.31 of a cell away from the site it was asked for, with
      every downstream check still green and the film sliding during
      relaxation to an energy that had nothing to do with the request.
    * the (001) registry is half a lattice vector, which exists as both
      ``+1/2`` and ``-1/2``. Unwrapped, "move the join onto the site the
      stacking sequence continues to" would move a (001) film by a whole
      lattice vector: a physically identical structure carrying
      different numbers, on the one orientation this package has a DFT
      cross-check for.

    So the order is always fractional -> wrapped -> cartesian, and a
    distance is only ever taken after the third step.
    """
    return delta - np.round(delta)


def _in_plane_frac(vector: np.ndarray, sub_2d: np.ndarray) -> np.ndarray:
    """Cartesian in-plane vector -> fractional coordinates of ``sub_2d``."""
    matrix = np.asarray(sub_2d, dtype=np.float64)
    return np.asarray(np.asarray(vector, dtype=np.float64) @ np.linalg.inv(matrix), float)


def _join_lateral(
    sub_coords: np.ndarray, film_coords: np.ndarray, sub_2d: np.ndarray
) -> np.ndarray | None:
    """As-built in-plane offset of the join: substrate minus film (Å).

    The minimum-image vector from a film atom in the plane facing the
    substrate to the nearest substrate atom in its own facing plane,
    minimised over that film plane; ``None`` when either board is empty
    or has no atomic plane to measure.

    Convert first, wrap second, multiply third: see :func:`_wrap_frac`.
    Reading this number on the two boards before they are stacked is
    what makes targeted placement possible -- it is the same quantity
    :func:`_closest_cross_pair` reports for the finished stack, so the
    request and the measurement cannot drift apart into two different
    definitions of "where the join is".
    """
    if not len(sub_coords) or not len(film_coords):
        return None
    sub_planes = _atomic_planes(sub_coords[:, 2])
    film_planes = _atomic_planes(film_coords[:, 2])
    if not sub_planes or not film_planes:
        return None

    inv_2d = np.linalg.inv(np.asarray(sub_2d, dtype=np.float64))
    sub_top = sub_coords[sub_planes[-1], :2] @ inv_2d
    film_bottom = film_coords[film_planes[0], :2] @ inv_2d
    best: np.ndarray | None = None
    best_norm = math.inf
    for point in film_bottom:
        offsets = np.asarray(_wrap_frac(sub_top - point) @ sub_2d, dtype=np.float64)
        norms = np.linalg.norm(offsets, axis=1)
        nearest = int(np.argmin(norms))
        if float(norms[nearest]) < best_norm:
            best_norm = float(norms[nearest])
            best = np.asarray(offsets[nearest], dtype=np.float64)
    return best


def _stacking_step(sub_coords: np.ndarray, sub_2d: np.ndarray) -> np.ndarray | None:
    """In-plane offset from the substrate's top plane back down one plane.

    Fractional coordinates, wrapped to the shortest representative.
    ``None`` when the board has fewer than two planes, i.e. when "which
    way the sequence is going" is not a question with an answer.

    **Sign.** This is the step *backwards* along the stacking sequence,
    the negative of the forward one :func:`_off_on_top_shift` moves the
    film by, and it is the sign the placement target needs. The two are
    not interchangeable. Written as offsets from an atom in the top
    plane, the site that continues the sequence is at ``+s_up`` (one
    more step up), which is ``-s_down`` subtracted back; the registry
    offset is measured the other way round (substrate minus film), so
    the target that names that site is ``s_down``. On fcc (111) the two
    signs are the difference between the fcc and the hcp hollow, and on
    fcc (110) they coincide only because there every site is half a
    lattice vector -- which is exactly the coincidence that makes a
    wrong sign invisible on the orientation where it was tested.
    """
    if not len(sub_coords):
        return None
    sub_planes = _atomic_planes(sub_coords[:, 2])
    if len(sub_planes) < 2:
        return None
    inv_2d = np.linalg.inv(np.asarray(sub_2d, dtype=np.float64))
    sub_top = sub_coords[sub_planes[-1], :2] @ inv_2d
    sub_below = sub_coords[sub_planes[-2], :2] @ inv_2d
    offsets = _wrap_frac(sub_below - sub_top[0])
    nearest = int(np.argmin(np.linalg.norm(np.asarray(offsets @ sub_2d, float), axis=1)))
    return np.asarray(offsets[nearest], dtype=np.float64)


def _stacking_period(step: np.ndarray) -> int:
    """Layers after which the stacking sequence repeats (0 if none found).

    The number of sublattices the sequence resolves: 2 for a surface
    that alternates, 3 for a close-packed one, 0 for a step that is not
    a rational fraction the search recognises. It decides which site
    names mean anything -- "the other hollow" needs three.
    """
    for count in (1, 2, 3, 4, 6):
        if float(np.max(np.abs(_wrap_frac(count * step)))) < _STACKING_STEP_TOL:
            return count
    return 0


def _sites_of_step(step: np.ndarray, count: int) -> str:
    """Text list of the ``k x step`` sites, in fractional in-plane coords."""
    return ", ".join(
        f"{index}x step -> ({site[0]:+.6f}, {site[1]:+.6f})"
        for index, site in enumerate(_wrap_frac(float(index) * step) for index in range(count))
    )


def _registry_target_frac(
    sub_coords: np.ndarray, sub_2d: np.ndarray, mode: str | Sequence[float]
) -> np.ndarray | None:
    """The site the join is to be placed on, in fractional in-plane coords.

    ``None`` means "this request cannot be resolved for this board",
    which the caller turns into an error: a site that cannot be named is
    not a licence to leave the join where the cut put it.

    The target is a site, not a displacement. That distinction is the
    whole point: the incremental reading ("move the film by the
    stacking step") and this one agree on fcc (111), where the cut
    leaves the film on-top and 0 + step is the fcc hollow, and they
    disagree on fcc (110), where the cut leaves the film on the ridge
    and ridge + step is the short bridge -- 1.5189 J/m^2 against a
    scanned 0.0881, 2.55x the correct target's value.
    """
    if isinstance(mode, str):
        if mode == "on-top":
            return np.zeros(2, dtype=np.float64)
        if mode == "auto":
            step = _stacking_step(sub_coords, sub_2d)
            return None if step is None else np.asarray(_wrap_frac(step), dtype=np.float64)
        if mode == "hcp":
            step = _stacking_step(sub_coords, sub_2d)
            if step is None:
                return None
            period = _stacking_period(step)
            if period != 3:
                raise ValueError(
                    "registry='hcp' names the other hollow of a close-packed surface, "
                    "which is only a different atom when the stacking sequence "
                    "resolves three sublattices. This one's stacking step is "
                    f"(u, v) = ({step[0]:+.6f}, {step[1]:+.6f}) and repeats after "
                    f"{period if period else 'more than 6'} layer(s), so the sites it "
                    f"resolves are: {_sites_of_step(step, max(period, 2))} -- 'hcp' and "
                    "'auto' would land on the same atom. Use registry='auto', "
                    "registry='on-top' or an explicit (u, v)."
                )
            return np.asarray(_wrap_frac(-step), dtype=np.float64)
        raise ValueError(
            f"registry must be one of {list(REGISTRY_MODES)}, or an explicit (u, v) "
            f"fractional in-plane offset; got {mode!r}"
        )
    values = np.asarray([float(item) for item in mode], dtype=np.float64)
    if values.shape != (2,):
        raise ValueError(
            f"registry (u, v) must have exactly 2 components, got {values.size}: {mode!r}"
        )
    return np.asarray(_wrap_frac(values), dtype=np.float64)


def _registry_shift(
    sub_coords: np.ndarray,
    film_coords: np.ndarray,
    sub_2d: np.ndarray,
    mode: str | Sequence[float],
) -> np.ndarray:
    """In-plane displacement that lands the join on the requested site (Å).

    Targeted, so the arithmetic is "where it is minus where it was asked
    to be": after ``film += shift`` the join reads ``d - shift``, and
    ``d - shift = target`` is the condition being solved. The result is
    wrapped to its shortest representative, which is what keeps a (001)
    request from moving the film by a whole lattice vector.

    Raises when either the target or the as-built offset cannot be
    measured. It never returns zeros as a way of saying "I could not
    tell": a join whose site is unknown is reported as unknown, not
    quietly built at the cut's position and shipped as if it were the
    answer.
    """
    target_frac = _registry_target_frac(sub_coords, sub_2d, mode)
    if target_frac is None:
        raise ValueError(
            f"{_describe_registry(mode)} was requested, but the site it names cannot "
            "be resolved for this board: the substrate does not have two atomic "
            "planes to read a stacking step from. Place the join with an explicit "
            "(u, v) offset, or drop the request -- this package does not build a "
            "structure at a site it cannot name."
        )
    as_built = _join_lateral(sub_coords, film_coords, sub_2d)
    if as_built is None:
        raise ValueError(
            f"{_describe_registry(mode)} was requested, but the join's as-built "
            "offset cannot be measured: one of the two boards has no atomic plane at "
            "the join. Nothing is being placed."
        )
    # An offset below the floor is not a site, it is the float scatter of an
    # exactly opposed pair: both boards were cut from the same fractional
    # coordinates and the difference is the 1e-16 of a cartesian round trip.
    # Snapping it to exact zeros before the subtraction is what makes "this
    # request moved the film onto the site the previous release produced" a
    # bit-exact statement instead of one that differs in the last bit.
    measured = np.zeros(2, dtype=np.float64)
    if float(np.linalg.norm(as_built)) >= _JOIN_ROUND_OFF:
        measured = _in_plane_frac(as_built, sub_2d)
    shift_frac = _wrap_frac(measured - target_frac)
    return np.asarray(shift_frac @ sub_2d, dtype=np.float64)


def _check_registry_placement(
    as_built: np.ndarray | None,
    target_frac: np.ndarray | None,
    request: str,
    sub_2d: np.ndarray,
) -> None:
    """Re-measure the join after the move; refuse to ship a miss.

    Placement is verified against a fresh measurement rather than
    trusted, because the failure mode this package exists for is a
    structure that is self-consistent and wrong. The miss it is looking
    for has been observed: a cartesian wrap put a (110) film 0.31 of a
    cell off its requested site, the relaxation converged, fmax was met,
    OOD stayed green and the reported energy belonged to no site anyone
    had asked for.

    The comparison is done in fractional coordinates and wrapped before
    it is subtracted, so half-a-cell and minus-half-a-cell -- the same
    atom -- never read as a whole cell of error.
    """
    if as_built is None or target_frac is None:
        raise ValueError(
            f"{request} was requested, but the join could not be re-measured after "
            "the film was moved, so there is no evidence it sits on the requested "
            "site. Refusing to report it."
        )
    placed_frac = _in_plane_frac(as_built, sub_2d)
    deviation = _wrap_frac(placed_frac - target_frac)
    worst = float(np.max(np.abs(deviation)))
    if worst <= REGISTRY_PLACEMENT_TOL:
        return
    cartesian = np.asarray(deviation @ sub_2d, dtype=np.float64)
    raise ValueError(
        f"{request} was requested, but the join landed "
        f"{float(np.linalg.norm(cartesian)):.6f} A from it and the structure is not "
        "being built: requested site (u, v) = "
        f"({target_frac[0]:+.6f}, {target_frac[1]:+.6f}), measured "
        f"({placed_frac[0]:+.6f}, {placed_frac[1]:+.6f}), deviation "
        f"({deviation[0]:+.3e}, {deviation[1]:+.3e}) fractional. A relaxation would "
        "converge on the wrong site and look healthy doing it."
    )


def _registry_errors(value: str | tuple[float, float] | list[float] | None) -> list[str]:
    """Validation errors for a requested :attr:`CoherentInterfaceConfig.registry`.

    Every rejection names what could have been asked for instead: a
    message that only says "bad registry" leaves the user to guess, and
    the plausible guesses (``trough``, ``ridge``, ``fcc-hollow``) are
    exactly the names this vocabulary deliberately refuses to parse.
    """
    if isinstance(value, str):
        errors: list[str] = []
        if value not in REGISTRY_MODES:
            errors.append(
                f"registry must be one of {list(REGISTRY_MODES)}, or a (u, v) explicit "
                f"fractional in-plane offset, or None; got {value!r}"
            )
        if value == "scan":
            errors.append(
                "registry='scan' is reserved for a later release: it will build one "
                "candidate per distinguishable site of the surface. It is not "
                "implemented yet, so it cannot be requested"
            )
        return errors
    if value is None:
        return []
    try:
        components = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return [
            f"registry must be one of {list(REGISTRY_MODES)}, or a (u, v) pair of "
            f"finite floats, or None; got {value!r}"
        ]
    if len(components) != 2:
        return [f"registry (u, v) must have exactly 2 components, got {len(components)}: {value!r}"]
    if not all(math.isfinite(item) for item in components):
        return [f"registry (u, v) must be finite floats, got {value!r}"]
    return []


def _registry_as_json(value: str | tuple[float, float] | None) -> str | list[float] | None:
    """A requested registry in a form a JSON report can carry.

    Tuples survive ``json.dumps`` as arrays, but a tuple in a metadata
    dict is a value that round-trips into a list and then compares
    unequal to the tuple the config holds; normalising here is what
    keeps a recorded run and its replay identical.
    """
    if value is None or isinstance(value, str):
        return value
    return [float(item) for item in value]


def _frac_or_none(value: Sequence[float] | None) -> list[float] | None:
    """A fractional in-plane pair from the interface properties, or ``None``."""
    if value is None:
        return None
    return [float(item) for item in value]


def _registry_lateral_ang(record: dict[str, Any] | None) -> float | None:
    """Lateral part of a recorded as-built registry, or ``None``."""
    if not record:
        return None
    return float(record["lateral"])


def _place_join(
    sub_coords: np.ndarray,
    film_coords: np.ndarray,
    sub_2d: np.ndarray,
    mode: str | Sequence[float] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Put the join on the site the configuration asks for (Å), then check.

    Mutates ``film_coords[:, :2]`` in place and returns ``(target_frac,
    as_built)`` -- the site that was resolved and the site the join
    measured at afterwards, either of which may be ``None`` when the
    request did not name a site.

    Two placements, and only one of them is a choice of site.
    ``registry=None`` is the v0.4.1 path, kept bit for bit: nudge the
    film off an exactly opposed join and otherwise leave the cut alone.
    Anything else names a site first and solves for the displacement
    that reaches it, which is the only reading that is correct on
    (110) -- there the incremental "add the stacking step" arrives at
    the short bridge, 1.5189 J/m^2 against the trough's 0.0881, while
    the as-built ridge reads 0.5952.

    The check runs on a measurement taken *after* the move, so it would
    catch a placement that solved the wrong problem, and it compares in
    fractional coordinates so that +1/2 and -1/2 -- one atom -- never
    read as a whole cell of error.
    """
    target_frac: np.ndarray | None = None
    if mode is None:
        shift = _off_on_top_shift(sub_coords, film_coords, sub_2d)
    else:
        target_frac = _registry_target_frac(sub_coords, sub_2d, mode)
        shift = _registry_shift(sub_coords, film_coords, sub_2d, mode)
    # A shift below the floor is skipped entirely rather than applied as
    # zero: a write to every film coordinate is a chance to disagree
    # with a pinned baseline in the last bit, and skipping it is what
    # makes "this changed nothing" structural instead of floating-point.
    if float(np.linalg.norm(shift)) >= _OFF_ON_TOP_MIN_SHIFT:
        film_coords[:, :2] += shift
    as_built = _join_lateral(sub_coords, film_coords, sub_2d)
    if target_frac is not None:
        _check_registry_placement(as_built, target_frac, _describe_registry(mode), sub_2d)
    return target_frac, as_built


def _record_registry_properties(
    interface, target_frac: np.ndarray | None, as_built: np.ndarray | None, sub_2d: np.ndarray
) -> None:
    """Write what was asked for and where the join actually is onto the stack.

    Three things a reader needs and only assembly can supply. Kept as
    separate entries because they are three separate claims: a request
    that names a site and a structure that sits on it are different
    facts, and collapsing them into the single number
    ``registry_lateral_offset`` carried before would make a request that
    was not honoured invisible.
    """
    if as_built is not None:
        as_built_frac = _in_plane_frac(as_built, sub_2d)
        interface.properties["registry_as_built"] = {
            "frac": [float(as_built_frac[0]), float(as_built_frac[1])],
            "lateral": float(np.linalg.norm(as_built)),
        }
    if target_frac is not None:
        interface.properties["registry_target_frac"] = [
            float(target_frac[0]),
            float(target_frac[1]),
        ]


def measure_registry(atoms: Any) -> dict[str, Any] | None:
    """Measure the lateral registry an interface currently sits on.

    The same quantity :func:`_assemble_interface` places and records,
    re-measured on whatever structure is handed in -- which is what
    makes it usable *after* a relaxation, and after a relaxation is
    where it matters: the archive holds a Cu/Ni(110) run at 21.6 A that
    slid from the ridge to the trough on its own, so an as-built site
    reported without a final one would let a reader take "where it was
    put" for "where it stopped".

    The two sides are told apart by the ``SLAB_ID`` array every
    assembled interface carries (``0`` substrate, ``1`` film), and the
    measurement is the plane-to-plane one used during assembly rather
    than a second route to the same number: two definitions of "the
    registry" would be two chances to disagree about what a run
    shipped.

    Parameters
    ----------
    atoms : ase.Atoms
        An assembled interface carrying a ``SLAB_ID`` array.

    Returns
    -------
    dict or None
        ``{"frac": [du, dv], "lateral": A}`` -- the join's in-plane
        offset in fractional coordinates of the interface cell and its
        length in Å -- or ``None`` when the registry cannot be
        measured: no ``SLAB_ID`` array, a side missing, or a board with
        no atomic plane at the join. "Not measurable" is a fact and is
        reported as one; it is never turned into a zero.
    """
    arrays = getattr(atoms, "arrays", None)
    if not arrays or SLAB_ID not in arrays:
        return None
    tags = np.asarray(arrays[SLAB_ID], dtype=np.int64)
    substrate = tags == 0
    film = tags == 1
    if not bool(np.any(substrate)) or not bool(np.any(film)):
        return None
    positions = np.asarray(atoms.positions, dtype=np.float64)
    cell = np.asarray(atoms.cell, dtype=np.float64)
    if cell.shape != (3, 3) or abs(float(np.linalg.det(cell[:2, :2]))) <= 1e-12:
        return None
    offset = _join_lateral(positions[substrate], positions[film], cell[:2, :2])
    if offset is None:
        return None
    frac = _in_plane_frac(offset, cell[:2, :2])
    return {
        "frac": [float(frac[0]), float(frac[1])],
        "lateral": float(np.linalg.norm(offset)),
    }
