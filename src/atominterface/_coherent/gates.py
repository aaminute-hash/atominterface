"""Geometry gates for the coherent builder.

Cell-level invariants checked before a board enters assembly: shortest
in-plane period, area conservation and in-plane period recovery.  These
depend only on the cell helpers and the constants, so they are a leaf of
the ``_coherent`` split.
"""

from __future__ import annotations

import math

import numpy as np

from atominterface._coherent._constants import (
    AREA_CONSERVATION_RTOL,
    INPLANE_PERIOD_RANGE,
    INPLANE_PERIOD_RTOL,
)
from atominterface._coherent._util import _pair_label, _to_ase
from atominterface._coherent.cells import _lattice_area, _multiplicity


class GeometryGateError(ValueError):
    """An assembled interface violates a geometry invariant.

    Raised by :func:`_assemble_interface` when a board's in-plane cell
    would be misread downstream.  Every slice the builder chain takes
    (``lattice[:2, :2]``, ``cart_coords[:, 2]``) reads the cell as if
    ``a`` and ``b`` were level and ``c`` carried the vacuum -- which
    :func:`_orient_slab` guarantees for generated boards and nothing
    guarantees for anything else.  A board that skips that step still
    looks fine to the slicer: it just gets a *smaller* cell, and every
    number derived from it (area, layer spacing, gamma) is quietly wrong.

    There is deliberately no way to switch this off.  A geometry that
    cannot be measured correctly cannot be reported correctly either, so
    the only honest answer is to refuse the structure.
    """


def _shortest_inplane_period(in_plane: np.ndarray) -> tuple[float, int, int]:
    """Shortest ``|m*a + n*b|`` over ``m, n`` in ``[-4, 4]``, as ``(length, m, n)``.

    Brute force rather than ``min(|a|, |b|)``: a basis can be reduced in
    area and still carry a *combination* far shorter than either vector --
    ``a = (2.556, 0)``, ``b = (3 * 2.556 + 0.05, 0.3)`` reads
    ``min(|a|, |b|) = 2.556`` but repeats every ``|b - 3 a| = 0.304 A``.
    Reading ``|a|`` alone would call that cell healthy.
    """
    best = (float("inf"), 0, 0)
    for m in range(-INPLANE_PERIOD_RANGE, INPLANE_PERIOD_RANGE + 1):
        for n in range(-INPLANE_PERIOD_RANGE, INPLANE_PERIOD_RANGE + 1):
            if m == 0 and n == 0:
                continue
            length = float(np.linalg.norm(m * in_plane[0] + n * in_plane[1]))
            if length < best[0]:
                best = (length, m, n)
    return best


def _check_area_conservation(side: str, before: np.ndarray, after, transform) -> None:
    """Criterion B: the in-plane block the stack uses spans the board's area.

    Replication by an integer supercell multiplies the area by exactly
    ``|det(transform)|``, and a level board's ``[:2, :2]`` block *is* its
    interface plane -- so the area the assembly writes into the interface
    cell has to equal the unscaled board's area times the multiplicity.
    Any gap between the two says ``a`` or ``b`` carries a z component and
    ``[:2, :2]`` is a projection onto xy rather than the plane itself.

    ``before`` is the pre-scaling lattice matrix and ``transform`` the
    supercell description (``None`` when nothing was scaled).
    """
    multiplicity = 1 if transform is None else _multiplicity(transform)
    expected = _lattice_area(np.asarray(before, dtype=np.float64)) * multiplicity
    if not math.isfinite(expected) or expected <= 0.0:
        return

    matrix = np.asarray(after.lattice.matrix, dtype=np.float64)
    # The two areas are only the same number for a level cell; reported
    # separately so the message says which step lost the area.
    true_area = _lattice_area(matrix)
    used_area = float(abs(np.linalg.det(matrix[:2, :2])))
    deviation = max(abs(true_area - expected), abs(used_area - expected)) / expected
    if deviation <= AREA_CONSERVATION_RTOL:
        return

    raise GeometryGateError(
        f"[geometry gate] criterion B (area conservation) failed on the {side} side. "
        f"measured in-plane block = {used_area:.6f} A^2, board's true cell area = "
        f"{true_area:.6f} A^2, reference = {multiplicity} x {expected / multiplicity:.6f} A^2 "
        f"= {expected:.6f} A^2; relative deviation {deviation:.3e} exceeds "
        f"{AREA_CONSERVATION_RTOL:.1e}. "
        f"a = ({matrix[0][0]:.6f}, {matrix[0][1]:.6f}, {matrix[0][2]:.6f}), "
        f"b = ({matrix[1][0]:.6f}, {matrix[1][1]:.6f}, {matrix[1][2]:.6f}): "
        "a or b carries a z component, so lattice[:2, :2] is the xy projection of the "
        "interface plane instead of the plane itself, and every area, spacing and gamma "
        "derived from this stack is scaled by that projection. Level the board first."
    )


def _check_inplane_period(side: str, slab, max_strain: float) -> float:
    """Criterion C: the in-plane cell must still repeat at a physical period.

    The board's tightest pair sets the scale: nothing in a relaxed board
    sits closer than a nearest-neighbour shell, and the coherent match is
    allowed to compress that by at most ``max_strain``.  A period below
    ``d_ref * (1 - max_strain)`` means the cell the stack will repeat was
    squeezed past what the chemistry allows -- which is what a projected
    in-plane block does, since a projection can only shorten vectors.

    Returns the ``d_ref`` it measured so the caller can reuse the scale
    (the proximity warning needs it); NaN when the board is too small to
    establish one.
    """
    from atominterface.relax.proximity import pairwise_min_distance

    if len(slab) < 2:
        return float("nan")
    d_ref, i, j = pairwise_min_distance(_to_ase(slab))
    if not math.isfinite(d_ref):
        return float("nan")

    matrix = np.asarray(slab.lattice.matrix, dtype=np.float64)
    period, m, n = _shortest_inplane_period(matrix[:2, :2])
    threshold = d_ref * (1.0 - float(max_strain))
    if period >= threshold * (1.0 - INPLANE_PERIOD_RTOL):
        return d_ref

    raise GeometryGateError(
        f"[geometry gate] criterion C (in-plane period) failed on the {side} side. "
        f"measured shortest in-plane period = {period:.6f} A "
        f"({m}*a + {n}*b, over m, n in [{-INPLANE_PERIOD_RANGE}, {INPLANE_PERIOD_RANGE}]), "
        f"threshold = {threshold:.6f} A = d_ref {d_ref:.6f} A x (1 - {float(max_strain):.6f}). "
        f"d_ref is the tightest pair in the board: {_pair_label(slab, i, j)}. "
        "A short period can be a compressed cell or an intended small gap -- read the "
        "fractional coordinates: a pair with the same u, v whose w straddles the join is "
        "the gap you asked for, a pair inside one plane is a cell that was squeezed."
    )
