"""Periodic-aware proximity helpers for the relax guard layer.

A bare cartesian difference silently misses neighbours that live across
a cell boundary.  That blind spot is not academic here: an interface
slab is periodic in the two in-plane directions, so the overlap a
relaxation produces at the cell edge is exactly the one a cartesian
check cannot see -- and the atom that trips the guard is then the atom
a cluster cut drops.

Every distance computed in this module therefore goes through the
minimum-image convention for whichever directions are actually
periodic.  The work is delegated to :func:`ase.geometry.find_mic`,
which Minkowski-reduces the cell first, so skewed (non-orthogonal)
cells are handled and mixed ``pbc`` masks such as ``[True, True,
False]`` are respected direction by direction.
"""

from __future__ import annotations

import numpy as np
from ase.geometry import find_mic

__all__ = [
    "minimum_image_distances",
    "nearest_neighbor_distances",
    "nearest_image_vectors",
    "pairwise_distance_matrix",
    "pairwise_min_distance",
]


def _pbc_mask(atoms) -> np.ndarray:
    """``atoms.pbc`` as a length-3 boolean array (accepts a scalar)."""
    return np.broadcast_to(np.asarray(atoms.pbc, dtype=bool), (3,))


def minimum_image_distances(atoms, i: int) -> np.ndarray:
    """Distances from atom ``i`` to every atom, honouring PBC.

    Entry ``i`` itself is ``inf`` so callers can take ``.min()`` without
    special-casing the self-distance.  A system with fewer than two
    atoms yields all ``inf``.
    """
    n = len(atoms)
    out = np.full(n, np.inf)
    if n < 2:
        return out
    others = np.delete(np.arange(n), i)
    pos = np.asarray(atoms.positions, dtype=float)
    _, lengths = find_mic(pos[others] - pos[i], atoms.cell, pbc=_pbc_mask(atoms))
    out[others] = np.asarray(lengths, dtype=float)
    return out


def nearest_image_vectors(atoms, origin) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-image displacement of every atom from ``origin``.

    Returns ``(vectors, lengths)``: ``vectors[j]`` is the shortest
    periodic image of ``positions[j] - origin`` and ``lengths[j]`` its
    norm.  Used to unwrap a local cluster into one contiguous ball.
    """
    pos = np.asarray(atoms.positions, dtype=float)
    vectors, lengths = find_mic(
        pos - np.asarray(origin, dtype=float), atoms.cell, pbc=_pbc_mask(atoms)
    )
    return np.asarray(vectors, dtype=float), np.asarray(lengths, dtype=float)


def pairwise_distance_matrix(atoms, *, block: int = 64) -> np.ndarray:
    """Full MIC distance matrix; diagonal set to ``inf``.

    Built in row blocks so a large slab never materialises an
    ``n * n * 3`` float array in one go.  The result is symmetrised:
    ``d(i, j)`` and ``d(j, i)`` are reached from opposite displacement
    vectors and can differ by an ulp, which would otherwise make
    ``argmin``/``argmax`` over the matrix pick whichever of a tied pair
    it happened to scan first.
    """
    pos = np.asarray(atoms.positions, dtype=float)
    n = len(pos)
    dm: np.ndarray = np.full((n, n), np.inf)
    if n < 2:
        return dm
    pbc = _pbc_mask(atoms)
    for start in range(0, n, block):
        stop = min(start + block, n)
        delta = pos[None, :, :] - pos[start:stop, None, :]
        _, lengths = find_mic(delta.reshape(-1, 3), atoms.cell, pbc=pbc)
        dm[start:stop] = np.asarray(lengths, dtype=float).reshape(stop - start, n)
    dm = np.minimum(dm, dm.T)
    np.fill_diagonal(dm, np.inf)
    return dm


def nearest_neighbor_distances(atoms) -> np.ndarray:
    """Distance from each atom to its nearest neighbour (``inf`` if alone)."""
    return np.asarray(pairwise_distance_matrix(atoms).min(axis=1), dtype=float)


def pairwise_min_distance(atoms) -> tuple[float, int, int]:
    """Tightest pair in the cell: ``(distance, i, j)`` with ``i < j``.

    ``(inf, 0, 0)`` for a system with fewer than two atoms.
    """
    dm = pairwise_distance_matrix(atoms)
    if len(dm) < 2:
        return float("inf"), 0, 0
    flat = int(np.argmin(dm))
    i, j = divmod(flat, len(dm))
    # Ties are broken by ``argmin``'s scan order; normalise so callers
    # always get the lower index first.
    if i > j:
        i, j = j, i
    return float(dm[i, j]), int(i), int(j)
