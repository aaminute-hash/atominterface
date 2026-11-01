from __future__ import annotations

import numpy as np

# Absolute tolerance (Å) below which a cell component counts as nothing
# when deciding whether a slab already satisfies the "+z normal" contract
# enforced by :func:`_orient_slab`.  Cell lengths are a few Å, so this is
# ~1e-10 relative -- far above float64 round-off, far below any physical
# or numerical quantity downstream.
_ORIENT_TOL = 1e-9


def _orient_slab(slab):
    """Return a slab whose surface normal is exactly ``+z``, ``a``/``b`` in ``xy``.

    ``SlabGenerator(primitive=True)`` returns cells whose in-plane vectors
    carry a z component for every surface but (001): fcc (111) comes back
    as ``a = (1.81, 0, -1.81)``, ``b = (0, 1.81, -1.81)``.  Every
    downstream ``[:2]`` slice then projects those vectors onto the xy
    plane, which silently shrinks the interface area by ``sqrt(3)``
    (3.2671 instead of 5.6587 Å²) and lets ``_assemble_interface`` rebuild
    a hexagonal cell as a square one.  Rotating once here turns that
    assumption into a fact for every consumer further down the chain
    instead of patching six separate slices.

    The rotation is applied to the orthogonal-c form so that ``c`` really
    is parallel to ``cross(a, b)`` -- clearing the in-plane part of a skew
    ``c`` would otherwise discard ~40 Å of real cell vector.

    A slab that already satisfies the contract skips the rotation and comes
    back with its coordinates untouched, which is what keeps the published
    (001) reference numbers pinned bit for bit.
    """
    from pymatgen.core import Structure
    from pymatgen.core.lattice import Lattice as _Lattice

    matrix = np.asarray(slab.lattice.matrix, dtype=np.float64)
    already_aligned = max(
        abs(float(matrix[0, 2])),
        abs(float(matrix[1, 2])),
        abs(float(matrix[2, 0])),
        abs(float(matrix[2, 1])),
    )
    # This tests the invariant itself -- whether a and b are level and c
    # carries no in-plane part -- not the Miller indices, and not any
    # guess about which surface is being built.  When the test passes the
    # rotation below is the identity, so skipping it changes nothing.  Do
    # not read this as a "(001) fast path": it fires for any already-level
    # cell from any source, and it must not be reused as an
    # orientation-recognition heuristic anywhere else.
    if already_aligned <= _ORIENT_TOL:
        # Copied rather than handed back as-is, so a caller cannot mutate
        # the generator's own slab through this result.  ``properties`` is
        # re-applied explicitly because pymatgen's ``copy()`` drops it --
        # which would silently strip the finite-slab provenance flag the
        # charge layer reads.
        aligned = slab.copy()
        aligned.properties.update(getattr(slab, "properties", {}) or {})
        return aligned

    orth = slab.get_orthogonal_c_slab()  # c perpendicular to a, b
    L = np.asarray(orth.lattice.matrix, dtype=np.float64)
    n = np.cross(L[0], L[1])
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    e1 = L[0] / np.linalg.norm(L[0])
    e2 = np.cross(n, e1)
    e2 = e2 / np.linalg.norm(e2)
    rot = np.vstack([e1, e2, n])  # rows are the new basis vectors, in old coordinates
    new_L = L @ rot.T
    new_L[2, :2] = 0.0  # defensive: drop any residual in-plane part of c
    coords = np.asarray(orth.cart_coords, dtype=np.float64) @ rot.T
    out = Structure(
        _Lattice(new_L),
        list(orth.species),
        coords,
        coords_are_cartesian=True,
        to_unit_cell=False,
        site_properties={key: list(values) for key, values in orth.site_properties.items()},
    )
    # Provenance comes from the *input*, not from ``orth``:
    # ``get_orthogonal_c_slab`` rebuilds its own structure and drops
    # ``properties`` on the floor, so reading them off it would silently
    # strip the ``ai_slab_generated`` flag the charge layer certifies
    # finite boards by.
    out.properties.update(getattr(slab, "properties", {}) or {})
    return out
