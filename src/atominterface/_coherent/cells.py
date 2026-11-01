"""In-plane cell helpers for the coherent builder.

Area accounting, supercell transforms and the in-plane strain machinery.
Depends only on :mod:`atominterface._coherent._constants` and pymatgen,
so it is a leaf of the ``_coherent`` split.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np

from atominterface._coherent._constants import AREA_POLICIES


def _lattice_area(lattice: np.ndarray) -> float:
    """Area of the interface plane (Å²)."""
    a = lattice[0]
    b = lattice[1]
    return float(np.linalg.norm(np.cross(a, b)))


@dataclass(frozen=True)
class InterfaceAreas:
    """The two sides' interface areas plus the one to normalise by.

    ``area_substrate`` / ``area_film`` are the in-plane areas (Å²) each
    side actually occupies -- the pair a "bilateral" energy ledger needs
    in order to say which side a normalisation came from.  A coherent
    stack shares one cell, so the pair is normally equal; it diverges as
    soon as one side is left unstrained.

    ``area_used`` is the single number a downstream energy division
    should take, resolved from ``policy``:

    ``"strict"``
        Refuse to choose when the sides disagree: ``area_used`` is NaN
        and ``note`` reports both numbers.
    ``"substrate"`` / ``"film"``
        Always take that side, disagreement or not.

    A geometric mean ``sqrt(a_sub * a_film)`` is deliberately **not**
    offered.  It would turn a real mismatch into a plausible-looking
    number, i.e. hide exactly the thing this record exists to expose.
    """

    area_substrate: float
    area_film: float
    area_used: float
    consistent: bool
    policy: str
    note: str


def _in_plane_vectors(lattice) -> np.ndarray:
    """First two lattice rows as 2x3 cartesian vectors.

    Accepts a 3x3 or 2x3 matrix, a 2x2 in-plane slice (padded with a
    zero z component) or anything exposing a pymatgen-style ``matrix``.
    """
    matrix = getattr(lattice, "matrix", lattice)
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape == (2, 2):
        padded = np.zeros((2, 3), dtype=np.float64)
        padded[:, :2] = mat
        return padded
    return mat[:2, :]


def interface_areas(
    substrate_lattice,
    film_lattice,
    *,
    tolerance: float = 1e-6,
    policy: str = "strict",
) -> InterfaceAreas:
    """Resolve the interface area of a two-sided stack.

    ``tolerance`` is a *relative* window on ``abs(a_sub - a_film)``
    against the larger of the two, so the verdict is scale-free.
    ``policy`` must be one of :data:`AREA_POLICIES`.

    Nothing here raises on a mismatch: an inconsistent pair is a fact to
    record, and deciding what to do with such a candidate belongs to the
    caller.  Only an unknown ``policy`` raises, because it is a
    programming error rather than a property of the structure.
    """
    if policy not in AREA_POLICIES:
        raise ValueError(f"area_policy must be one of {list(AREA_POLICIES)}, got {policy!r}")

    a_sub = _lattice_area(_in_plane_vectors(substrate_lattice))
    a_film = _lattice_area(_in_plane_vectors(film_lattice))
    scale = max(a_sub, a_film)
    consistent = bool(abs(a_sub - a_film) <= tolerance * scale) if scale > 0 else True

    if policy == "film":
        area_used = a_film
    elif policy == "substrate" or consistent:
        area_used = a_sub
    else:
        area_used = float("nan")

    if consistent:
        note = f"both sides {a_sub:.6g} Å²; area taken from the {policy} policy"
    elif policy == "strict":
        note = (
            f"area mismatch: substrate {a_sub:.6g} Å² vs film {a_film:.6g} Å²; "
            "no single area is valid and no average is substituted"
        )
    else:
        note = (
            f"area mismatch: substrate {a_sub:.6g} Å² vs film {a_film:.6g} Å²; "
            f"area taken from the {policy} side by policy"
        )

    return InterfaceAreas(
        area_substrate=a_sub,
        area_film=a_film,
        area_used=area_used,
        consistent=consistent,
        policy=policy,
        note=note,
    )


def _as_pymatgen_lattice(obj):
    """Normalize a lattice argument to a pymatgen Lattice."""
    from pymatgen.core.lattice import Lattice  # noqa: WPS433

    if isinstance(obj, Lattice):
        return obj
    arr = np.asarray(obj, dtype=np.float64)
    if arr.ndim == 2 and arr.shape == (2, 2):
        # In-plane 2x2 slice -> pad a unit vacuum vector along z.
        mat = np.eye(3, dtype=np.float64)
        mat[:2, :2] = arr
        return Lattice(mat)
    if arr.ndim == 2 and arr.shape == (2, 3):
        # Two in-plane vectors already carrying a z component.
        mat = np.eye(3, dtype=np.float64)
        mat[:2, :] = arr
        return Lattice(mat)
    return Lattice(arr)


def _supercell_transform(matrix: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Round a ZSL 2x2 transformation matrix to its integer form.

    ZSL returns matrices whose entries are integers up to floating-point
    noise.  Returning the rounded matrix (instead of, say, its diagonal
    entries) keeps rotated supercells honest: a 45-degree cell has
    off-diagonal entries that carry the whole meaning.  Use
    :func:`_supercell_multiplicity` for the number of primitive cells it
    contains.
    """
    rounded = np.rint(np.asarray(matrix, dtype=np.float64)).astype(int)
    return (
        (int(rounded[0, 0]), int(rounded[0, 1])),
        (int(rounded[1, 0]), int(rounded[1, 1])),
    )


def _multiplicity(transform) -> int:
    """Number of primitive cells in a supercell description.

    Accepts either the integer 2x2 matrix produced by
    :func:`_supercell_transform` or a legacy flat ``(m, n)`` pair.
    """
    rows = tuple(transform)
    if len(rows) == 2 and all(np.isscalar(row) for row in rows):
        return max(abs(int(rows[0]) * int(rows[1])), 1)
    (a, b), (c, d) = rows
    return max(abs(int(a) * int(d) - int(b) * int(c)), 1)


def _format_matrix(matrix) -> str:
    """Format a 2x2 supercell matrix as an unambiguous, filename-safe string.

    The determinant alone cannot identify a supercell (det = 1, 2, 3 covers
    both ``1x1``/``2x2``/``3x3`` diagonal cells and rotated cells such as
    ``sqrt(2) x sqrt(2) R45deg``), which is how duplicate ``interface_id``
    values used to collide.  The full integer matrix is written with ``_``
    delimiters so no two distinct matrices can produce the same label
    (``((11, 0), (0, 1))`` and ``((1, 10), (0, 1))`` stay distinguishable).
    """
    dense = np.rint(_dense_transform(matrix)).astype(int)
    return "_".join(str(int(value)) for value in dense.ravel())


def _dense_transform(transform) -> np.ndarray:
    """2x2 float form of a supercell description."""
    rows = tuple(transform) if transform is not None else ()
    if len(rows) == 2 and all(np.isscalar(row) for row in rows):
        return np.diag([float(rows[0]), float(rows[1])])
    return np.asarray(rows, dtype=np.float64)


def _scale_in_plane(structure, transform):
    """Replicate a slab by its integer in-plane supercell.

    Two guards matter here:

    * the third row of the scaling matrix is fixed to 1 so the slab
      thickness (and hence the vacuum budget) is untouched;
    * the slab is copied before scaling.  Newer pymatgen versions default
      ``make_supercell`` to ``in_place=True``, which silently turned the
      candidate loop into a compounding supercell (6.5 -> 85 -> 1104 A^2
      within four candidates) because every candidate re-scaled the same
      slab object.
    """
    mat = np.rint(_dense_transform(transform)).astype(int)
    scaling = np.array(
        [
            [mat[0, 0], mat[0, 1], 0],
            [mat[1, 0], mat[1, 1], 0],
            [0, 0, 1],
        ],
        dtype=int,
    )
    slab = structure.copy()
    if "in_place" in inspect.signature(slab.make_supercell).parameters:
        slab = slab.make_supercell(scaling, in_place=False)
    else:
        slab = slab.make_supercell(scaling)
    # Slab.copy() in some pymatgen versions preserves site properties but
    # drops Structure.properties, including the finite-slab provenance.
    slab.properties.update(structure.properties)
    return slab


def _strain_film_onto(film_slab, substrate_slab_2d):
    """Strain ``film_slab`` in-plane onto the substrate cell.

    Returns a new ``Structure`` whose in-plane lattice *is*
    ``substrate_slab_2d`` and whose cartesian coordinates carry the
    matching deformation ``sub_2d @ inv(film_2d)`` applied to the
    in-plane components (``coords @ deformation.T``).  Because the same
    map acts on the coordinates and on the cell, the fractional
    coordinates -- the stacking sequence -- are preserved.

    Shared by :func:`_assemble_interface` and :func:`reference_slabs`
    so an interface and the reference boards it is referenced against
    are strained by one implementation, not two approximations.  The
    slab's vacuum column along z is carried over untouched and the
    input is never modified.
    """
    from pymatgen.core import Structure
    from pymatgen.core.lattice import Lattice as _Lattice

    film_mat = np.asarray(film_slab.lattice.matrix, dtype=np.float64)
    # ``[:2, :2]`` is the true in-plane block because the z components of a
    # and b are zero -- guaranteed by ``_orient_slab`` for every slab that
    # comes out of :func:`_get_terminations`.
    film_2d = np.asarray(film_mat[:2, :2], dtype=np.float64)
    sub_2d = np.asarray(substrate_slab_2d, dtype=np.float64)

    deformation = sub_2d @ np.linalg.inv(film_2d)
    coords = np.asarray(film_slab.cart_coords, dtype=np.float64).copy()
    coords[:, :2] = coords[:, :2] @ deformation.T

    # Clean z axis, mirroring the cell ``_assemble_interface`` builds:
    # the in-plane rows carry no z component and the film keeps its own
    # vacuum column, so a strained reference board stays centred.
    rows = np.zeros((3, 3), dtype=np.float64)
    rows[0, :2] = sub_2d[0]
    rows[1, :2] = sub_2d[1]
    rows[2, 2] = film_mat[2, 2] if film_mat[2, 2] > 0 else 1.0

    site_properties = {key: list(values) for key, values in film_slab.site_properties.items()}
    strained = Structure(
        _Lattice(rows),
        list(film_slab.species),
        coords,
        coords_are_cartesian=True,
        to_unit_cell=False,
        site_properties=site_properties or None,
    )
    strained.properties.update(film_slab.properties)
    return strained
