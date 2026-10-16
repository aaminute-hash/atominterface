"""Strain / lattice-mismatch helpers.

These are deliberately geometry-only routines: no database lookups,
no models.  They exist so the builder can reject candidates **before**
spending seconds on structure generation.

pymatgen is treated as an optional dependency: it is imported lazily
so the builder package stays importable in minimal CI environments,
but every public routine here is typed against its Lattice because
the strain math is meaningless without a proper periodicity object.
"""

from __future__ import annotations

import numpy as np

# Sentinel used for type annotations; the real class is resolved at
# runtime by :func:`_lattice_cls` so we never pay the import cost at
# module load time.
Lattice = None  # type: ignore  # noqa: F811  # re-export for typing


def _lattice_cls():  # noqa: F811
    """Return the pymatgen Lattice class, importing it on first use."""
    try:
        from pymatgen.core.lattice import Lattice as _L  # noqa: N811, WPS433
    except ImportError:  # pragma: no cover - optional dependency
        raise ImportError(
            "pymatgen is required for strain calculations. "
            "Install it with: pip install 'atominterface[pymatgen]'"
        )
    return _L


def _normalize(obj):
    """Accept a pymatgen Lattice or a raw 3x3 / 2x2 / 2x3 array.

    The returned object is always a pymatgen Lattice, which guarantees
    that downstream code can safely access ``.matrix`` and ``.abc``.
    """
    _L = _lattice_cls()
    if isinstance(obj, _L):
        return obj
    arr = np.asarray(obj, dtype=np.float64)
    if arr.shape == (2, 2):
        # In-plane 2x2 slice -> pad a unit vacuum vector along z.
        mat = np.eye(3, dtype=np.float64)
        mat[:2, :2] = arr
        return _L(mat)
    if arr.shape == (2, 3):
        # Two in-plane vectors already carrying a z component.
        mat = np.eye(3, dtype=np.float64)
        mat[:2, :] = arr
        return _L(mat)
    return _L(arr)


def compute_mismatch(
    lattice_a,
    lattice_b,
    direction: tuple[int, int, int] = (0, 0, 1),
) -> float:
    """Volumetric mismatch between two lattices along ``direction``.

    Uses the 2D in-plane lattice vectors normal to ``direction``,
    following the usual definition:

        mismatch = 2 * |a - b| / (a + b)

    Returns the **maximum** mismatch among the two in-plane vectors,
    which is the conservative quantity for interface stability.
    """
    lattice_a = _normalize(lattice_a)
    lattice_b = _normalize(lattice_b)

    a = np.array(lattice_a.abc, dtype=float)
    b = np.array(lattice_b.abc, dtype=float)
    idx = [i for i in range(3) if i != _normal_axis(direction)]
    mismatches = [2.0 * abs(a[i] - b[i]) / (a[i] + b[i] + 1e-12) for i in idx]
    return float(max(mismatches))


def max_shear_strain(
    lattice_a,
    lattice_b,
    direction: tuple[int, int, int] = (0, 0, 1),
) -> float:
    """Shear-strain proxy from the angle between in-plane vectors.

    Large shear is what makes an interface relax into a mess even
    when the length mismatch looks small.  Returned in radians.
    """
    a1, a2 = _inplane_vectors(lattice_a, direction)
    b1, b2 = _inplane_vectors(lattice_b, direction)
    ca = _cos_angle(a1, b1)
    cb = _cos_angle(a2, b2)
    # Angle deviation from ideal registry; clip to avoid acos(>1).
    dev = float(np.arccos(min(max(ca, -1.0), 1.0)) + np.arccos(min(max(cb, -1.0), 1.0)))
    return dev


def _normal_axis(direction: tuple[int, int, int]) -> int:
    for i, v in enumerate(direction):
        if v != 0:
            return i
    raise ValueError("direction must be a non-zero Miller index")


def _inplane_vectors(lattice, direction: tuple[int, int, int]) -> tuple:
    m = np.array(_normalize(lattice).matrix, dtype=float)
    basis = [m[i] for i in range(3) if i != _normal_axis(direction)]
    return tuple(basis)


def _cos_angle(u: np.ndarray, v: np.ndarray) -> float:
    un = u / (np.linalg.norm(u) + 1e-12)
    vn = v / (np.linalg.norm(v) + 1e-12)
    return float(np.dot(un, vn))
