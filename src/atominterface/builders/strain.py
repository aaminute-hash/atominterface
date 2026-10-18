"""Strain metrics for interface lattice matching.

Two conventions live in this module and they are **not** interchangeable:

:func:`compute_mismatch`
    Full three-dimensional Green--Lagrange strain between two periodic
    lattices.  Correct when both cells are genuinely periodic along all
    three axes.

:func:`in_plane_mismatch`
    Two-dimensional engineering strain between the *in-plane* blocks of
    two surface cells, with the third axis treated as vacuum.  **This is
    the metric interface builders must use.**  The vacuum axis carries no
    material, so folding it into a 3D deviatoric decomposition invents
    shear that does not exist: a pure 5 % isotropic in-plane stretch
    would otherwise be reported as 3.4 % shear, and every well-matched
    candidate would be rejected by a 5 % tolerance.

Both functions report **fractional** strain: ``0.05`` means 5 %.
"""

from __future__ import annotations

import numpy as np

_LATTICE_CLS = None

# Below this |det| the in-plane vectors are treated as degenerate.
_DEGENERATE = 1e-12


def _lattice_cls():
    # Cached lazy import of the optional pymatgen dependency; the module-level
    # cache is intentional so repeated calls do not retry a failed import.
    global _LATTICE_CLS  # noqa: PLW0603
    if _LATTICE_CLS is None:
        try:
            from pymatgen.core.lattice import Lattice

            _LATTICE_CLS = Lattice
        except ImportError:
            _LATTICE_CLS = False
    return _LATTICE_CLS


def _as_pymatgen_lattice(matrix):
    Lattice = _lattice_cls()
    if not Lattice:
        raise ImportError("pymatgen not available")
    arr = np.asarray(matrix, dtype=float)
    if arr.shape == (3, 3):
        return Lattice(arr)
    elif arr.shape == (2, 2):
        m = np.eye(3)
        m[:2, :2] = arr
        return Lattice(m)
    elif arr.shape == (2, 3):
        m = np.eye(3)
        m[0, :2] = arr[0, :2]
        m[1, :2] = arr[1, :2]
        if arr.shape[1] == 3:
            m[0, 2] = arr[0, 2]
            m[1, 2] = arr[1, 2]
        return Lattice(m)
    else:
        raise ValueError(f"Cannot convert shape {arr.shape} to Lattice")


def _to_matrix(obj):
    """Extract 3x3 matrix from Lattice object, array, or list."""
    if hasattr(obj, "matrix"):
        return np.asarray(obj.matrix, dtype=float)
    elif hasattr(obj, "a") and hasattr(obj, "b") and hasattr(obj, "c"):
        # pymatgen Lattice-like duck type
        return np.array([obj.a, obj.b, obj.c], dtype=float)
    else:
        return np.asarray(obj, dtype=float)


def _in_plane_block(obj) -> np.ndarray:
    """Return the 2x2 in-plane block of a surface cell.

    The interface normal is assumed to be the third Cartesian axis -- the
    convention every builder in this package enforces
    (``BuilderConfig.vacuum_dir == 2``).  The first two lattice vectors
    are therefore the in-plane cell vectors and the third points into the
    vacuum.
    """
    arr = _to_matrix(obj)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:  # noqa: PLR2004
        raise ValueError(f"expected at least two lattice vectors, got shape {arr.shape}")
    return np.asarray(arr[:2, :2], dtype=float)


def compute_mismatch(lat_source, lat_target):
    """Compute full 3D tensile and shear strain between two lattices.

    The deformation gradient is ``F = lat_target @ inv(lat_source)``, i.e.
    the strain required to deform ``lat_source`` onto ``lat_target``.
    Only use this when both cells are periodic along **all three** axes;
    for surface cells where the third axis is vacuum, use
    :func:`in_plane_mismatch` instead.
    """
    sub = _to_matrix(lat_source)
    film = _to_matrix(lat_target)
    try:
        F = film @ np.linalg.inv(sub)
    except np.linalg.LinAlgError:
        return {
            "tensile": float("inf"),
            "shear": float("inf"),
            "von_mises": float("inf"),
            "strain_tensor": np.full((3, 3), float("inf")),
        }
    E = 0.5 * (F.T @ F - np.eye(3))
    tensile = np.trace(E) / 3.0
    dev = E - tensile * np.eye(3)
    shear = np.sqrt(2.0 / 3.0 * np.trace(dev @ dev))
    von_mises = np.sqrt(3.0 / 2.0 * np.trace(dev @ dev))
    return {
        "tensile": float(tensile),
        "shear": float(shear),
        "von_mises": float(von_mises),
        "strain_tensor": E,
    }


def in_plane_mismatch(lat_source, lat_target):
    """Two-dimensional strain required to deform ``lat_source`` onto ``lat_target``.

    The deformation gradient ``F = lat_target @ inv(lat_source)`` is
    evaluated on the in-plane 2x2 block only, then split with the
    small-strain (engineering) tensor ``eps = (F + F.T) / 2 - I``:

    ``tensile``
        Half the trace of ``eps``: the average in-plane normal strain.
        A pure 5 % isotropic stretch reports ``0.05``.
    ``shear``
        Spread of the in-plane principal strains, ``eps_1 - eps_2``.
        This is the maximum engineering shear strain in the interface
        plane, so an isotropic stretch reports ``0`` and a pure in-plane
        shear ``gamma`` reports ``gamma``.

    Parameters
    ----------
    lat_source, lat_target
        Lattices or matrices; at least two vectors, third axis is the
        interface normal.  Argument order matters: the interface builder
        passes ``(film, substrate)`` because the film is the cell that
        gets deformed.

    Returns
    -------
    dict
        ``tensile``, ``shear``, ``max_principal``, ``min_principal`` and
        the 2x2 ``strain_tensor``.  Degenerate in-plane vectors yield
        ``inf`` rather than an exception.
    """
    sub = _in_plane_block(lat_source)
    film = _in_plane_block(lat_target)

    if min(abs(float(np.linalg.det(sub))), abs(float(np.linalg.det(film)))) < _DEGENERATE:
        return _degenerate_mismatch()

    F = film @ np.linalg.inv(sub)
    eps = 0.5 * (F + F.T) - np.eye(2)
    principal = np.linalg.eigvalsh(eps)

    return {
        "tensile": 0.5 * float(np.trace(eps)),
        "shear": float(principal[-1] - principal[0]),
        "max_principal": float(principal[-1]),
        "min_principal": float(principal[0]),
        "strain_tensor": eps,
    }


def _degenerate_mismatch() -> dict:
    """Sentinel result for a collapsed in-plane cell."""
    return {
        "tensile": float("inf"),
        "shear": float("inf"),
        "max_principal": float("inf"),
        "min_principal": float("-inf"),
        "strain_tensor": np.full((2, 2), float("inf")),
    }


def max_shear_strain(lat_source, lat_target):
    """Maximum engineering shear strain in the interface plane."""
    return in_plane_mismatch(lat_source, lat_target)["shear"]


def max_stretch_strain(lat_source, lat_target):
    """Largest in-plane principal normal strain (absolute value)."""
    result = in_plane_mismatch(lat_source, lat_target)
    return max(abs(result["max_principal"]), abs(result["min_principal"]))


def compute_area(vec_a, vec_b):
    """Compute area of parallelogram spanned by two vectors."""
    return float(np.linalg.norm(np.cross(np.asarray(vec_a), np.asarray(vec_b))))


def strain_filter(candidates, max_strain=0.05):
    """Filter candidate interfaces by maximum allowed strain."""
    filtered = []
    for c in candidates:
        strain = getattr(c, "strain", None)
        if strain is None:
            filtered.append(c)
        elif isinstance(strain, dict):
            # Accept either the 2D or the 3D dict shape.
            worst = max(
                abs(float(strain.get("tensile", 0.0))),
                float(strain.get("shear", 0.0)),
            )
            if worst <= max_strain:
                filtered.append(c)
        elif float(strain) <= max_strain:
            filtered.append(c)
    return filtered
