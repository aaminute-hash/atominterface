"""Strain computation for interface matching."""

from __future__ import annotations

import numpy as np

_LATTICE_CLS = None

def _lattice_cls():
    global _LATTICE_CLS
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
    if hasattr(obj, 'matrix'):
        return np.asarray(obj.matrix, dtype=float)
    elif hasattr(obj, 'a') and hasattr(obj, 'b') and hasattr(obj, 'c'):
        # pymatgen Lattice-like duck type
        return np.array([obj.a, obj.b, obj.c], dtype=float)
    else:
        return np.asarray(obj, dtype=float)


def compute_mismatch(lat_sub, lat_film):
    """Compute tensile and shear strain between two lattices."""
    sub = _to_matrix(lat_sub)
    film = _to_matrix(lat_film)
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


def max_shear_strain(lat_sub, lat_film):
    """Return the maximum shear strain between two lattices."""
    result = compute_mismatch(lat_sub, lat_film)
    return result["shear"]


def max_stretch_strain(lat_sub, lat_film):
    """Return the maximum stretch (tensile) strain between two lattices."""
    result = compute_mismatch(lat_sub, lat_film)
    return abs(result["tensile"])


def compute_area(vec_a, vec_b):
    """Compute area of parallelogram spanned by two vectors."""
    return float(np.linalg.norm(np.cross(np.asarray(vec_a), np.asarray(vec_b))))


def strain_filter(candidates, max_strain=0.05):
    """Filter candidate interfaces by maximum allowed strain."""
    filtered = []
    for c in candidates:
        if hasattr(c, 'strain') and c.strain is not None:
            if isinstance(c.strain, dict):
                if c.strain.get("von_mises", 0) <= max_strain:
                    filtered.append(c)
            else:
                if float(c.strain) <= max_strain:
                    filtered.append(c)
        else:
            filtered.append(c)
    return filtered
