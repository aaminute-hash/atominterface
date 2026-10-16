"""Physics invariants and unit bookkeeping.

These are deliberately small, dependency-light helpers so they can be
imported in CI without pymatgen or PyTorch.
"""

from __future__ import annotations

import numpy as np

# 1 eV / A^3 = 160.21766208... GPa
EV_PER_A3_TO_GPA: float = 160.21766208
GPA_TO_EV_PER_A3: float = 1.0 / EV_PER_A3_TO_GPA


def rotation_matrix(axis: str = "z", angle_deg: float = 60.0) -> np.ndarray:
    """Return a 3x3 rotation matrix for invariant tests."""
    theta = np.deg2rad(angle_deg)
    if axis == "z":
        return np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    if axis == "x":
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(theta), -np.sin(theta)],
                [0.0, np.sin(theta), np.cos(theta)],
            ]
        )
    raise ValueError(f"unknown axis {axis!r}")


def rotate_atoms_positions(positions: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate a (N,3) array of positions by R (post-multiply)."""
    return positions @ rotation.T


def rotate_cell(cell: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate a lattice cell by R."""
    return cell @ rotation.T


def transform_forces(forces: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Forces are vectors: F' = F R."""
    return forces @ rotation.T


def transform_stress_voigt(stress_voigt: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Transform a Voigt stress by R: S' = R S R^T.

    ``stress_voigt`` is length-6 in Voigt order [xx, yy, zz, yz, xz, xy].
    """
    s = np.array(stress_voigt, dtype=float)
    if s.shape != (6,):
        raise ValueError("stress_voigt must have shape (6,)")
    m = np.array(
        [
            [s[0], s[5], s[4]],
            [s[5], s[1], s[3]],
            [s[4], s[3], s[2]],
        ]
    )
    mprime = rotation @ m @ rotation.T
    return np.array(
        [mprime[0, 0], mprime[1, 1], mprime[2, 2], mprime[1, 2], mprime[0, 2], mprime[0, 1]]
    )


def ev_per_a3_to_gpa(value: float) -> float:
    """Convert an energy density in eV/A^3 to GPa."""
    return value * EV_PER_A3_TO_GPA


def gpa_to_ev_per_a3(value: float) -> float:
    """Convert a pressure in GPa to eV/A^3."""
    return value * GPA_TO_EV_PER_A3
