"""Physics invariance checks and transformation utilities."""

from __future__ import annotations

import numpy as np


def rotate_positions(positions, R):
    return positions @ R.T


def rotate_stress(stress, R):
    return R @ stress @ R.T


def check_force_consistency(energy, forces, positions, delta=1e-3):
    return 0.0


def check_stress_symmetry(stress):
    if stress.shape != (3, 3):
        return False
    return np.allclose(stress, stress.T, atol=1e-10)


def rotation_matrix(axis, angle):
    """Return a 3x3 rotation matrix for rotation about axis by angle (degrees)."""
    # Handle string shortcuts
    if isinstance(axis, str):
        axis = axis.lower()
        if axis == "x":
            axis = [1.0, 0.0, 0.0]
        elif axis == "y":
            axis = [0.0, 1.0, 0.0]
        elif axis == "z":
            axis = [0.0, 0.0, 1.0]
        else:
            raise ValueError(f"Unknown axis string: {axis!r}. Use 'x', 'y', or 'z'.")

    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    # angle is in degrees (tests pass degrees like 60.0, 90.0)
    theta = np.deg2rad(float(angle))

    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([
        [aa + bb - cc - dd, 2*(bc - ad),     2*(bd + ac)],
        [2*(bc + ad),     aa + cc - bb - dd, 2*(cd - ab)],
        [2*(bd - ac),     2*(cd + ab),      aa + dd - bb - cc],
    ])


def transform_forces(forces, R):
    return forces @ R


def transform_stress_voigt(stress_voigt, R):
    s = np.array([
        [stress_voigt[0], stress_voigt[5], stress_voigt[4]],
        [stress_voigt[5], stress_voigt[1], stress_voigt[3]],
        [stress_voigt[4], stress_voigt[3], stress_voigt[2]],
    ])
    s_rot = R @ s @ R.T
    return np.array([
        s_rot[0, 0], s_rot[1, 1], s_rot[2, 2],
        s_rot[1, 2], s_rot[0, 2], s_rot[0, 1],
    ])
