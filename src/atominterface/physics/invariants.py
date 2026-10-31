"""Physics invariance checks and transformation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


def rotate_positions(positions, R):
    return positions @ R.T


def rotate_stress(stress, R):
    return R @ stress @ R.T


def check_force_consistency(
    energy_fn: Callable[[np.ndarray], float],
    forces: np.ndarray,
    positions: np.ndarray,
    delta: float = 1e-3,
) -> float:
    """Largest deviation between given forces and the energy surface.

    Computes ``max_i |F_i - (-dE/dr_i)|`` where ``-dE/dr_i`` is estimated
    by a central finite difference of ``energy_fn`` along every Cartesian
    direction of every atom.  The return value is the maximum absolute
    force residual, in the same units as ``forces`` (eV/A when the energy
    is in eV and positions in A).

    Typical usage -- check that a model's forces really are the gradient
    of its own energies::

        check_force_consistency(
            lambda r: calc.get_potential_energy(atoms.copy(), r),
            calc_forces,
            positions,
        )

    Threshold guidance (empirical, not a contract): a float32 uMLIP
    typically shows residuals around 1e-2 eV/A; a float64 model or an
    analytic potential should sit near 1e-6.  A residual far above the
    expected scale means the forces and the energy do not come from the
    same potential.

    Args:
        energy_fn: Receives an ``(N, 3)`` positions array, returns a
            scalar energy.  It must not mutate the array it is given.
        forces: The ``(N, 3)`` force array to verify (e.g. from a model).
        positions: The ``(N, 3)`` coordinates at which the forces were
            evaluated.  Never mutated: shifted copies are passed to
            ``energy_fn``.
        delta: Finite-difference step in the units of ``positions``.

    Returns:
        ``max_i |F_i - (-dE/dr_i)|``; ``0.0`` for empty input.

    Note:
        Signature changed in 0.5.1.  The old signature accepted an
        ``energy`` scalar and returned a constant ``0.0`` -- it verified
        nothing and had no real consumers.
    """
    f = np.asarray(forces, dtype=float)
    r = np.asarray(positions, dtype=float)
    if f.size == 0 or r.size == 0:
        return 0.0

    # Central difference per atom per Cartesian direction.  ``r.copy()``
    # keeps the caller's array untouched, and a fresh copy per probe
    # keeps a mutating ``energy_fn`` from contaminating later probes.
    fd = np.zeros_like(r)
    for i in range(r.shape[0]):
        for a in range(3):
            r_plus = r.copy()
            r_minus = r.copy()
            r_plus[i, a] += delta
            r_minus[i, a] -= delta
            fd[i, a] = (float(energy_fn(r_plus)) - float(energy_fn(r_minus))) / (2.0 * delta)
    return float(np.max(np.abs(f + fd)))


def check_stress_symmetry(stress):
    if stress.shape != (3, 3):
        return False
    return np.allclose(stress, stress.T, atol=1e-10)


def rotation_matrix(axis, angle):
    """Return a 3x3 rotation matrix for rotation about axis by angle (degrees).

    The result is the standard right-handed (active, counter-clockwise looking
    down the axis towards the origin) rotation, identical to what
    ``ase.Atoms.rotate`` applies.  Concretely, ``rotation_matrix("z", 90)``
    maps ``+x`` onto ``+y``.

    The matrix acts on **column** vectors in the usual maths sense
    (``R @ v``).  Because the rest of this module works in the row-vector
    world, positions are rotated as ``positions @ R.T``, which is the same
    transformation.  Use :func:`rotate_positions`, :func:`transform_forces`
    and :func:`transform_stress_voigt` rather than multiplying by ``R``
    directly, so forces and stress keep co-rotating with the coordinates.

    .. versionchanged:: 0.5.2
       The quaternion's vector part used to be negated, which silently
       produced ``R(-angle)`` -- a matrix of the opposite handedness.  Nothing
       inside the package consumed it, and ``rotate_positions`` used ``R.T``,
       so the two sign flips cancelled and the geometry came out right.  The
       danger was at the boundary: passing an ASE rotation (for example the
       ZSL ``match.rotation``) into these helpers would have mirrored the
       cell.  The matrix is now the same object ASE produces.
    """
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
    b, c, d = axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac)],
            [2 * (bc + ad), aa + cc - bb - dd, 2 * (cd - ab)],
            [2 * (bd - ac), 2 * (cd + ab), aa + dd - bb - cc],
        ]
    )


def transform_forces(forces, R):
    """Rotate per-atom forces with the same convention as ``rotate_positions``.

    The code lives in the row-vector world: coordinates transform as
    ``positions @ R.T``.  Forces are polar vectors and co-rotate with the
    rigid body, so they take the same right-multiplication, ``forces @ R.T``.
    (Before 0.5.1 this returned ``forces @ R``, which rotates forces in the
    *opposite* direction to the coordinates under the same ``R``.)
    """
    return forces @ R.T


def transform_stress_voigt(stress_voigt, R):
    s = np.array(
        [
            [stress_voigt[0], stress_voigt[5], stress_voigt[4]],
            [stress_voigt[5], stress_voigt[1], stress_voigt[3]],
            [stress_voigt[4], stress_voigt[3], stress_voigt[2]],
        ]
    )
    s_rot = R @ s @ R.T
    return np.array(
        [
            s_rot[0, 0],
            s_rot[1, 1],
            s_rot[2, 2],
            s_rot[1, 2],
            s_rot[0, 2],
            s_rot[0, 1],
        ]
    )
