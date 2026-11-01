"""Per-frame Green-Lagrange strain and Cauchy stress from ASE trajectories.

This is the data pipeline behind Phase 0.4 (AutoConstitutive): it turns a
sequence of simulation cells into the two constitutive channels a fitting
routine needs -- Green-Lagrange strain and Cauchy stress -- together with the
instantaneous temperature and the cell/volume history.

Conventions (fixed, do not change without updating ``transform_stress_voigt``):

* **Voigt order** is ``[xx, yy, zz, yz, xz, xy]`` -- the same order ASE uses in
  ``Atoms.get_stress(voigt=True)`` and the order consumed by
  :func:`atominterface.physics.invariants.transform_stress_voigt`.
* **Shear components are the tensor components, not the engineering (2x)
  components.**  ``E_xy`` is ``E[0, 1]`` and ``sigma_yz`` is ``sigma[1, 2]``,
  with no factor of two.  This is called out on every conversion helper so the
  engineering-Voigt convention is never assumed silently.
* **Units.** Cauchy stress is returned in eV/A^3 (and, separately, GPa via
  ``EV_A3_TO_GPA``); Green-Lagrange strain is dimensionless (fractional,
  ``0.05`` means 5 %); temperature is kelvin.  The :class:`StressStrainTable`
  carries an explicit ``units`` dict.

``extract`` is a pure, side-effect-free pipeline: it never imports or touches
``atominterface.relax`` and consumes any ASE trajectory-like object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Explicit unit labels attached to every table.  Kept as a module constant so
# the labels are stable across calls and easy to test.
_DEFAULT_UNITS: dict[str, str] = {
    "strain_voigt": "dimensionless (fraction; Green-Lagrange)",
    "cauchy_stress_voigt": "eV/A^3",
    "cauchy_stress_voigt_gpa": "GPa",
    "temperature": "K",
    "cells": "Angstrom",
    "volumes": "Angstrom^3",
    "reference_cell": "Angstrom",
}


@dataclass
class StressStrainTable:
    """Strain/stress channels extracted from a trajectory.

    Every row ``i`` corresponds to trajectory frame ``i``.  Channels that are
    absent from *all* frames are ``None``; a channel present on some frames
    uses ``NaN`` for the frames where it is missing (``has_stress`` is the
    authoritative per-frame flag for stress).
    """

    strain_voigt: np.ndarray  # (N, 6)
    cauchy_stress_voigt: np.ndarray | None  # (N, 6), eV/A^3
    cauchy_stress_voigt_gpa: np.ndarray | None  # (N, 6), GPa
    temperature: np.ndarray | None  # (N,), K
    cells: np.ndarray  # (N, 3, 3)
    volumes: np.ndarray  # (N,)
    has_stress: np.ndarray  # (N,) bool
    reference_cell: np.ndarray  # (3, 3)
    units: dict[str, str]
    n_frames: int


def _as_3x3(matrix, name: str) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must be a (3, 3) array, got shape {arr.shape}")
    return arr


def green_lagrange_strain(cell, cell_ref) -> np.ndarray:
    """Green-Lagrange strain ``E = 0.5 (F^T F - I)`` with ``F = cell @ inv(cell_ref)``.

    ``cell`` and ``cell_ref`` are the (3, 3) row-vector lattice matrices of the
    deformed and reference configurations.  ``F`` maps reference coordinates
    ``X`` to current coordinates ``x = F @ X``.  The result is a symmetric 3x3
    fractional strain tensor.
    """
    cell = _as_3x3(cell, "cell")
    cell_ref = _as_3x3(cell_ref, "cell_ref")
    F = cell @ np.linalg.inv(cell_ref)
    return np.asarray(0.5 * (F.T @ F - np.eye(3)), dtype=float)


def strain_to_voigt(E) -> np.ndarray:
    """Flatten a 3x3 symmetric strain tensor to ``[xx, yy, zz, yz, xz, xy]``.

    The shear components are the **tensor components** (no factor of two), so
    ``E_xy`` maps to ``E[0, 1]`` and ``E_yz`` maps to ``E[1, 2]``.
    """
    E = _as_3x3(E, "E")
    return np.array([E[0, 0], E[1, 1], E[2, 2], E[1, 2], E[0, 2], E[0, 1]])


def virial_to_cauchy(virial, volume) -> np.ndarray:
    """Convert a (3, 3) virial to Cauchy stress, ``sigma = -virial / volume``.

    ASE stores the virial as the negative of the stress times volume, so this
    recovers the per-volume Cauchy stress with the correct sign.
    """
    virial = _as_3x3(virial, "virial")
    return np.asarray(-virial / float(volume), dtype=float)


def cauchy_to_voigt(sigma) -> np.ndarray:
    """Flatten a 3x3 symmetric stress tensor to ``[xx, yy, zz, yz, xz, xy]``.

    The shear components are the **tensor components** (no factor of two), so
    ``sigma_yz`` maps to ``sigma[1, 2]`` and ``sigma_xy`` maps to
    ``sigma[0, 1]`` -- matching ASE's ``get_stress(voigt=True)`` order.
    """
    sigma = _as_3x3(sigma, "sigma")
    return np.array([sigma[0, 0], sigma[1, 1], sigma[2, 2], sigma[1, 2], sigma[0, 2], sigma[0, 1]])


def _voigt_to_matrix(v) -> np.ndarray:
    """Inverse of :func:`cauchy_to_voigt` / :func:`strain_to_voigt` (symmetric)."""
    v = np.asarray(v, dtype=float)
    return np.array(
        [
            [v[0], v[5], v[4]],
            [v[5], v[1], v[3]],
            [v[4], v[3], v[2]],
        ]
    )


def _as_frame_list(trajectory) -> list:
    """Normalise the three accepted trajectory inputs into a list of Atoms."""
    if isinstance(trajectory, (str, Path)):
        from ase.io import read

        return list(read(str(trajectory), index=":"))
    if isinstance(trajectory, (list, tuple)):
        return list(trajectory)
    # A single Atoms is accepted as a one-frame sequence for convenience.
    if hasattr(trajectory, "get_positions") and hasattr(trajectory, "get_cell"):
        return [trajectory]
    try:
        return list(trajectory)
    except TypeError as exc:
        raise TypeError(
            "trajectory must be a file path, an ase.io.Trajectory, or a sequence of Atoms"
        ) from exc


def _extract_stress(atoms) -> np.ndarray | None:  # noqa: PLR0911 - normalise 6/3x3/virial shapes
    """Return the per-frame Cauchy stress in Voigt order, or ``None``.

    Primary path is ``Atoms.get_stress(voigt=True)`` (ASE already returns
    Cauchy stress).  Depending on the ASE version/calculator, that call may
    hand back a 6-component Voigt vector or a full 3x3 tensor; both are
    normalised here.  On failure it falls back to a virial from ``atoms.info``
    or the calculator.
    """
    try:
        raw = atoms.get_stress(voigt=True)
    except Exception:
        raw = None
    if raw is not None:
        arr = np.asarray(raw, dtype=float)
        if arr.shape == (6,):
            return arr
        if arr.shape == (3, 3):
            return cauchy_to_voigt(arr)
        return None

    virial = _get_virial(atoms)
    if virial is None:
        return None
    virial = np.asarray(virial, dtype=float)
    volume = float(atoms.get_volume())
    if virial.shape == (3, 3):
        return cauchy_to_voigt(virial_to_cauchy(virial, volume))
    if virial.shape == (6,):
        return cauchy_to_voigt(virial_to_cauchy(_voigt_to_matrix(virial), volume))
    return None


def _get_virial(atoms):
    """Best-effort virial from ``atoms.info`` or the attached calculator."""
    info = getattr(atoms, "info", None)
    if info is not None and "virial" in info:
        return info["virial"]
    calc = getattr(atoms, "calc", None)
    if calc is not None:
        get_virial = getattr(calc, "get_virial", None)
        if get_virial is not None:
            try:
                return get_virial(atoms)
            except Exception:
                return None
    return None


def _extract_temperature(atoms) -> float | None:
    """Instantaneous temperature in kelvin, or ``None`` when no momenta exist.

    ASE's ``get_temperature`` silently reports 0 K for a static frame (it gets
    zero velocities back), so the absence of kinetic energy is detected up
    front by looking for a ``velocities``/``momenta`` array rather than by the
    numeric value.
    """
    arrays = getattr(atoms, "arrays", None)
    if arrays is None or ("velocities" not in arrays and "momenta" not in arrays):
        return None
    try:
        temperature = atoms.get_temperature()
    except Exception:
        return None
    if temperature is None:
        return None
    temperature = float(temperature)
    if np.isnan(temperature):
        return None
    return temperature


def _stack_stress(rows: list[np.ndarray | None], n: int) -> np.ndarray:
    arr = np.full((n, 6), np.nan)
    for i, row in enumerate(rows):
        if row is not None:
            arr[i] = row
    return arr


def _stack_temperature(rows: list[float | None], n: int) -> np.ndarray:
    arr = np.full(n, np.nan)
    for i, row in enumerate(rows):
        if row is not None:
            arr[i] = row
    return arr


def extract(trajectory, *, reference_cell=None) -> StressStrainTable:
    """Extract strain/stress/temperature channels from an ASE trajectory.

    Args:
        trajectory: One of a file path (``str``/``Path``, read with
            ``ase.io.read(path, index=':')``), an ``ase.io.Trajectory``, or a
            sequence of ``Atoms`` (``list``/``tuple``).
        reference_cell: Optional (3, 3) reference lattice.  Defaults to the
            first frame's cell, so the first frame always has zero strain.

    Returns:
        A :class:`StressStrainTable`.  Per-frame missing values are ``NaN``; a
        whole channel is ``None`` only when no frame carries it.

    Raises:
        ValueError: if the trajectory has zero frames.
    """
    frames = _as_frame_list(trajectory)
    if len(frames) == 0:
        raise ValueError("cannot extract stress/strain from an empty trajectory (0 frames)")

    ref_cell = _as_3x3(
        frames[0].cell if reference_cell is None else reference_cell,
        "reference_cell",
    )

    n = len(frames)
    strain_voigt = np.empty((n, 6), dtype=float)
    cells = np.empty((n, 3, 3), dtype=float)
    volumes = np.empty(n, dtype=float)
    has_stress = np.zeros(n, dtype=bool)
    stress_rows: list[np.ndarray | None] = [None] * n
    temperature_rows: list[float | None] = [None] * n
    any_stress = False
    any_temperature = False

    for i, atoms in enumerate(frames):
        cell = np.asarray(atoms.cell, dtype=float)
        cells[i] = cell
        volumes[i] = float(atoms.get_volume())
        strain_voigt[i] = strain_to_voigt(green_lagrange_strain(cell, ref_cell))

        stress = _extract_stress(atoms)
        if stress is not None:
            stress_rows[i] = stress
            has_stress[i] = True
            any_stress = True

        temperature_val = _extract_temperature(atoms)
        if temperature_val is not None:
            temperature_rows[i] = temperature_val
            any_temperature = True

    cauchy_stress_voigt: np.ndarray | None = None
    cauchy_stress_voigt_gpa: np.ndarray | None = None
    if any_stress:
        cauchy_stress_voigt = _stack_stress(stress_rows, n)
        # Lazy import avoids a circular import with ``physics/__init__``.
        from atominterface.physics import EV_A3_TO_GPA

        cauchy_stress_voigt_gpa = cauchy_stress_voigt * EV_A3_TO_GPA

    temperature: np.ndarray | None = None
    if any_temperature:
        temperature = _stack_temperature(temperature_rows, n)

    return StressStrainTable(
        strain_voigt=strain_voigt,
        cauchy_stress_voigt=cauchy_stress_voigt,
        cauchy_stress_voigt_gpa=cauchy_stress_voigt_gpa,
        temperature=temperature,
        cells=cells,
        volumes=volumes,
        has_stress=has_stress,
        reference_cell=ref_cell,
        units=dict(_DEFAULT_UNITS),
        n_frames=n,
    )


__all__ = [
    "StressStrainTable",
    "cauchy_to_voigt",
    "extract",
    "green_lagrange_strain",
    "strain_to_voigt",
    "virial_to_cauchy",
]
