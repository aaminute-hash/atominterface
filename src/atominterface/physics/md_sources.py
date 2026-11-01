"""Bridge real molecular-dynamics trajectories into ASE ``Atoms`` frames.

This module connects :mod:`atominterface.physics.stress_strain` to two real MD
data sources -- a LAMMPS classical run and a CP2K AIMD run -- so
``virial_to_cauchy`` / ``extract`` can be validated against production data
rather than synthetic trajectories alone.

Conventions (fixed; mirrored by the Phase 0.4 validation record):

* **Sign.** LAMMPS and CP2K both report a *pressure* tensor that is positive
  under compression.  Cauchy stress is positive under tension, so the bridge
  negates: ``sigma = -P``.
* **Units.** Both sources report pressure in ``bar``.  The bridge converts
  ``bar -> eV/A^3`` with ``1e-4 / EV_A3_TO_GPA`` and stores the
  resulting Cauchy stress in eV/A^3 (the unit ``extract`` consumes).
* **Kinetic term.** The full Cauchy stress includes the ideal-gas (kinetic)
  contribution.  For LAMMPS this is the thermo ``pxx..pyz`` columns (NOT the
  ``compute pressure NULL virial`` values, which are potential-only); for CP2K
  it is the ``.stress`` file (which already includes the kinetic term).  The
  bridge never reconstructs a virial from absolute coordinates: ``sum(r_i x
  f_i)`` is translation-dependent for a periodic system with a nonzero
  instantaneous net force, and disagrees with the software's pair virial by
  2-5 GPa.
* **Velocities.** LAMMPS ``metal`` units report velocities in A/ps while ASE
  stores velocities in a unit where ``0.5 * m * v^2`` is already in eV, so the
  conversion is ``v_ase = v_lammps * sqrt(mvv2e)``.

Both readers return ``list[ase.Atoms]``, one frame per MD step, each carrying a
:class:`ase.calculators.singlepoint.SinglePointCalculator` whose ``stress`` is
the full Cauchy stress as a symmetric 3x3 matrix in eV/A^3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

# LAMMPS "metal" units: E[eV] = mvv2e * m[amu] * v^2[A/ps].  ASE's velocity
# unit satisfies 0.5 * m[amu] * v_ase^2 = E[eV], so v_ase = v[A/ps] * sqrt(mvv2e).
LAMMPS_MVV2E = 1.0364269e-4


def _bar_to_ev_a3() -> float:
    """One bar expressed in eV/A^3 (``1e-4 / EV_A3_TO_GPA``).

    ``EV_A3_TO_GPA`` is imported lazily to avoid a circular import: the
    ``atominterface.physics`` package re-exports this module, so importing the
    constant at module level would run before the package finishes loading.
    """
    from atominterface.physics import EV_A3_TO_GPA

    return 1e-4 / EV_A3_TO_GPA


def _voigt_to_matrix(voigt) -> np.ndarray:
    """Voigt ``[xx, yy, zz, yz, xz, xy]`` -> symmetric 3x3 matrix."""
    xx, yy, zz, yz, xz, xy = voigt
    return np.array(
        [
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz],
        ]
    )


def _parse_lammps_dump(path) -> list[tuple[int, np.ndarray, np.ndarray | None, np.ndarray]]:
    """Parse a LAMMPS dump text file into ``(step, positions, velocities, cell)``.

    ``positions`` are shifted by the box origin so they lie in ``[0, L]`` as ASE
    expects; ``velocities`` are ``None`` when the dump omits ``vx vy vz``.  Only
    orthorhombic boxes are supported (a triclinic tilt factor raises
    ``ValueError``).
    """
    lines = Path(path).read_text().splitlines()
    frames: list[tuple[int, np.ndarray, np.ndarray | None, np.ndarray]] = []
    i = 0
    n_lines = len(lines)
    while i < n_lines:
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        timestep = int(lines[i + 1].strip())
        i += 2
        if not lines[i].startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError(f"malformed LAMMPS dump near line {i}: expected NUMBER OF ATOMS")
        n_atoms = int(lines[i + 1].strip())
        i += 2
        if not lines[i].startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"malformed LAMMPS dump near line {i}: expected BOX BOUNDS")
        i += 1
        bounds = []
        for _ in range(3):
            parts = lines[i].split()
            # Orthorhombic bound lines carry exactly "lo hi"; a third column is
            # a triclinic tilt factor, which this reader does not support.
            if len(parts) != 2:
                raise ValueError(f"only orthorhombic LAMMPS boxes are supported (line {i})")
            bounds.append((float(parts[0]), float(parts[1])))
            i += 1
        if not lines[i].startswith("ITEM: ATOMS"):
            raise ValueError(f"malformed LAMMPS dump near line {i}: expected ITEM: ATOMS")
        header = lines[i].split()
        columns = header[2:]  # drop "ITEM:" and "ATOMS"
        col_index = {name: idx for idx, name in enumerate(columns)}
        missing = [c for c in ("x", "y", "z") if c not in col_index]
        if missing:
            raise ValueError(f"LAMMPS dump is missing atom columns: {missing}")
        i += 1
        positions = np.empty((n_atoms, 3))
        velocities: np.ndarray | None = None
        if all(c in col_index for c in ("vx", "vy", "vz")):
            velocities = np.empty((n_atoms, 3))
        for a in range(n_atoms):
            parts = lines[i + a].split()
            positions[a] = [float(parts[col_index[c]]) for c in ("x", "y", "z")]
            if velocities is not None:
                velocities[a] = [float(parts[col_index[c]]) for c in ("vx", "vy", "vz")]
        i += n_atoms

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        cell = np.diag(hi - lo)
        positions = positions - lo  # bring the box origin to (0, 0, 0)
        frames.append((timestep, positions, velocities, cell))
    return frames


def _parse_lammps_thermo(path) -> dict[int, np.ndarray]:
    """Parse a LAMMPS ``log`` thermo table into ``{step: pressure_voigt (bar)}``.

    ``pressure_voigt`` is the full pressure tensor in LAMMPS order
    ``[xx, yy, zz, xy, xz, yz]`` (compression positive, kinetic term included).
    """
    out: dict[int, np.ndarray] = {}
    for line in Path(path).read_text().splitlines():
        tokens = line.split()
        # A thermo data row has exactly 18 columns:
        # Step Temp Press Pxx Pyy Pzz Pxy Pxz Pyz c_myp[1..6] KinEng PotEng Volume
        if len(tokens) != 18:
            continue
        try:
            step = int(tokens[0])
        except ValueError:
            continue
        pxx = float(tokens[3])
        pyy = float(tokens[4])
        pzz = float(tokens[5])
        pxy = float(tokens[6])
        pxz = float(tokens[7])
        pyz = float(tokens[8])
        out[step] = np.array([pxx, pyy, pzz, pxy, pxz, pyz])
    return out


def read_lammps_md(dump_path, thermo_path, *, mass=63.546, mvv2e=LAMMPS_MVV2E) -> list[Atoms]:
    """Read a LAMMPS ``dump`` + ``log`` pair into ASE frames with Cauchy stress.

    The stress is the full Cauchy stress (tension positive, eV/A^3) taken from
    the thermo ``pxx..pyz`` columns (which include the kinetic term), negated
    and unit-converted: ``sigma = -P * 1e-4 / EV_A3_TO_GPA``.  Velocities are
    converted from LAMMPS metal A/ps to ASE units via ``sqrt(mvv2e)`` and stored with a
    uniform per-atom ``mass`` (amu), so ``Atoms.get_temperature()`` is
    meaningful.

    Args:
        dump_path: LAMMPS dump text file (``id type x y z vx vy vz fx fy fz``).
        thermo_path: LAMMPS ``log.lammps`` thermo output (18-column rows).
        mass: Per-atom mass in amu (LAMMPS dumps carry no element identity).
        mvv2e: LAMMPS metal-units ``mvv2e`` constant (eV per amu*(A/ps)^2).

    Returns:
        One ``Atoms`` per dump frame that has a matching thermo ``step`` (frames
        without a matching thermo row are skipped, since they carry no stress).

    Raises:
        ValueError: if the dump is empty, or no dump frame has a matching thermo
            ``step``.
    """
    frames = _parse_lammps_dump(dump_path)
    thermo = _parse_lammps_thermo(thermo_path)
    if not frames:
        raise ValueError("no frames found in LAMMPS dump")
    if not thermo:
        raise ValueError("no thermo rows found in LAMMPS log")

    velocity_scale = np.sqrt(float(mvv2e))
    bar_to_ev_a3 = _bar_to_ev_a3()
    out: list[Atoms] = []
    for timestep, positions, velocities, cell in frames:
        if timestep not in thermo:
            continue
        # thermo voigt is LAMMPS order [xx, yy, zz, xy, xz, yz]; reorder to the
        # package Voigt order [xx, yy, zz, yz, xz, xy] while negating and
        # converting bar -> eV/A^3.
        pressure = thermo[timestep]  # bar, compression positive
        sigma = -pressure * bar_to_ev_a3
        sigma_voigt = np.array([sigma[0], sigma[1], sigma[2], sigma[5], sigma[4], sigma[3]])

        atoms = Atoms(positions=positions, cell=cell, pbc=True)
        atoms.set_masses([mass] * len(positions))
        if velocities is not None:
            atoms.set_velocities(velocities * velocity_scale)
        atoms.calc = SinglePointCalculator(atoms, stress=_voigt_to_matrix(sigma_voigt))
        out.append(atoms)
    if not out:
        raise ValueError("no dump frames have a matching thermo step")
    return out


def _parse_cp2k_stress(path) -> dict[int, np.ndarray]:
    """Parse a CP2K ``.stress`` file into ``{step: 3x3 pressure (bar)}``.

    The 11 columns are ``step time xx xy xz yx yy yz zx zy zz``; the tensor is
    symmetric, so it is symmetrized defensively.
    """
    out: dict[int, np.ndarray] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) != 11:
            continue
        step = int(tokens[0])
        xx = float(tokens[2])
        xy = float(tokens[3])
        xz = float(tokens[4])
        yx = float(tokens[5])
        yy = float(tokens[6])
        yz = float(tokens[7])
        zx = float(tokens[8])
        zy = float(tokens[9])
        zz = float(tokens[10])
        p = np.array([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]])
        out[step] = 0.5 * (p + p.T)
    return out


def _parse_cp2k_cell(path) -> dict[int, np.ndarray]:
    """Parse a CP2K ``.cell`` file into ``{step: 3x3 cell (A, row vectors)}``.

    The 12 columns are ``step time Ax Ay Az Bx By Bz Cx Cy Cz Volume``.
    """
    out: dict[int, np.ndarray] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) != 12:
            continue
        step = int(tokens[0])
        a = [float(tokens[2]), float(tokens[3]), float(tokens[4])]
        b = [float(tokens[5]), float(tokens[6]), float(tokens[7])]
        c = [float(tokens[8]), float(tokens[9]), float(tokens[10])]
        out[step] = np.array([a, b, c])
    return out


def _parse_cp2k_ener(path) -> dict[int, float]:
    """Parse a CP2K ``.ener`` file into ``{step: temperature (K)}``.

    The 7 columns are ``step time kin temp pot cons_qty used_time``.
    """
    out: dict[int, float] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) < 7:
            continue
        try:
            step = int(tokens[0])
        except ValueError:
            continue
        out[step] = float(tokens[3])
    return out


def _parse_cp2k_xyz(path) -> list[tuple[int, list[str], np.ndarray]]:
    """Parse a CP2K ``*-pos-1.xyz`` trajectory into ``(step, symbols, positions)``.

    The xyz comment line is ``" i = <step>, time = ..., E = ..."``; the ``step``
    is recovered from the ``i = <step>`` field so frames can be aligned to the
    ``.stress``/``.cell``/``.ener`` files by step.
    """
    lines = Path(path).read_text().splitlines()
    frames: list[tuple[int, list[str], np.ndarray]] = []
    i = 0
    n_lines = len(lines)
    while i < n_lines:
        if not lines[i].strip():
            i += 1
            continue
        n_atoms = int(lines[i].split()[0])
        comment = lines[i + 1]
        step = int(comment.split(",")[0].split("=")[1].strip())
        symbols: list[str] = []
        positions = np.empty((n_atoms, 3))
        for a in range(n_atoms):
            parts = lines[i + 2 + a].split()
            symbols.append(parts[0])
            positions[a] = [float(parts[1]), float(parts[2]), float(parts[3])]
        i += 2 + n_atoms
        frames.append((step, symbols, positions))
    return frames


def read_cp2k_md(stress_path, pos_path, cell_path, ener_path=None) -> list[Atoms]:
    """Read CP2K AIMD output files into ASE frames with Cauchy stress.

    Positions/cell come from ``*-pos-1.xyz`` and ``.cell``; the stress is the
    full Cauchy stress (tension positive, eV/A^3) from the ``.stress`` file
    (which includes the kinetic term), negated and unit-converted:
    ``sigma = -P * 1e-4 / EV_A3_TO_GPA``.  The instantaneous temperature is
    stored in ``atoms.info["temperature"]`` (kelvin) when ``ener_path`` is given; CP2K xyz
    trajectories carry no velocities, so ``extract`` does not report a
    temperature channel for these frames.

    Args:
        stress_path: CP2K ``.stress`` file (bar, 9 tensor components per step).
        pos_path: CP2K ``*-pos-1.xyz`` trajectory.
        cell_path: CP2K ``.cell`` file (row-vector lattice per step).
        ener_path: Optional CP2K ``.ener`` file for per-step temperature.

    Returns:
        One ``Atoms`` per xyz frame that has matching ``.stress`` *and* ``.cell``
        data (frames missing either are skipped -- CP2K writes these files at
        different flush intervals during a run).

    Raises:
        ValueError: if the xyz trajectory is empty, or no xyz frame has matching
            ``.stress``/``.cell`` data.
    """
    stress = _parse_cp2k_stress(stress_path)
    cells = _parse_cp2k_cell(cell_path)
    ener = _parse_cp2k_ener(ener_path) if ener_path is not None else {}
    pos_frames = _parse_cp2k_xyz(pos_path)
    if not pos_frames:
        raise ValueError("no frames found in CP2K xyz trajectory")

    bar_to_ev_a3 = _bar_to_ev_a3()
    out: list[Atoms] = []
    for step, symbols, positions in pos_frames:
        cell = cells.get(step)
        if step not in stress or cell is None:
            continue
        sigma = -stress[step] * bar_to_ev_a3  # tension positive, eV/A^3
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        atoms.calc = SinglePointCalculator(atoms, stress=sigma)
        if step in ener:
            atoms.info["temperature"] = ener[step]
        out.append(atoms)
    if not out:
        raise ValueError("no xyz frames have matching .stress/.cell data")
    return out


__all__ = [
    "read_cp2k_md",
    "read_lammps_md",
]
