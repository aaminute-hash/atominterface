"""Bridge DFT single-point outputs into ASE ``Atoms`` with energy / stress.

This module is the static-structure counterpart of
:mod:`atominterface.physics.md_sources`: where the MD bridge reads *trajectories*
into ``list[Atoms]``, these readers read a *single* converged structure (the last
ionic step by default) out of a DFT code's output into one ``Atoms`` carrying an
:class:`ase.calculators.singlepoint.SinglePointCalculator` with ``energy``,
``stress`` and ``forces``.

Conventions (fixed; mirrored by the MD bridge and the validation records):

* **Energy.** The converged ground-state energy is used -- VASP
  ``energy(sigma->0)`` rather than ``free energy TOTEN`` (see the VASP
  convention in the project validation notes).  All readers return eV.
* **Stress.** Returned as a symmetric 3x3 Cauchy stress, *tension positive*, in
  eV/A^3 -- the same unit ``extract`` consumes.  Codes that report a pressure
  tensor (positive under compression) are negated: ``sigma = -P``.
* **Forces.** eV/A.

Code support matrix (ASE 3.29):

* VASP  -- ``ase.io.vasp.read_vasp_out`` (OUTCAR); energy is ``sigma->0`` and
  stress is already tension-positive eV/A^3 (verified against a real OUTCAR).
* QE    -- ``ase.io.espresso.read_espresso_out`` (``pw.x`` stdout); the reader
  only handles ``ibrav == 0`` inputs but the *output* always carries an explicit
  ``CELL_PARAMETERS`` block, so output parsing is unaffected.
* SIESTA / ABACUS -- no ASE reader exists; parsed here by hand from the ``.fdf``
  / ``STRU`` structure files plus the energy/stress stdout.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import cast

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import chemical_symbols

# Physical unit conversions (values do not depend on the code).
RY_TO_EV = 13.605693122994  # 1 Rydberg in eV
HARTREE_TO_EV = 2.0 * RY_TO_EV  # 1 Hartree = 2 Rydberg
BOHR_TO_ANGSTROM = 0.529177210903


def _bar_to_ev_a3() -> float:
    """One bar expressed in eV/A^3 (``1e-4 / EV_A3_TO_GPA``).

    Lazy import avoids a circular import (the package re-exports this module).
    """
    from atominterface.physics import EV_A3_TO_GPA

    return 1e-4 / EV_A3_TO_GPA


def _kbar_to_ev_a3() -> float:
    """One kbar expressed in eV/A^3 (``1 kbar = 1000 bar``)."""
    return 1000.0 * _bar_to_ev_a3()


def read_vasp(outcar_path, index=-1) -> Atoms:
    """Read a VASP ``OUTCAR`` into ``Atoms`` with energy/stress/forces.

    Delegates to :func:`ase.io.vasp.read_vasp_out`, which parses the ``energy
    (sigma->0)`` value, the full stress tensor (already tension positive,
    eV/A^3) and the forces.  The default ``index=-1`` selects the last ionic
    step (the converged geometry).

    Args:
        outcar_path: Path to ``OUTCAR``.
        index: Ionic step to return (default ``-1`` = last).

    Returns:
        ``Atoms`` with ``.calc`` carrying ``energy`` (eV), ``stress``
        (3x3, tension positive, eV/A^3) and ``forces`` (eV/A).
    """
    from ase.io.vasp import read_vasp_out

    atoms = read_vasp_out(str(Path(outcar_path)), index=index)
    if atoms.calc is None:
        raise ValueError("VASP OUTCAR produced no calculator results")
    return cast("Atoms", atoms)


def read_qe(pw_out_path) -> Atoms:
    """Read a Quantum ESPRESSO ``pw.x`` stdout into ``Atoms``.

    Delegates to :func:`ase.io.espresso.read_espresso_out`.  The output carries
    ``CELL_PARAMETERS``, ``ATOMIC_POSITIONS`` and the converged ``! total
    energy``; stress is present only when ``tstress=.true.`` was set.

    Args:
        pw_out_path: Path to the ``pw.x`` stdout file.

    Returns:
        ``Atoms`` with ``.calc`` carrying ``energy`` (eV), ``stress`` (3x3,
        tension positive, eV/A^3) and ``forces`` (eV/A).
    """
    from ase.io.espresso import read_espresso_out

    # read_espresso_out is a generator over ionic steps; take the last
    # (converged) configuration.  It is decorated with @reader, so a path
    # would be opened and closed before the lazy generator runs -- pass an
    # open handle instead and keep it alive while the generator is drained.
    with Path(pw_out_path).open() as handle:
        frames = list(read_espresso_out(handle))
    if not frames:
        raise ValueError("no configurations found in QE output")
    atoms = frames[-1]
    if atoms.calc is None:
        raise ValueError("QE output produced no calculator results")
    return cast("Atoms", atoms)


def _parse_fdf_structure(path) -> tuple[np.ndarray, list[int], np.ndarray]:  # noqa: PLR0912, PLR0915
    """Parse a SIESTA ``.fdf`` structure block.

    Reads ``LatticeConstant`` + ``LatticeVectors``, ``AtomicCoordinatesFormat``
    and ``AtomicCoordinatesAndAtomicSpecies``.  Returns ``(cell, species_ids,
    positions)`` in Angstrom, where ``species_ids`` are the integer species
    labels used by ``ChemicalSpeciesLabel``.
    """
    text = Path(path).read_text()
    scale = 1.0
    fmt = "Ang"
    lattice: list[list[float]] = []
    coords: list[tuple[int, float, float, float]] = []
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        # "LatticeConstant <value> <unit>"
        if line.startswith("LatticeConstant"):
            parts = line.split()
            scale = float(parts[1])
            if len(parts) > 2:
                unit = parts[2].lower()
                if unit.startswith("bohr") or unit.startswith("au"):
                    scale *= BOHR_TO_ANGSTROM
            i += 1
            continue
        if line.lower().startswith("atomiccoordinatesformat"):
            parts = line.split()
            fmt = parts[1] if len(parts) > 1 else "Ang"
            i += 1
            continue
        if line.lower().startswith("%block latticevectors"):
            i += 1
            while i < n and not lines[i].strip().lower().startswith("%endblock"):
                parts = lines[i].split()
                if len(parts) >= 3:
                    lattice.append([float(parts[0]), float(parts[1]), float(parts[2])])
                i += 1
            i += 1
            continue
        if line.lower().startswith("%block atomiccoordinatesandatomicspecies"):
            i += 1
            while i < n and not lines[i].strip().lower().startswith("%endblock"):
                parts = lines[i].split()
                # "<x> <y> <z> <species_id> [# comment]"
                if len(parts) >= 4:
                    try:
                        sid = int(parts[3])
                    except ValueError:
                        i += 1
                        continue
                    coords.append((sid, float(parts[0]), float(parts[1]), float(parts[2])))
                i += 1
            i += 1
            continue
        i += 1

    if len(lattice) != 3:
        raise ValueError("SIESTA fdf must define a 3-vector LatticeVectors block")
    cell = np.array(lattice, dtype=float) * scale
    species_ids = [c[0] for c in coords]
    positions = np.array([[c[1], c[2], c[3]] for c in coords], dtype=float)
    if fmt.lower().startswith("bohr") or fmt.lower().startswith("au"):
        positions *= BOHR_TO_ANGSTROM
    elif fmt.lower().startswith("fraction") or fmt.lower().startswith("direct"):
        positions = positions @ cell
    return cell, species_ids, positions


def _parse_fdf_species(path) -> dict[int, str]:
    """Parse ``ChemicalSpeciesLabel`` from a SIESTA ``.fdf`` into ``{id: symbol}``.

    The block is ``<id> <Z> <symbol> [label]``.
    """
    text = Path(path).read_text()
    lines = text.splitlines()
    mapping: dict[int, str] = {}
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip().lower().startswith("%block chemicalspecieslabel"):
            i += 1
            while i < n and not lines[i].strip().lower().startswith("%endblock"):
                parts = lines[i].split()
                if len(parts) >= 3:
                    try:
                        sid = int(parts[0])
                    except ValueError:
                        i += 1
                        continue
                    mapping[sid] = parts[2].split("#")[0].strip()
                i += 1
            break
        i += 1
    return mapping


def read_siesta(fdf_path, species_map=None, out_path=None) -> Atoms:
    """Read a SIESTA ``.fdf`` structure into ``Atoms``.

    ``fdf_path`` must contain ``LatticeVectors`` and
    ``AtomicCoordinatesAndAtomicSpecies``.  Species IDs are resolved to element
    symbols via ``species_map`` (``{id: symbol}``); when omitted, the function
    tries to read a ``ChemicalSpeciesLabel`` block from the same file.

    Args:
        fdf_path: Path to the ``.fdf`` file with the structure blocks.
        species_map: Optional ``{species_id: element_symbol}`` mapping (taken
            from ``ChemicalSpeciesLabel`` when omitted).
        out_path: Optional ``.out`` file; when given, the converged total energy
            is read from the ``siesta: E_KS(eV)`` / ``Total = ... eV`` line.

    Returns:
        ``Atoms`` with the structure; when ``out_path`` is given a
        ``SinglePointCalculator`` carrying the total energy (eV) is attached.
    """
    cell, species_ids, positions = _parse_fdf_structure(fdf_path)
    if not positions.size:
        raise ValueError("no atomic coordinates found in SIESTA fdf")
    mapping = species_map if species_map is not None else _parse_fdf_species(fdf_path)
    symbols = [mapping[sid] for sid in species_ids]
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    if out_path is not None:
        energy = _parse_siesta_energy(out_path)
        atoms.calc = SinglePointCalculator(atoms, energy=energy)
    return atoms


def _parse_siesta_energy(path) -> float:
    """Extract the converged total energy (eV) from a SIESTA ``.out``.

    SIESTA prints ``siesta: E_KS(eV) = <value>`` at the end of the SCF loop;
    older versions print ``Total = <value>``.  The last such line is returned.
    """
    energy: float | None = None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if "E_KS(eV)" in line and "=" in line or line.startswith("Total ="):
            energy = float(line.split("=")[-1].split()[0])
    if energy is None:
        raise ValueError("no total energy found in SIESTA output")
    return energy


def _parse_abacus_stru(path) -> tuple[list[str], np.ndarray, np.ndarray]:  # noqa: PLR0912, PLR0915
    """Parse an ABACUS ``STRU`` into ``(symbols, positions, cell)`` in Angstrom.

    Handles ``ATOMIC_SPECIES`` (element + mass), ``LATTICE_CONSTANT`` (Bohr by
    default), ``LATTICE_VECTORS`` and ``ATOMIC_POSITIONS`` (``Direct`` or
    ``Cartesian``).  The per-atom trailing integer flags are ignored.
    """
    text = Path(path).read_text()
    lines = text.splitlines()
    elements: list[str] = []
    lattice_constant = BOHR_TO_ANGSTROM  # ABACUS default unit is Bohr
    lattice_vectors: list[list[float]] = []
    positions_direct = False
    positions: list[tuple[str, float, float, float]] = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        upper = line.upper()
        if upper.startswith("ATOMIC_SPECIES"):
            i += 1
            while (
                i < n
                and lines[i].strip()
                and not lines[i]
                .strip()
                .upper()
                .startswith(
                    (
                        "NUMERICAL_ORBITAL",
                        "LATTICE_CONSTANT",
                        "LATTICE_VECTORS",
                        "LATTICE_PARAMETERS",
                        "ATOMIC_POSITIONS",
                    )
                )
            ):
                parts = lines[i].split()
                if parts:
                    elements.append(parts[0])
                i += 1
            continue
        if upper.startswith("NUMERICAL_ORBITAL"):
            i += 1
            while (
                i < n
                and lines[i].strip()
                and not lines[i]
                .strip()
                .upper()
                .startswith(
                    (
                        "LATTICE_CONSTANT",
                        "LATTICE_VECTORS",
                        "LATTICE_PARAMETERS",
                        "ATOMIC_POSITIONS",
                    )
                )
            ):
                i += 1
            continue
        if upper.startswith("LATTICE_CONSTANT"):
            i += 1
            # value may sit on the same line or the next; ABACUS default is
            # Bohr, so scale it to Angstrom here (LATTICE_VECTORS / PARAMETERS
            # are multiples of this constant).
            parts = lines[i - 1].split()
            if len(parts) >= 2:
                value = float(parts[1])
            else:
                value = float(lines[i].split()[0])
                i += 1
            lattice_constant = value * BOHR_TO_ANGSTROM
            continue
        if upper.startswith("LATTICE_VECTORS"):
            i += 1
            while i < n and len(lattice_vectors) < 3:
                parts = lines[i].split()
                if len(parts) >= 3:
                    lattice_vectors.append([float(parts[0]), float(parts[1]), float(parts[2])])
                i += 1
            continue
        if upper.startswith("LATTICE_PARAMETERS"):
            # ABACUS expands LATTICE_PARAMETERS through the INPUT ``latname``.
            # Only the cubic / orthorhombic forms (1 or 3 parameters) have an
            # unambiguous expansion; the hexagonal/rhombohedral 2-parameter
            # forms need ``latname`` and are rejected in favour of explicit
            # ``LATTICE_VECTORS``.
            i += 1
            params: list[float] = []
            while i < n and len(params) < 3:
                parts = lines[i].split()
                if not parts:
                    i += 1
                    continue
                try:
                    params.extend(float(x) for x in parts)
                except ValueError:
                    break
                i += 1
            if len(params) == 1:
                a = params[0]
                lattice_vectors = [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
            elif len(params) == 3:
                a, b, c = params
                lattice_vectors = [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]]
            else:
                raise ValueError(
                    "ABACUS LATTICE_PARAMETERS with "
                    f"{len(params)} values is ambiguous (needs INPUT latname); "
                    "use explicit LATTICE_VECTORS"
                )
            continue
        if upper.startswith("ATOMIC_POSITIONS"):
            i += 1
            # The coordinate mode is the first token of the following block.
            if i < n and lines[i].strip():
                mode = lines[i].strip().lower()
                positions_direct = mode.startswith(("direct", "fractional", "frac"))
                if mode in ("direct", "cartesian", "fractional", "frac"):
                    i += 1
            while i < n:
                parts = lines[i].split()
                if not parts:
                    i += 1
                    continue
                # element header line: "<Element> #label"
                elem = parts[0]
                i += 1
                # "0 #magnetism"
                if i < n:
                    i += 1
                # "<n_atoms> #number of atoms"
                if i < n:
                    n_atoms = int(lines[i].split()[0])
                    i += 1
                for _ in range(n_atoms):
                    if i >= n:
                        break
                    p = lines[i].split()
                    if len(p) >= 3:
                        positions.append((elem, float(p[0]), float(p[1]), float(p[2])))
                    i += 1
            continue
        i += 1

    if not positions:
        raise ValueError("no atomic positions found in ABACUS STRU")
    cell = np.array(lattice_vectors, dtype=float) * lattice_constant
    symbols = [p[0] for p in positions]
    pos = np.array([[p[1], p[2], p[3]] for p in positions], dtype=float)
    if positions_direct:
        pos = pos @ cell
    return symbols, pos, cell


def read_abacus(stru_path, log_path=None) -> Atoms:
    """Read an ABACUS ``STRU`` (+ optional ``running_scf.log``) into ``Atoms``.

    The structure is parsed from ``STRU``.  When ``log_path`` is given, the
    converged total energy (eV) and total stress (kbar, compression
    positive -> tension positive eV/A^3) are read from the SCF log.

    Args:
        stru_path: Path to the ABACUS ``STRU`` file.
        log_path: Optional ``running_scf.log`` with the converged energy/stress.

    Returns:
        ``Atoms`` with the structure; when ``log_path`` is given a
        ``SinglePointCalculator`` carrying ``energy`` (eV) and ``stress``
        (3x3, tension positive, eV/A^3) is attached.
    """
    symbols, positions, cell = _parse_abacus_stru(stru_path)
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    if log_path is not None:
        energy, stress = _parse_abacus_log(log_path)
        atoms.calc = SinglePointCalculator(atoms, energy=energy, stress=stress)
    return atoms


def _parse_abacus_log(path) -> tuple[float, np.ndarray]:
    """Extract ``(energy_eV, stress_matrix)`` from an ABACUS ``running_scf.log``.

    Energy is the last ``!FINAL_ETOT_IS <value> eV`` line -- ABACUS already
    reports eV (verified against a real ``running_scf.log``), so no Ry
    conversion is applied.  Stress is the 3x3 ``TOTAL-STRESS (KBAR)`` block
    (compression positive); it is negated and converted to eV/A^3 (tension
    positive).
    """
    lines = Path(path).read_text().splitlines()
    energy_eV: float | None = None
    stress_kbar: list[list[float]] | None = None
    for i, raw in enumerate(lines):
        line = raw.strip()
        if "FINAL_ETOT_IS" in line:
            # "!FINAL_ETOT_IS -19871.3045170697369031 eV" -> the float token.
            for tok in line.split():
                try:
                    energy_eV = float(tok)
                    break
                except ValueError:
                    continue
        if line.startswith("TOTAL-STRESS"):
            # Header "TOTAL-STRESS (KBAR)" then a dashed separator then three
            # rows of three components (the symmetric Cauchy tensor).
            block: list[list[float]] = []
            j = i + 1
            while j < len(lines) and len(block) < 3:
                parts = lines[j].strip().split()
                if len(parts) >= 3:
                    with contextlib.suppress(ValueError):
                        block.append([float(x) for x in parts[:3]])
                j += 1
            if len(block) == 3:
                stress_kbar = block
    if energy_eV is None:
        raise ValueError("no total energy found in ABACUS running_scf.log")
    if stress_kbar is None:
        raise ValueError("no TOTAL-STRESS block found in ABACUS running_scf.log")
    # Compression positive -> tension positive; kbar -> eV/A^3.
    matrix = -np.array(stress_kbar, dtype=float) * _kbar_to_ev_a3()
    return energy_eV, matrix


def _parse_gaussian_log(path) -> tuple[list[str], np.ndarray, float, np.ndarray | None]:  # noqa: PLR0912, PLR0915
    """Extract ``(symbols, positions, energy_eV, forces)`` from a Gaussian ``.log``.

    Gaussian is a molecular code: there is no periodic cell and no stress.
    Energy is the last ``SCF Done`` / ``Energy=`` value (Hartree -> eV); the
    structure is the last ``Standard/Input orientation`` block (Angstrom);
    forces are the ``Forces (Hartrees/Bohr)`` block converted to eV/A.
    """
    lines = Path(path).read_text().splitlines()
    symbols: list[str] = []
    positions: list[list[float]] = []
    forces: list[list[float]] = []
    energy_eV: float | None = None

    def _is_orient(line: str) -> bool:
        return line in (
            "Standard orientation:",
            "Input orientation:",
            "Z-Matrix orientation:",
        )

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line.startswith("SCF Done") or line.startswith("Energy="):
            value = line.split("=")[1].split()[0].replace("D", "E")
            energy_eV = float(value) * HARTREE_TO_EV
        elif _is_orient(line):
            # orientation line + 4 header rows, then "<center> <Z> <type> x y z".
            syms: list[str] = []
            pos: list[list[float]] = []
            j = i + 5
            while j < n:
                parts = lines[j].split()
                if len(parts) < 6:
                    break
                try:
                    z = int(parts[1])
                except ValueError:
                    break
                syms.append(chemical_symbols[z])
                pos.append([float(parts[3]), float(parts[4]), float(parts[5])])
                j += 1
            if syms:
                symbols = syms
                positions = pos
        elif "Forces (Hartrees/Bohr)" in line:
            # header "Number Number X Y Z" + dashed row, then "<c> <Z> fx fy fz".
            frc: list[list[float]] = []
            j = i + 3
            while j < n and lines[j].strip() and not lines[j].strip().startswith("---"):
                parts = lines[j].split()
                if len(parts) >= 5:
                    with contextlib.suppress(ValueError):
                        frc.append([float(parts[2]), float(parts[3]), float(parts[4])])
                j += 1
            if frc:
                forces = frc
        i += 1

    if not symbols:
        raise ValueError("no orientation block found in Gaussian output")
    if energy_eV is None:
        raise ValueError("no SCF energy found in Gaussian output")
    positions_arr = np.array(positions, dtype=float)
    forces_arr = (
        np.array(forces, dtype=float) * HARTREE_TO_EV / BOHR_TO_ANGSTROM if forces else None
    )
    return symbols, positions_arr, energy_eV, forces_arr


def read_gaussian(log_path) -> Atoms:
    """Read a Gaussian ``.log`` into ``Atoms`` (energy + forces; molecular).

    Gaussian is a molecular quantum-chemistry code, so the returned ``Atoms``
    has no periodic cell and no stress -- only ``energy`` (eV) and ``forces``
    (eV/A) are attached via a ``SinglePointCalculator``.

    Args:
        log_path: Path to the Gaussian ``.log`` / ``.out`` file.

    Returns:
        ``Atoms`` with ``.calc`` carrying ``energy`` (eV) and ``forces``
        (eV/A) when the ``Forces`` block is present.
    """
    symbols, positions, energy, forces = _parse_gaussian_log(log_path)
    atoms = Atoms(symbols=symbols, positions=positions)
    kwargs: dict = {"energy": energy}
    if forces is not None:
        kwargs["forces"] = forces
    atoms.calc = SinglePointCalculator(atoms, **kwargs)
    return atoms


def read_materials_studio(car_path) -> Atoms:
    """Read a Materials Studio / DMol3 ``.car`` structure into ``Atoms``.

    Delegates to :func:`ase.io.dmol.read_dmol_car`, which reads the ``PBC``
    cell block (``a b c alpha beta gamma``) and the atomic coordinate rows.
    A ``.car`` carries the structure only (no energy); DMol3 ``.outmol`` and
    CASTEP ``.castep`` outputs are not parsed, and the ``.xtd`` trajectory is
    proprietary (see :func:`read_forcite_md`).

    Args:
        car_path: Path to the Materials Studio ``.car`` file.

    Returns:
        ``Atoms`` with the structure (and cell when ``PBC=ON``).
    """
    from ase.io.dmol import read_dmol_car

    with Path(car_path).open() as handle:
        return cast("Atoms", read_dmol_car(handle))


__all__ = [
    "read_abacus",
    "read_gaussian",
    "read_materials_studio",
    "read_qe",
    "read_siesta",
    "read_vasp",
]
