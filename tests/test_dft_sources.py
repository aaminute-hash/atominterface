"""Synthetic (CI-closed) tests for the DFT single-point readers.

Real-data cross-checks live outside the test suite (they need SCNET output
files); these tests pin the *structure* parsing and the energy/stress
sign+unit conventions on hand-written minimal files.
"""

from __future__ import annotations

import numpy as np
import pytest

from atominterface.physics.dft_sources import (
    HARTREE_TO_EV,
    RY_TO_EV,
    _kbar_to_ev_a3,
    read_abacus,
    read_gaussian,
    read_materials_studio,
    read_qe,
    read_siesta,
)


def _siesta_fdf() -> str:
    return """\
LatticeConstant  1.0 Ang
%block LatticeVectors
       10.00000000       0.00000000       0.00000000
        0.00000000      20.00000000       0.00000000
        0.00000000       0.00000000       5.00000000
%endblock LatticeVectors
%block ChemicalSpeciesLabel
   1    6   C
   2    1   H
%endblock ChemicalSpeciesLabel
AtomicCoordinatesFormat Ang
%block AtomicCoordinatesAndAtomicSpecies
        1.00000000       2.00000000       0.50000000   1
        3.00000000       4.00000000       1.50000000   2
%endblock AtomicCoordinatesAndAtomicSpecies
"""


def test_read_siesta_structure(tmp_path) -> None:
    p = tmp_path / "struct.fdf"
    p.write_text(_siesta_fdf())
    atoms = read_siesta(p)
    assert atoms.get_chemical_formula() == "CH"
    assert len(atoms) == 2
    np.testing.assert_allclose(atoms.cell, np.diag([10.0, 20.0, 5.0]), atol=1e-12)
    np.testing.assert_allclose(atoms.positions[0], [1.0, 2.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(atoms.positions[1], [3.0, 4.0, 1.5], atol=1e-12)


def test_read_siesta_explicit_species_map(tmp_path) -> None:
    p = tmp_path / "struct.fdf"
    p.write_text(_siesta_fdf())
    atoms = read_siesta(p, species_map={1: "Si", 2: "O"})
    assert atoms.get_chemical_formula() == "OSi"


def _abacus_stru() -> str:
    return """\
ATOMIC_SPECIES
Cu  63.546  ./PP_ORB/Cu_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
6.83  0.0  0.0
0.0   6.83  0.0
0.0   0.0   6.83

ATOMIC_POSITIONS
Direct

Cu #label
0 #magnetism
1 #number of atoms
0.0  0.0  0.0  1  1  1
"""


def _abacus_cubic_stru() -> str:
    return """\
ATOMIC_SPECIES
Cu  63.546  Cu.UPF

LATTICE_CONSTANT
1.0

LATTICE_PARAMETERS
6.83  6.83  6.83

ATOMIC_POSITIONS
Direct

Cu #label
0 #magnetism
1 #number of atoms
0.0  0.0  0.0  1  1  1
"""


def test_read_abacus_lattice_vectors(tmp_path) -> None:
    p = tmp_path / "STRU"
    p.write_text(_abacus_stru())
    atoms = read_abacus(p)
    assert atoms.get_chemical_formula() == "Cu"
    assert len(atoms) == 1
    # LATTICE_CONSTANT 1.0 Bohr x LATTICE_VECTORS 6.83 Bohr -> 6.83 Bohr = 3.615 A
    expected = 6.83 * 0.529177210903
    np.testing.assert_allclose(atoms.cell, np.diag([expected] * 3), atol=1e-6)


def test_read_abacus_cubic_lattice_parameters(tmp_path) -> None:
    p = tmp_path / "STRU"
    p.write_text(_abacus_cubic_stru())
    atoms = read_abacus(p)
    expected = 6.83 * 0.529177210903
    np.testing.assert_allclose(atoms.cell, np.diag([expected] * 3), atol=1e-6)


def _abacus_log() -> str:
    # Real ABACUS running_scf.log format: a 3x3 TOTAL-STRESS block (kbar,
    # compression positive) and an already-in-eV !FINAL_ETOT_IS line.
    return """\
TOTAL-STRESS (KBAR)
-----------------------------------------------------------------
    -1376.2277552933        -0.0000000000        -0.0000000000
       -0.0000000000     -1376.2277552933         0.0000000000
       -0.0000000000         0.0000000000     -1376.2277552933
-----------------------------------------------------------------
 TOTAL-PRESSURE: -1376.227755 KBAR

 --------------------------------------------
 !FINAL_ETOT_IS -19871.3045170697369031 eV
 --------------------------------------------
"""


def test_read_abacus_log_energy_stress(tmp_path) -> None:
    stru = tmp_path / "STRU"
    stru.write_text(_abacus_stru())
    log = tmp_path / "running_scf.log"
    log.write_text(_abacus_log())
    atoms = read_abacus(stru, log_path=log)
    # Energy is already eV in the log; no Ry conversion is applied.
    assert atoms.calc.get_potential_energy() == pytest.approx(-19871.3045170697369031, rel=1e-12)
    # Stress: compression-positive kbar -> tension-positive eV/A^3, symmetric.
    expected = 1376.2277552933 * _kbar_to_ev_a3()
    stress = atoms.calc.get_stress()
    np.testing.assert_allclose(np.diag(stress), [expected] * 3, rtol=1e-12)
    np.testing.assert_allclose(stress - np.diag(np.diag(stress)), 0.0, atol=1e-12)


def test_read_abacus_ambiguous_lattice_parameters_raises(tmp_path) -> None:
    stru = """\
ATOMIC_SPECIES
Sn  118.710  Sn.UPF

LATTICE_CONSTANT
8.78

LATTICE_PARAMETERS
0.9817  3.5127

ATOMIC_POSITIONS
Direct

Sn #label
0 #magnetism
1 #number of atoms
0.0  0.0  0.0  1  1  1
"""
    p = tmp_path / "STRU"
    p.write_text(stru)
    with pytest.raises(ValueError, match="LATTICE_VECTORS"):
        read_abacus(p)


def _pwscf_out() -> str:
    return """\
     Program PWSCF v.7.0 starts

     bravais-lattice index     =            0
     lattice parameter (alat)  =       6.802180  a.u.
     number of atoms/cell      =            2
     number of atomic types    =            1
     number of electrons       =       22.00000

     celldm(1)=   6.802180  celldm(2)=   0.000000  celldm(3)=   0.000000

     crystal axes: (cart. coord. in units of alat)
               a(1) = (   1.000000   0.000000   0.000000 )
               a(2) = (   0.000000   1.000000   0.000000 )
               a(3) = (   0.000000   0.000000   1.000000 )

     Cartesian axes

     site n.     atom                  positions (alat units)
         1           Cu  tau(   1) = (   0.0000000   0.0000000   0.0000000  )
         2           Cu  tau(   2) = (   0.5000000   0.5000000   0.5000000  )

     ATOMIC_POSITIONS (angstrom)
     Cu       0.000000000   0.000000000   0.000000000
     Cu       1.800000000   1.800000000   1.800000000

     CELL_PARAMETERS (angstrom)
        3.600000000   0.000000000   0.000000000
        0.000000000   3.600000000   0.000000000
        0.000000000   0.000000000   3.600000000

!    total energy              =     -1234.56789000 Ry

     total   stress  (Ry/bohr**3)                   (kbar)     P=    0.00
       0.00010000   0.00000000   0.00000000         0.00000      0.00000      0.00000
       0.00000000   0.00010000   0.00000000         0.00000      0.00000      0.00000
       0.00000000   0.00000000   0.00010000         0.00000      0.00000      0.00000
"""


def test_read_qe_structure_energy_stress(tmp_path) -> None:
    p = tmp_path / "pwscf.out"
    p.write_text(_pwscf_out())
    atoms = read_qe(p)
    assert atoms.get_chemical_formula() == "Cu2"
    assert len(atoms) == 2
    # cell comes from celldm(1) x crystal axes, not the ATOMIC_POSITIONS block.
    expected_cell = 6.802180 * 0.529177210903
    np.testing.assert_allclose(atoms.cell, np.diag([expected_cell] * 3), atol=1e-4)
    # Energy is the Ry total energy converted to eV.
    assert atoms.calc.get_potential_energy() == pytest.approx(-1234.56789 * RY_TO_EV, rel=1e-6)
    # QE stress is compression-positive Ry/bohr^3; reader negates to
    # tension-positive eV/A^3.  1 Ry/bohr^3 = 91.8202 eV/A^3.
    expected_stress = -0.0001 * 13.605693122994 / (0.529177210903**3)
    stress = atoms.calc.get_stress()
    np.testing.assert_allclose(stress[:3], [expected_stress] * 3, rtol=1e-6)


def _gaussian_log() -> str:
    return """\
 Entering Link1 = C:\\G16W\\l1.exe
 SCF Done:  E(RB3LYP) =  -76.12345678     A.U. after   10 cycles

 Standard orientation:
 ---------------------------------------------------------------------
  Center     Atomic      Atomic             Coordinates (Angstroms)
  Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
       1          6             0        0.000000    0.000000    0.000000
       2          8             0        0.000000    0.000000    1.200000
 ---------------------------------------------------------------------

 -------------------------------------------------------------------
 Center     Atomic                   Forces (Hartrees/Bohr)
 Number     Number              X              Y              Z
 -------------------------------------------------------------------
      1        6           0.000000000    0.000000000    0.000000000
      2        8           0.000000000    0.000000000   -0.050000000
 -------------------------------------------------------------------
"""


def test_read_gaussian_energy_forces(tmp_path) -> None:
    p = tmp_path / "co.log"
    p.write_text(_gaussian_log())
    atoms = read_gaussian(p)
    assert atoms.get_chemical_formula() == "CO"
    assert len(atoms) == 2
    # molecular code -> no periodic cell
    assert not atoms.pbc.any()
    # energy is Hartree -> eV
    assert atoms.calc.get_potential_energy() == pytest.approx(
        -76.12345678 * HARTREE_TO_EV, rel=1e-9
    )
    # forces are Hartree/Bohr -> eV/A
    expected_fz = -0.05 * HARTREE_TO_EV / 0.529177210903
    forces = atoms.calc.get_forces()
    np.testing.assert_allclose(forces[0], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(forces[1], [0.0, 0.0, expected_fz], rtol=1e-9)


def _car() -> str:
    return """\
!BIOSYM archive 3
PBC=ON
Cu fcc
!DATE
PBC       3.60000    3.60000    3.60000   90.00000   90.00000   90.00000
Cu        1.8000000000      1.8000000000      1.8000000000 xxxx 1      xx      Cu      0.000
end
end
"""


def test_read_materials_studio_car(tmp_path) -> None:
    p = tmp_path / "cu.car"
    p.write_text(_car())
    atoms = read_materials_studio(p)
    assert atoms.get_chemical_formula() == "Cu"
    assert len(atoms) == 1
    np.testing.assert_allclose(atoms.cell, np.diag([3.6] * 3), atol=1e-6)
    np.testing.assert_allclose(atoms.positions[0], [1.8, 1.8, 1.8], atol=1e-6)
