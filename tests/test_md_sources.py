"""Tests for the real-MD bridge readers (``physics.md_sources``).

These use hand-computed, synthetic LAMMPS/CP2K files written to ``tmp_path`` --
no network and no SCNET data, so CI stays hermetic.  The three physical
conventions the bridge encodes are pinned here so a future regression fails
loudly:

* **Sign**: LAMMPS/CP2K pressure is positive under compression; Cauchy stress
  is positive under tension, so ``sigma = -P``.
* **Units**: ``bar -> eV/A^3`` via ``1e-4 / EV_A3_TO_GPA``.
* **Kinetic term**: the full thermo ``pxx..pyz`` (LAMMPS) / ``.stress`` (CP2K)
  is used, never a virial rebuilt from coordinates.
"""

from __future__ import annotations

import numpy as np
import pytest

from atominterface.physics import EV_A3_TO_GPA, extract, read_cp2k_md, read_lammps_md

BAR_TO_EV_A3 = 1e-4 / EV_A3_TO_GPA
MVV2E = 1.0364269e-4


def _write(path, text: str) -> None:
    # Explicit LF so the synthetic files are byte-stable regardless of platform.
    with open(path, "w", newline="\n") as fh:
        fh.write(text)


# ── LAMMPS synthetic data ─────────────────────────────────────


def _lammps_dump(step=100, box_lo=(0.0, 0.0, 0.0), box_hi=(10.0, 10.0, 10.0)) -> str:
    lo = box_lo
    hi = box_hi
    return "\n".join(
        [
            "ITEM: TIMESTEP",
            f"{step}",
            "ITEM: NUMBER OF ATOMS",
            "2",
            "ITEM: BOX BOUNDS pp pp pp",
            f"{lo[0]:.1f} {hi[0]:.1f}",
            f"{lo[1]:.1f} {hi[1]:.1f}",
            f"{lo[2]:.1f} {hi[2]:.1f}",
            "ITEM: ATOMS id type x y z vx vy vz fx fy fz",
            f"1 1 {lo[0]:.1f} {lo[1]:.1f} {lo[2]:.1f} 1.0 0.0 0.0 0.1 0.0 0.0",
            f"2 1 {lo[0] + 5.0:.1f} {lo[1] + 5.0:.1f} {lo[2] + 5.0:.1f} 0.0 2.0 0.0 0.0 -0.1 0.0",
            "",
        ]
    )


def _lammps_thermo(step=100, *, pxx=1000.0, pyy=2000.0, pzz=3000.0) -> str:
    # pxy/pxz/pyz are the LAMMPS-order off-diagonals; c_myp[1..6] is the
    # potential-only virial (deliberately different from pxx..pyz to prove the
    # reader consumes the full pressure, not the virial).
    pxy, pxz, pyz = 40.0, 50.0, 60.0
    c_myp = [999.0, 999.0, 999.0, 999.0, 999.0, 999.0]
    header = (
        "Step Temp Press Pxx Pyy Pzz Pxy Pxz Pyz "
        "c_myp[1] c_myp[2] c_myp[3] c_myp[4] c_myp[5] c_myp[6] KinEng PotEng Volume"
    )
    row = (
        f"{step} 300.0 1500.0 {pxx} {pyy} {pzz} {pxy} {pxz} {pyz} "
        + " ".join(str(x) for x in c_myp)
        + " 1.0 -10.0 1000.0"
    )
    return header + "\n" + row + "\n"


def test_read_lammps_md_stress_sign_order_units(tmp_path) -> None:
    _write(tmp_path / "dump.prod", _lammps_dump())
    _write(tmp_path / "log.lammps", _lammps_thermo())

    frames = read_lammps_md(tmp_path / "dump.prod", tmp_path / "log.lammps")
    assert len(frames) == 1

    stress = frames[0].get_stress(voigt=True)
    # sigma = -P; LAMMPS order [xx,yy,zz,xy,xz,yz] -> ASE [xx,yy,zz,yz,xz,xy].
    expected = np.array([-1000.0, -2000.0, -3000.0, -60.0, -50.0, -40.0]) * BAR_TO_EV_A3
    np.testing.assert_allclose(stress, expected, atol=1e-15)

    # The reader uses the full pressure (pxx), not the potential virial (c_myp
    # = 999), which is the whole point of the kinetic-term convention.
    assert stress[0] == pytest.approx(-1000.0 * BAR_TO_EV_A3, abs=1e-15)
    assert stress[0] != pytest.approx(-999.0 * BAR_TO_EV_A3, abs=1e-15)


def test_read_lammps_md_positions_velocities_cell(tmp_path) -> None:
    # Non-zero box origin exercises the origin shift: positions must be brought
    # into [0, L] as ASE expects.
    box_lo = (1.0, 2.0, 3.0)
    box_hi = (11.0, 12.0, 13.0)
    _write(tmp_path / "dump.prod", _lammps_dump(box_lo=box_lo, box_hi=box_hi))
    _write(tmp_path / "log.lammps", _lammps_thermo())

    frames = read_lammps_md(tmp_path / "dump.prod", tmp_path / "log.lammps")
    atoms = frames[0]

    np.testing.assert_allclose(atoms.cell, np.diag([10.0, 10.0, 10.0]), atol=1e-12)
    np.testing.assert_allclose(
        atoms.positions,
        [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]],
        atol=1e-12,
    )
    # velocities: A/ps -> ASE units via sqrt(mvv2e).
    np.testing.assert_allclose(
        atoms.get_velocities(),
        np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]) * np.sqrt(MVV2E),
        atol=1e-12,
    )
    np.testing.assert_allclose(atoms.get_masses(), [63.546, 63.546], atol=1e-12)
    # With real velocities and masses the temperature is finite (not 0 K).
    assert np.isfinite(atoms.get_temperature())


def test_read_lammps_md_empty_dump_raises(tmp_path) -> None:
    _write(tmp_path / "dump.prod", "")
    _write(tmp_path / "log.lammps", _lammps_thermo())
    with pytest.raises(ValueError, match="no frames"):
        read_lammps_md(tmp_path / "dump.prod", tmp_path / "log.lammps")


def test_read_lammps_md_disjoint_steps_raise(tmp_path) -> None:
    _write(tmp_path / "dump.prod", _lammps_dump())  # dump step 100
    _write(tmp_path / "log.lammps", _lammps_thermo(step=200))  # thermo only has 200
    with pytest.raises(ValueError, match="no dump frames have a matching thermo step"):
        read_lammps_md(tmp_path / "dump.prod", tmp_path / "log.lammps")


def test_read_lammps_md_skips_frames_without_thermo(tmp_path) -> None:
    # Two dump frames (steps 100, 200); thermo only has step 100 -> one frame out.
    _write(tmp_path / "dump.prod", _lammps_dump(step=100) + _lammps_dump(step=200))
    _write(tmp_path / "log.lammps", _lammps_thermo(step=100))
    frames = read_lammps_md(tmp_path / "dump.prod", tmp_path / "log.lammps")
    assert len(frames) == 1


def test_read_lammps_md_feeds_extract(tmp_path) -> None:
    _write(tmp_path / "dump.prod", _lammps_dump())
    _write(tmp_path / "log.lammps", _lammps_thermo())
    frames = read_lammps_md(tmp_path / "dump.prod", tmp_path / "log.lammps")
    table = extract(frames)
    assert table.n_frames == 1
    np.testing.assert_allclose(
        table.cauchy_stress_voigt[0],
        np.array([-1000.0, -2000.0, -3000.0, -60.0, -50.0, -40.0]) * BAR_TO_EV_A3,
        atol=1e-15,
    )


# ── CP2K synthetic data ───────────────────────────────────────


def _cp2k_stress(step=0) -> str:
    # 9 tensor components (bar, compression positive), symmetric by symmetry.
    return (
        "#   Step   Time [fs]            xx [bar]            xy [bar]            xz [bar]"
        "            yx [bar]            yy [bar]            yz [bar]            zx [bar]"
        "            zy [bar]            zz [bar]\n"
        f"       {step}       0.000    1000.0    10.0    20.0    10.0    2000.0    30.0"
        "    20.0    30.0    3000.0\n"
    )


def _cp2k_cell(step=0) -> str:
    return (
        "#   Step   Time [fs]       Ax [Angstrom]       Ay [Angstrom]       Az [Angstrom]"
        "       Bx [Angstrom]       By [Angstrom]       Bz [Angstrom]       Cx [Angstrom]"
        "       Cy [Angstrom]       Cz [Angstrom]      Volume [Angstrom^3]\n"
        f"       {step}       0.000        10.0000000000        0.0000000000"
        "        0.0000000000        0.0000000000        10.0000000000        0.0000000000"
        "        0.0000000000        0.0000000000        10.0000000000           1000.0000000000\n"
    )


def _cp2k_xyz(step=0) -> str:
    return (
        "      2\n"
        f" i =        {step}, time =        0.000, E =     -1.0000000000\n"
        " Cu         0.0000000000        0.0000000000        0.0000000000\n"
        " Cu         5.0000000000        5.0000000000        5.0000000000\n"
    )


def _cp2k_ener(step=0, temp=300.0) -> str:
    return (
        "#     Step Nr.          Time[fs]        Kin.[a.u.]          Temp[K]"
        "            Pot.[a.u.]        Cons Qty[a.u.]        UsedTime[s]\n"
        f"         {step}            0.000000         0.044177074"
        f"       {temp:15.9f}     -1539.793339164     -1539.748687067         0.000000000\n"
    )


def test_read_cp2k_md_stress_sign_units_cell(tmp_path) -> None:
    _write(tmp_path / "cu.stress", _cp2k_stress())
    _write(tmp_path / "cu.cell", _cp2k_cell())
    _write(tmp_path / "cu-pos-1.xyz", _cp2k_xyz())

    frames = read_cp2k_md(tmp_path / "cu.stress", tmp_path / "cu-pos-1.xyz", tmp_path / "cu.cell")
    assert len(frames) == 1
    atoms = frames[0]

    stress = atoms.get_stress(voigt=True)
    # P = [[1000,10,20],[10,2000,30],[20,30,3000]] -> sigma = -P in Voigt order.
    expected = np.array([-1000.0, -2000.0, -3000.0, -30.0, -20.0, -10.0]) * BAR_TO_EV_A3
    np.testing.assert_allclose(stress, expected, atol=1e-15)

    np.testing.assert_allclose(atoms.cell, np.diag([10.0, 10.0, 10.0]), atol=1e-12)
    np.testing.assert_allclose(atoms.positions, [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], atol=1e-12)
    assert atoms.get_chemical_symbols() == ["Cu", "Cu"]


def test_read_cp2k_md_temperature_from_ener(tmp_path) -> None:
    _write(tmp_path / "cu.stress", _cp2k_stress())
    _write(tmp_path / "cu.cell", _cp2k_cell())
    _write(tmp_path / "cu-pos-1.xyz", _cp2k_xyz())
    _write(tmp_path / "cu.ener", _cp2k_ener(temp=312.5))

    frames = read_cp2k_md(
        tmp_path / "cu.stress",
        tmp_path / "cu-pos-1.xyz",
        tmp_path / "cu.cell",
        ener_path=tmp_path / "cu.ener",
    )
    assert frames[0].info["temperature"] == pytest.approx(312.5, abs=1e-6)


def test_read_cp2k_md_empty_xyz_raises(tmp_path) -> None:
    _write(tmp_path / "cu.stress", _cp2k_stress())
    _write(tmp_path / "cu.cell", _cp2k_cell())
    _write(tmp_path / "cu-pos-1.xyz", "")
    with pytest.raises(ValueError, match="no frames"):
        read_cp2k_md(tmp_path / "cu.stress", tmp_path / "cu-pos-1.xyz", tmp_path / "cu.cell")


def test_read_cp2k_md_disjoint_steps_raise(tmp_path) -> None:
    _write(tmp_path / "cu.stress", _cp2k_stress(step=0))
    _write(tmp_path / "cu.cell", _cp2k_cell(step=0))
    _write(tmp_path / "cu-pos-1.xyz", _cp2k_xyz(step=5))  # xyz has step 5, not 0
    with pytest.raises(ValueError, match="no xyz frames have matching"):
        read_cp2k_md(tmp_path / "cu.stress", tmp_path / "cu-pos-1.xyz", tmp_path / "cu.cell")


def test_read_cp2k_md_skips_frames_without_stress(tmp_path) -> None:
    # xyz has steps 0 and 1; .stress/.cell only have step 0 -> one frame out.
    _write(tmp_path / "cu.stress", _cp2k_stress(step=0))
    _write(tmp_path / "cu.cell", _cp2k_cell(step=0))
    _write(tmp_path / "cu-pos-1.xyz", _cp2k_xyz(step=0) + _cp2k_xyz(step=1))
    frames = read_cp2k_md(tmp_path / "cu.stress", tmp_path / "cu-pos-1.xyz", tmp_path / "cu.cell")
    assert len(frames) == 1
    assert frames[0].info == {}  # no ener path given -> no temperature key
