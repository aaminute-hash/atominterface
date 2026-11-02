"""Synthetic tests for the GROMACS / DCD / Forcite MD readers."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from atominterface.physics.md_sources import (
    read_dcd_md,
    read_forcite_md,
    read_gromacs_gro,
)


def test_read_gromacs_gro_structure(tmp_path) -> None:
    gro = """\
Cu box
    2
    1CU      CU    1   0.000   0.000   0.000
    1CU      CU    2   1.800   1.800   1.800
   3.60000   3.60000   3.60000
"""
    p = tmp_path / "box.gro"
    p.write_text(gro)
    atoms = read_gromacs_gro(p, symbols=["Cu", "Cu"])
    assert atoms.get_chemical_formula() == "Cu2"
    assert len(atoms) == 2
    # .gro is nm; reader scales to Angstrom (x10).
    np.testing.assert_allclose(atoms.cell, np.diag([36.0] * 3), atol=1e-6)
    np.testing.assert_allclose(atoms.positions[1], [18.0, 18.0, 18.0], atol=1e-6)


def _build_dcd(n_frames: int, n_atoms: int, has_cell: bool, positions) -> bytes:
    """Build a minimal little-endian DCD file."""
    header = struct.pack("<4s", b"CORD")  # int32[0]
    header += struct.pack("<i", n_frames)  # int32[1] NSET
    header += struct.pack("<i", 0)  # int32[2] ISTART
    header += struct.pack("<i", 1)  # int32[3] NSAVC
    header += struct.pack("<5i", 0, 0, 0, 0, 0)  # int32[4:9] NSTEP
    header += struct.pack("<i", 0)  # int32[9]
    header += struct.pack("<i", 1 if has_cell else 0)  # int32[10] unit-cell flag
    header += struct.pack("<10i", *([0] * 10))  # int32[11:21]
    assert len(header) == 84, len(header)

    out = bytearray()
    out += struct.pack("<i", 84) + header + struct.pack("<i", 84)
    for frame in positions:
        if has_cell:
            cell_record = struct.pack("<6d", 36.0, 36.0, 36.0, 90.0, 90.0, 90.0)
            out += struct.pack("<i", 48) + cell_record + struct.pack("<i", 48)
        coord_record = struct.pack(f"<{n_atoms * 3}f", *np.asarray(frame).ravel())
        out += struct.pack("<i", n_atoms * 12) + coord_record + struct.pack("<i", n_atoms * 12)
    return bytes(out)


def test_read_dcd_md_no_cell(tmp_path) -> None:
    frames = [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]]
    data = _build_dcd(2, 2, has_cell=False, positions=frames)
    p = tmp_path / "traj.dcd"
    p.write_bytes(data)
    atoms = read_dcd_md(p, symbols=["C", "C"])
    assert len(atoms) == 2
    np.testing.assert_allclose(atoms[0].positions[1], [1.0, 1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(atoms[1].positions[0], [0.5, 0.5, 0.5], atol=1e-6)
    assert not atoms[0].pbc.any()


def test_read_dcd_md_with_cell(tmp_path) -> None:
    frames = [[[0.0, 0.0, 0.0]]]
    data = _build_dcd(1, 1, has_cell=True, positions=frames)
    p = tmp_path / "traj.dcd"
    p.write_bytes(data)
    atoms = read_dcd_md(p, symbols=["Cu"])
    assert len(atoms) == 1
    assert atoms[0].pbc.all()
    assert atoms[0].info["dcd_cell"] == [36.0, 36.0, 36.0, 90.0, 90.0, 90.0]


def test_read_forcite_md_raises() -> None:
    with pytest.raises(NotImplementedError, match="Forcite"):
        read_forcite_md("whatever.xtd")
