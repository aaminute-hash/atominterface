"""Deterministic extxyz output: canonical per-atom array ordering.

pymatgen's ``AseAtomsAdaptor`` copies site properties in
set-iteration order, so without normalisation the extxyz ``Properties``
column order -- and with it the file bytes -- drift with
``PYTHONHASHSEED``.  These tests pin ``canonicalize_arrays`` and the
CLI's use of it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
from ase.atoms import Atoms
from ase.io import write as ase_write

from atominterface.io import canonicalize_arrays

ROOT = Path(__file__).parents[1]
DATA = ROOT / "examples" / "data"
SUBSTRATE = DATA / "Cu_fcc.vasp"
FILM = DATA / "Ni_fcc.vasp"


def _atoms_with_arrays(order: list[str]) -> Atoms:
    """Same content, per-atom arrays inserted in the given key order."""
    atoms = Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 0.0], [1.8, 1.8, 1.8]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )
    values = np.arange(len(atoms), dtype=float).reshape(-1, 1)
    for key in order:
        atoms.set_array(key, values.copy())
    return atoms


def _properties_names(header: str) -> list[str]:
    """Column names from an extxyz ``Properties=...`` header line."""
    segment = header.split("Properties=", 1)[1].split(" ", 1)[0]
    parts = segment.split(":")
    return [parts[index] for index in range(0, len(parts), 3)]  # name:type:count


def test_canonicalize_orders_extxyz_columns() -> None:
    """A scrambled array order normalises to sorted extxyz columns."""
    atoms = _atoms_with_arrays(["zeta", "alpha", "mid"])
    canonicalize_arrays(atoms)
    assert list(atoms.arrays) == sorted(atoms.arrays)

    buffer = StringIO()
    ase_write(buffer, atoms, format="extxyz")
    names = _properties_names(buffer.getvalue().splitlines()[1])
    assert names[:2] == ["species", "pos"]  # the writer's fixed prefix
    assert names[2:] == sorted(names[2:])


def test_canonicalize_makes_output_byte_identical() -> None:
    """Two insertion orders of the same structure write the same bytes."""
    first = _atoms_with_arrays(["zeta", "alpha", "mid"])
    second = _atoms_with_arrays(["mid", "zeta", "alpha"])
    canonicalize_arrays(first)
    canonicalize_arrays(second)

    buffer_first, buffer_second = StringIO(), StringIO()
    ase_write(buffer_first, first, format="extxyz")
    ase_write(buffer_second, second, format="extxyz")
    assert buffer_first.getvalue() == buffer_second.getvalue()


def test_canonicalize_is_noop_on_canonical_order() -> None:
    """Canonical input passes through unchanged: same keys, same values."""
    atoms = _atoms_with_arrays(["alpha", "mid"])
    canonicalize_arrays(atoms)
    before = {key: value.copy() for key, value in atoms.arrays.items()}

    canonicalize_arrays(atoms)
    assert set(atoms.arrays) == set(before)
    for key, value in before.items():
        assert np.array_equal(atoms.arrays[key], value)


@pytest.mark.skipif(not SUBSTRATE.exists(), reason="example data missing")
def test_cli_extxyz_stable_across_hash_seeds(tmp_path) -> None:
    """PYTHONHASHSEED must not change the CLI's candidate files.

    Two independent processes per configuration:

    - geometry only: the whole file must be byte-identical, which is
      what the hash-seed drift (and its fix) is about
    - ``--relax``: torch CPU thread-reduction noise (~1e-7, load
      dependent) legitimately changes numeric columns and relaxed
      positions between processes, so only the ``Properties`` segment
      -- the hash-seed-sensitive part -- is compared, and it must be
      canonical.
    """
    pytest.importorskip("pymatgen")
    for relax in (False, True):
        outputs: list[bytes] = []
        for seed in ("1", "2"):
            outdir = tmp_path / f"{'relax' if relax else 'geometry'}-{seed}"
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = seed
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            command = [
                sys.executable,
                "-m",
                "atominterface",
                "build",
                str(SUBSTRATE),
                str(FILM),
                "--outdir",
                str(outdir),
            ]
            if relax:
                command += ["--relax", "--backend", "dummy"]
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=env,
                timeout=600,
                check=False,
            )
            assert proc.returncode == 0, proc.stdout + proc.stderr
            outputs.append((outdir / "candidate_01.extxyz").read_bytes())

        if relax:
            segments = [
                output.splitlines()[1].decode().split("Properties=", 1)[1].split(" ", 1)[0]
                for output in outputs
            ]
            assert segments[0] == segments[1]
            names = _properties_names(outputs[0].splitlines()[1].decode())
            assert names[:2] == ["species", "pos"]
            assert names[2:] == sorted(names[2:])
        else:
            assert outputs[0] == outputs[1]
