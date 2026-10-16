"""Quarantine: cut out an OOD-local cluster and emit a DFT single-point input."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class QuarantineRecord:
    """One isolated OOD event."""

    run_id: str
    step: int
    center_index: int
    ood_score: float
    reasons: list[str] = field(default_factory=list)
    cluster_indices: list[int] = field(default_factory=list)
    dft_input_path: str = ""
    meta: dict = field(default_factory=dict)


def cut_cluster(
    atoms,
    center_index: int,
    radius: float = 6.0,
    *,
    include_images: bool = True,
) -> Atoms:  # noqa: D401
    """Return a new Atoms containing atoms within ``radius`` of the center.

    This is a minimal reference implementation.  Production code should
    use a neighbour list and honour PBC images.
    """
    pos = atoms.positions
    center = pos[center_index]
    dists = np.linalg.norm(pos - center, axis=1)
    keep = np.where(dists <= radius)[0].tolist()
    from ase.atoms import Atoms

    return Atoms(numbers=atoms.numbers[keep], positions=pos[keep], cell=atoms.cell, pbc=atoms.pbc)


def write_dft_input(
    atoms,
    record: QuarantineRecord,
    directory: str | Path = "./flagged",
    fmt: str = "vasp",
) -> Path:
    """Write a single-point input for the quarantined cluster."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"q_{record.run_id}_{record.step:04d}_{record.center_index}"
    if fmt == "vasp":
        atoms.write(str(p.with_suffix(".POSCAR")), format="vasp")
        (p.with_suffix(".INCAR.json")).write_text(
            json.dumps(
                {
                    "SYSTEM": f"AtomInterface quarantine step {record.step}",
                    "IBRION": -1,
                    "NSW": 0,
                    "ISIF": 2,
                    "ENCUT": 520,
                    "EDIFF": 1e-6,
                },
                indent=2,
            )
        )
        return p.with_suffix(".POSCAR")
    raise ValueError(f"unsupported dft format {fmt!r}")
