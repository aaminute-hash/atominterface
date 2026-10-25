"""Quarantine: cut out an OOD-local cluster and emit a DFT single-point input."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .proximity import nearest_image_vectors

if TYPE_CHECKING:
    from ase.atoms import Atoms


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
    record: QuarantineRecord | None = None,
) -> Atoms:  # noqa: D401
    """Return a new Atoms containing atoms within ``radius`` of the center.

    With ``include_images`` (the default) distances are minimum-image
    distances and every selected atom is emitted at the image nearest
    the center, so a neighbour sitting across a periodic boundary is
    included *and* the result is one contiguous ball rather than two
    fragments at opposite faces of the cell.  Each source atom
    contributes at most one entry, which keeps ``len(cluster) <=
    len(atoms)`` and keeps ``cluster_indices`` a 1:1 map back to the
    parent structure.  With ``include_images=False`` the distance is a
    bare cartesian one -- the pre-PBC behaviour, kept explicit.

    ``radius`` is measured from the center atom, which is therefore
    always included.  An out-of-range ``center_index`` raises instead of
    silently falling back to index 0: a quarantine anchored on the wrong
    atom is a DFT single-point that says nothing about the OOD event.

    Passing ``record`` fills its :attr:`QuarantineRecord.cluster_indices`
    with the indices kept (ascending); the same list is always available
    as ``cluster.info["cluster_indices"]``.
    """
    n = len(atoms)
    if not 0 <= center_index < n:
        raise IndexError(f"center_index {center_index} out of range for {n} atoms")

    pos = np.asarray(atoms.positions, dtype=float)
    center = pos[center_index]
    if include_images and np.any(np.asarray(atoms.pbc, dtype=bool)):
        vectors, dists = nearest_image_vectors(atoms, center)
        cluster_pos = center + vectors
    else:
        dists = np.linalg.norm(pos - center, axis=1)
        cluster_pos = pos
    keep = np.where(dists <= radius)[0]

    from ase.atoms import Atoms

    cluster = Atoms(
        numbers=atoms.numbers[keep],
        positions=cluster_pos[keep],
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    indices = [int(i) for i in keep]
    cluster.info["cluster_indices"] = list(indices)
    cluster.info["center_index"] = int(center_index)
    cluster.info["cut_radius"] = float(radius)
    if record is not None:
        record.cluster_indices = list(indices)
    return cluster


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
