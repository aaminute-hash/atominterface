"""Deterministic structure I/O helpers.

The extxyz writer emits per-atom property columns in ``atoms.arrays``
key order.  pymatgen's ``AseAtomsAdaptor`` copies site properties in
``Structure.site_properties`` iteration order, and that mapping is
built upstream from a ``set``, so string hash randomisation
(``PYTHONHASHSEED``) makes the column order drift between processes:
same numbers, different bytes.  ``canonicalize_arrays`` normalises the
key order before every write, so identical structures produce
byte-identical files regardless of the hash seed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.atoms import Atoms


def canonicalize_arrays(atoms: Atoms) -> None:
    """Reorder ``atoms.arrays`` keys into sorted order, in place.

    Pure key reordering: the set of keys and every value are untouched
    (verified by the ``test_io`` suite).  The extxyz writer maps
    ``numbers``/``positions`` to the fixed leading ``species``/``pos``
    columns regardless of this order, so only the remaining per-atom
    properties (calculator results, pymatgen site properties) are
    affected.

    Args:
        atoms: An ASE ``Atoms`` whose per-atom arrays should be written
            in a canonical, hash-seed-independent order.
    """
    arrays = atoms.arrays
    ordered = {key: arrays[key] for key in sorted(arrays)}
    arrays.clear()
    arrays.update(ordered)
