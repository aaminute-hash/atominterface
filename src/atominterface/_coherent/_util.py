"""Shared conversion and label helpers used across the coherent submodules."""

from __future__ import annotations

import numpy as np


def _pair_label(slab, i: int, j: int) -> str:
    """``#3 Cu (u=..., v=..., w=...) / #7 Ni (...)`` for two sites.

    The fractional coordinates are the point.  A pair that is too close
    is ambiguous on distance alone: two atoms facing each other across
    the join (same ``u, v``, ``w`` straddling the interface) are the
    intended small gap, while two atoms inside one plane are a cell that
    was compressed.  Only ``w`` separates those, so it is always shown.
    """
    # Wrapped into the cell: the builders keep ``to_unit_cell=False``, so
    # raw fractional coordinates arrive as -6.33 rather than 0.67 and say
    # nothing about where in the stack the pair actually sits.
    coords = np.mod(np.asarray(slab.frac_coords, dtype=np.float64), 1.0)
    species = [str(entry) for entry in slab.species]

    def _site(index: int) -> str:
        u, v, w = coords[index]
        return f"#{index} {species[index]} (u={u:.6f}, v={v:.6f}, w={w:.6f})"

    return f"{_site(i)} / {_site(j)}"


def _reduced_formula(structure) -> str:
    """Best-effort reduced formula without requiring composition."""
    try:
        return str(structure.composition.reduced_formula)
    except Exception:  # noqa: BLE001
        try:
            from collections import Counter

            c = Counter(str(s.species) for s in structure.sites)
            return "".join(f"{k}{v}" for k, v in sorted(c.items()))
        except Exception:  # noqa: BLE001
            return "unknown"


def _to_ase(obj):
    """Return an ASE ``Atoms`` view of a pymatgen ``Structure``.

    ``Candidate.atoms`` is documented as the ASE view, and ASE is what the
    relaxers and every exporter consume, so the conversion happens here
    rather than at each call site.
    """
    from ase.atoms import Atoms

    if isinstance(obj, Atoms):
        return obj

    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_atoms(obj)
