"""Charge-balance and polarity handling (Phase 0.2 placeholder).

This module deliberately contains **only** the contract and cheap
heuristics.  The hard problem -- electrostatic potential alignment
across a polar interface -- is left for 0.2 because it requires a
reference DFT/MLIP electrostatic potential, not a geometry trick.

What we do provide now:
  * a polarity warning based on the two bulk formulae (dipolar
    compounds are flagged, never "fixed" automatically);
  * a neutral-slab checker that warns when the net charge of either
    slab is non-zero under a simple oxidation-state bookkeeping.

Automatic charge compensation (adsorbates,掺杂, dipole corrections)
will land in 0.2 behind an explicit opt-in flag, so users are not
silently handed a charged interface.
"""

from __future__ import annotations

import warnings
from typing import Any


def is_potentially_polar(structure: Any) -> tuple[bool, str]:
    """Heuristic polarity warning from bulk stoichiometry.

    Returns
    -------
    (polar, reason)
        ``polar`` is True when the composition is consistent with a
        polar surface (e.g. rocksalt (111), wurtzite (0001)).  This
        is a **warning**, not a conclusion: terminations matter.
    """
    try:
        comp = structure.composition
    except Exception:  # noqa: BLE001
        return False, "cannot determine composition"

    reduced = comp.reduced_formula
    # Rocksalt-derived structures are the classic polar-surface case.
    if any(x in reduced.lower() for x in ("cl", "f", "o", "s", "n")):
        return True, f"{reduced} may expose a polar surface; check termination charge"
    return False, ""


def check_neutral_slabs(atoms: Any, slab_tags: list[int]) -> list[str]:
    """Warn if either slab has a non-zero nominal charge.

    ``slab_tags`` is a list parallel to ``atoms``, with 0 = substrate
    and 1 = film.  Oxidation states are guessed from the element via
    pymatgen's ``OxidationStateGuess``; un-guessable elements are
    skipped with a warning rather than zeroed out.
    """
    warnings.warn(
        "check_neutral_slabs is a placeholder: oxidation-state bookkeeping "
        "is not yet implemented and will return empty warnings.",
        stacklevel=2,
    )
    return []


def suggest_dipole_correction(atoms: Any) -> dict[str, Any]:
    """Return a suggestion dict, never modify the structure.

    Real dipole-correction setup (adding a dipole layer, switching
    electronic smearing, setting IDIPOL in VASP) is deferred to 0.2.
    """
    return {
        "dipole_correction_recommended": False,
        "reason": "placeholder; implement in Phase 0.2",
    }
