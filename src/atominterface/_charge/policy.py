from __future__ import annotations

from typing import Any

from atominterface._charge._constants import _GROUPS
from atominterface._charge.analyze import analyze_charge_polarity
from atominterface._charge.resolve import validate_charge_options


def charge_rejection_reasons(analysis: dict[str, Any], policy: str) -> list[str]:
    """Explain strict-policy failures for total, substrate AND film groups."""
    validate_charge_options(policy, 0.0, 0.0)
    if policy == "report":
        return []
    reasons = []
    groups = analysis.get("groups", {})
    for name in _GROUPS:
        group = groups.get(name, {})
        neutrality = group.get("neutrality", "unknown")
        if neutrality != "neutral":
            reasons.append(f"{name}: neutrality {neutrality}")
        polarity = group.get("polarity", "unknown")
        if policy == "require-nonpolar" and polarity != "nonpolar":
            reasons.append(f"{name}: polarity {polarity}")
    return reasons


def is_potentially_polar(structure: Any) -> tuple[bool, str]:
    """Composition-only hint that a polar termination is possible.

    Only multi-element compositions containing a polar-bond-forming anion
    (F, Cl, Br, I, O, S, N) are flagged; a single-element or anion-free
    composition returns ``(False, "")``.  The empty reason is not a
    nonpolarity determination -- it only means the composition carries no
    polar-termination warning.  Exact element symbols are matched, so Co,
    Ni and Fe are never flagged by substring accidents.
    """
    try:
        if hasattr(structure, "get_chemical_symbols"):
            elements = set(structure.get_chemical_symbols())
        else:
            elements = {element.symbol for element in structure.composition.elements}
    except (AttributeError, TypeError, ValueError):
        return False, "polarity unknown: cannot determine element composition"
    if len(elements) > 1 and elements.intersection({"F", "Cl", "Br", "I", "O", "S", "N"}):
        return (
            True,
            "composition-only hint: a polar termination is possible; actual polarity unknown",
        )
    return False, ""


def check_neutral_slabs(atoms: Any, slab_tags: list[int]) -> list[str]:
    """Return charged/unknown group problems without guessing oxidation states."""
    return charge_rejection_reasons(analyze_charge_polarity(atoms, slab_tags), "require-neutral")


def suggest_dipole_correction(atoms: Any) -> dict[str, Any]:
    """Suggest True/False/None only from the total-slab model; never change DFT settings."""
    total = analyze_charge_polarity(atoms)["groups"]["total"]
    polarity = total["polarity"]
    if polarity == "unknown":
        return {
            "dipole_correction_recommended": None,
            "reason": "polarity unknown: " + ", ".join(total["reasons"]),
        }
    return {
        "dipole_correction_recommended": polarity == "polar",
        "reason": f"total slab is {polarity} under the supplied charge model; calculator setup is unchanged",
    }


def tasker_classification(analysis: dict[str, Any]) -> int | None:
    """Tasker-type proxy derived from a supplied-charge analysis.

    Returns 3 when any group is polar or charged (an asymmetric board's
    normal dipole grows with thickness), 1 when every group is neutral and
    nonpolar, and ``None`` when the supplied charge model does not
    determine it.  This is bookkeeping over the supplied charges, not the
    full plane-resolved Tasker analysis, and it never guesses charges.
    """
    groups = analysis.get("groups", {})
    if not groups:
        return None
    if any(
        group.get("polarity") == "polar" or group.get("neutrality") == "charged"
        for group in groups.values()
    ):
        return 3
    if all(
        group.get("polarity") == "nonpolar" and group.get("neutrality") == "neutral"
        for group in groups.values()
    ):
        return 1
    return None
