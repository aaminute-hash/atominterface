from __future__ import annotations

from typing import Any

ANALYSIS_CHARGE = "ai_charge"
CHARGE_KNOWN = "ai_charge_known"
CHARGE_SOURCE = "ai_charge_source"
SLAB_ID = "ai_slab_id"

_POLICIES = {"report", "require-neutral", "require-nonpolar"}
_SOURCES = {
    "explicit_charges",
    "oxidation_mapping",
    "initial_charges",
    "oxidation_states",
    "unknown",
}
_GROUPS = ("total", "substrate", "film")

# Bound on the symmetric-slab thickening loop; each pass adds one
# oriented-unit-cell repeat to the half-board.
_MAX_THICKNESS_PASSES = 16


class _GeometryDefault:
    """Distinguish omitted geometry options from explicit default-valued options."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:
        return repr(self.value)


_DEFAULT_AXIS = _GeometryDefault(2)
_DEFAULT_FINITE = _GeometryDefault(False)
_DEFAULT_VACUUM = _GeometryDefault(None)
