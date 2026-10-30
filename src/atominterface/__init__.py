from __future__ import annotations

__version__ = "0.4.0"


"""Physics constants and utilities."""


# ── 单位换算常量 ──────────────────────────────────────────────
EV_A2_TO_J_M2: float = 16.02176634
J_M2_TO_EV_A2: float = 1.0 / EV_A2_TO_J_M2
EV_TO_KJ_MOL: float = 96.48533212
BOHR_TO_ANG: float = 0.529177210903

# ── 导出 ──────────────────────────────────────────────────────
from atominterface.physics.invariants import (
    check_force_consistency,
    check_stress_symmetry,
    rotate_positions,
    rotate_stress,
)

__all__ = [
    "EV_A2_TO_J_M2",
    "J_M2_TO_EV_A2",
    "EV_TO_KJ_MOL",
    "BOHR_TO_ANG",
    "rotate_positions",
    "rotate_stress",
    "check_force_consistency",
    "check_stress_symmetry",
]
