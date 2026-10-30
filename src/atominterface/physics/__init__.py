"""Physics constants and utilities."""

from __future__ import annotations

# ── 单位换算常量 ──────────────────────────────────────────────
EV_A2_TO_J_M2 = 16.02176634
J_M2_TO_EV_A2 = 1.0 / EV_A2_TO_J_M2

EV_A3_TO_PA = 1.602176634e11
# Derived, not a second literal: the historical 160.21766208 disagreed with
# EV_A3_TO_PA in the last four digits (the exact value is 160.2176634).
EV_A3_TO_GPA = EV_A3_TO_PA / 1e9


def ev_per_a3_to_gpa(val):
    """Convert eV/Å³ to GPa."""
    return float(val) * EV_A3_TO_GPA


def gpa_to_ev_per_a3(val):
    """Convert GPa to eV/Å³."""
    return float(val) / EV_A3_TO_GPA


EV_TO_KJ_MOL = 96.48533212
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG

# ── 导出 ──────────────────────────────────────────────────────
from atominterface.physics.invariants import (
    check_force_consistency,
    check_stress_symmetry,
    rotate_positions,
    rotate_stress,
    rotation_matrix,
    transform_forces,
    transform_stress_voigt,
)

__all__ = [
    "EV_A2_TO_J_M2",
    "J_M2_TO_EV_A2",
    "EV_A3_TO_PA",
    "EV_A3_TO_GPA",
    "ev_per_a3_to_gpa",
    "gpa_to_ev_per_a3",
    "EV_TO_KJ_MOL",
    "BOHR_TO_ANG",
    "ANG_TO_BOHR",
    "rotate_positions",
    "rotate_stress",
    "rotation_matrix",
    "transform_forces",
    "transform_stress_voigt",
    "check_force_consistency",
    "check_stress_symmetry",
]
