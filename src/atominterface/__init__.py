"""AtomInterface: coherent interface construction with a guarded relax layer."""

from __future__ import annotations

__version__ = "0.5.1"

# ── 单一事实来源转发 ─────────────────────────────────────────
# The physics constants and conversion helpers live in
# ``atominterface.physics`` (single definition point).  The top level only
# forwards them, so ``atominterface.EV_A2_TO_J_M2 is
# atominterface.physics.EV_A2_TO_J_M2`` holds by identity -- redefining a
# literal here is exactly the drift (``EV_A3_TO_GPA``) that motivated this.
from atominterface.physics import (
    ANG_TO_BOHR,
    BOHR_TO_ANG,
    EV_A2_TO_J_M2,
    EV_A3_TO_GPA,
    EV_A3_TO_PA,
    EV_TO_KJ_MOL,
    J_M2_TO_EV_A2,
    check_force_consistency,
    check_stress_symmetry,
    ev_per_a3_to_gpa,
    gpa_to_ev_per_a3,
    rotate_positions,
    rotate_stress,
    rotation_matrix,
    transform_forces,
    transform_stress_voigt,
)

__all__ = [
    "ANG_TO_BOHR",
    "BOHR_TO_ANG",
    "EV_A2_TO_J_M2",
    "EV_A3_TO_GPA",
    "EV_A3_TO_PA",
    "EV_TO_KJ_MOL",
    "J_M2_TO_EV_A2",
    "rotate_positions",
    "rotate_stress",
    "rotation_matrix",
    "transform_forces",
    "transform_stress_voigt",
    "check_force_consistency",
    "check_stress_symmetry",
    "ev_per_a3_to_gpa",
    "gpa_to_ev_per_a3",
]
