"""Tests for the top-level API surface and the physics constants single source."""

import atominterface
from atominterface import physics


def test_all_names_resolve():
    """Every name advertised in ``__all__`` is actually importable."""
    for name in atominterface.__all__:
        getattr(atominterface, name)


def test_top_level_covers_physics_export_surface():
    """The top level forwards at least everything physics exports."""
    missing = set(physics.__all__) - set(atominterface.__all__)
    assert not missing, f"top level is missing physics exports: {sorted(missing)}"


def test_constants_are_forwards_not_copies():
    """Constants must be re-exported by reference: identity locks single source."""
    for name in (
        "EV_A2_TO_J_M2",
        "J_M2_TO_EV_A2",
        "EV_A3_TO_PA",
        "EV_A3_TO_GPA",
        "EV_TO_KJ_MOL",
        "BOHR_TO_ANG",
        "ANG_TO_BOHR",
    ):
        assert getattr(atominterface, name) is getattr(physics, name), name


def test_functions_are_forwards_not_copies():
    """Function exports must be the same objects as the physics package's."""
    for name in (
        "rotate_positions",
        "rotate_stress",
        "rotation_matrix",
        "transform_forces",
        "transform_stress_voigt",
        "check_force_consistency",
        "check_stress_symmetry",
        "ev_per_a3_to_gpa",
        "gpa_to_ev_per_a3",
    ):
        assert getattr(atominterface, name) is getattr(physics, name), name


def test_constants_single_source_of_truth():
    """The derived pressure constant stays self-consistent with its base."""
    # EV_A3_TO_GPA is derived from EV_A3_TO_PA, so the pair cannot drift
    # apart the way the two historical literals did (160.21766208 vs the
    # exact 160.2176634 implied by 1.602176634e11 Pa).
    assert physics.EV_A3_TO_GPA == physics.EV_A3_TO_PA / 1e9
