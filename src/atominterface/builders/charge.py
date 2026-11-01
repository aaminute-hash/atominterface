"""Explicit atomic-charge bookkeeping and finite-slab dipole diagnostics.

Unknown charges are not zero charges.  No oxidation states are guessed, and no
calculator is consulted or modified.  These diagnostics describe the supplied
charge model, not electronic polarization, interface stability, or a prescription
for compensating atoms or DFT dipole-correction settings.

The implementation lives in :mod:`atominterface._charge`; this module is a thin
facade that re-exports the public surface unchanged.
"""

from atominterface._charge._constants import (
    ANALYSIS_CHARGE,
    CHARGE_KNOWN,
    CHARGE_SOURCE,
    SLAB_ID,
)
from atominterface._charge.analyze import analyze_charge_polarity
from atominterface._charge.orient import _orient_slab
from atominterface._charge.policy import (
    charge_rejection_reasons,
    check_neutral_slabs,
    is_potentially_polar,
    suggest_dipole_correction,
    tasker_classification,
)
from atominterface._charge.resolve import resolve_charge_data, validate_charge_options
from atominterface._charge.symmetric import (
    SymmetricSlab,
    _normal_extent,
    _site_dependent_charge,
    finite_slab_geometry,
    make_symmetric_slab,
    mirror_slab,
    termination_cuts,
)

__all__ = [
    "ANALYSIS_CHARGE",
    "CHARGE_KNOWN",
    "CHARGE_SOURCE",
    "SLAB_ID",
    "resolve_charge_data",
    "validate_charge_options",
    "analyze_charge_polarity",
    "charge_rejection_reasons",
    "check_neutral_slabs",
    "is_potentially_polar",
    "suggest_dipole_correction",
    "tasker_classification",
    "SymmetricSlab",
    "finite_slab_geometry",
    "mirror_slab",
    "termination_cuts",
    "make_symmetric_slab",
    "_normal_extent",
    "_orient_slab",
    "_site_dependent_charge",
]
