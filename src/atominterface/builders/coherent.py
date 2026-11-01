"""Coherent interface builder.

Wraps ``pymatgen.analysis.interfaces.coherent_interfaces`` and
:mod:`pymatgen.analysis.interfaces.zsl` behind a backend-agnostic
protocol, adding:

* lattice-match / strain filtering **before** structure generation
* substrate / film termination enumeration with bookkeeping
* gap and vacuum handling
* rotation-matrix preservation for downstream stress transforms
"""

from __future__ import annotations

from atominterface._coherent._constants import (
    _PYMTG_COHERENT_BUILDER,  # noqa: F401 (re-exported for patch targets)
    _PYMTG_STRUCTURE,  # noqa: F401 (re-exported for patch targets)
    AREA_POLICIES,  # noqa: F401 (re-exported for tests)
    ON_TOP_LATERAL_TOL,  # noqa: F401 (re-exported for tests)
    REGISTRY_MODES,  # noqa: F401 (re-exported for tests)
)
from atominterface._coherent.assembly import (
    _assemble_interface,  # noqa: F401 (re-exported for tests)
    _flatten_terminations,  # noqa: F401 (re-exported for tests)
    _get_terminations,  # noqa: F401 (re-exported for tests)
    _make_slabs,  # noqa: F401 (re-exported for tests)
    _prepare_charge_structure,  # noqa: F401 (re-exported for tests)
    reference_slabs,
)
from atominterface._coherent.builder import (
    CoherentInterfaceBuilder,
    _to_candidate,  # noqa: F401 (re-exported for tests)
    _warn_charge_diagnostics,  # noqa: F401 (re-exported for tests)
)
from atominterface._coherent.cells import (
    InterfaceAreas,  # noqa: F401 (re-exported for tests)
    _lattice_area,  # noqa: F401 (re-exported for tests)
    _multiplicity,  # noqa: F401 (re-exported for tests)
    _scale_in_plane,  # noqa: F401 (re-exported for tests)
    _strain_film_onto,  # noqa: F401 (re-exported for tests)
    _supercell_transform,  # noqa: F401 (re-exported for tests)
    interface_areas,  # noqa: F401 (re-exported for tests)
)
from atominterface._coherent.config import CoherentInterfaceConfig
from atominterface._coherent.gates import (
    GeometryGateError,
    _check_area_conservation,  # noqa: F401 (re-exported for tests)
    _check_inplane_period,  # noqa: F401 (re-exported for tests)
    _shortest_inplane_period,  # noqa: F401 (re-exported for tests)
)
from atominterface._coherent.match import (
    _fallback_matches,  # noqa: F401 (re-exported for tests)
    _match_lattices,  # noqa: F401 (re-exported for tests)
    _MatchInfo,  # noqa: F401 (re-exported for tests)
)
from atominterface._coherent.placement import (
    _check_registry_placement,  # noqa: F401 (re-exported for tests)
    _off_on_top_shift,  # noqa: F401 (re-exported for tests)
    _registry_shift,  # noqa: F401 (re-exported for tests)
    _registry_target_frac,  # noqa: F401 (re-exported for tests)
    _stacking_period,  # noqa: F401 (re-exported for tests)
    _stacking_step,  # noqa: F401 (re-exported for tests)
    measure_registry,
)
from atominterface.builders.charge import SLAB_ID  # noqa: F401 (re-exported for tests)

__all__ = [
    "CoherentInterfaceConfig",
    "CoherentInterfaceBuilder",
    "GeometryGateError",
    "measure_registry",
    "reference_slabs",
]


# Register the coherent builder eagerly so `list-builders` works
# without importing the class explicitly.
try:
    from atominterface.builders.registry import builder_registry as _registry

    _registry.register(CoherentInterfaceBuilder.name, CoherentInterfaceBuilder)
except Exception:  # noqa: BLE001
    pass
