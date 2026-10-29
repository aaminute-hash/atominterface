"""Interface builder subpackage.

The builder layer turns two bulk structures into a list of candidate
interfaces (``ase.Atoms``) ready for relaxation.  It is intentionally a
thin, opinionated wrapper around ``pymatgen``'s mature interface
generators: we add strain filtering, termination bookkeeping and a
uniform ``Candidate`` contract -- we do **not** re-implement lattice
matching.
"""

from atominterface.builders.base import (
    BuilderConfig,
    Candidate,
    InterfaceBuilder,
)
from atominterface.builders.charge import analyze_charge_polarity
from atominterface.builders.coherent import (
    CoherentInterfaceBuilder,
    CoherentInterfaceConfig,
    GeometryGateError,
    reference_slabs,
)
from atominterface.builders.registry import (
    BuilderRegistry,
    autodiscover,
    builder_registry,
    get_builder,
    register_builder,
)

# Eagerly execute module-level registrations (``@register_builder``
# decorators live in the builder modules and only fire on import).
autodiscover()

__all__ = [
    "BuilderConfig",
    "Candidate",
    "InterfaceBuilder",
    "CoherentInterfaceBuilder",
    "CoherentInterfaceConfig",
    "GeometryGateError",
    "analyze_charge_polarity",
    "BuilderRegistry",
    "builder_registry",
    "get_builder",
    "reference_slabs",
    "register_builder",
]
