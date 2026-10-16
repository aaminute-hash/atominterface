"""AtomInterface: reliable interface modeling for AI4S.

Public API surface.  Keep this narrow -- adding a symbol here is a
compatibility promise.
"""

__version__ = "0.1.0"

from atominterface.builders import (
    BuilderConfig,
    Candidate,
    CoherentInterfaceBuilder,
    InterfaceBuilder,
    builder_registry,
    get_builder,
    register_builder,
)

__all__ = [
    "__version__",
    "BuilderConfig",
    "Candidate",
    "InterfaceBuilder",
    "CoherentInterfaceBuilder",
    "builder_registry",
    "get_builder",
    "register_builder",
]
