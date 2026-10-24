"""Calculator backend subpackage."""

from atominterface.backends.base import CalculatorLike, Capability
from atominterface.backends.model_id import model_identity_for_meta, resolve_model_meta
from atominterface.backends.registry import (
    autodiscover,
    backend_registry,
    get_calculator,
    list_backends,
)

__all__ = [
    "Capability",
    "CalculatorLike",
    "backend_registry",
    "get_calculator",
    "list_backends",
    "autodiscover",
    "model_identity_for_meta",
    "resolve_model_meta",
]
