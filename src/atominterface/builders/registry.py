"""Builder registry.

A tiny plugin mechanism: third-party builders can be registered under
a name and then selected from the CLI (``atominterface build --builder
<name>``).  This keeps the core from growing a monolithic if/elif
chain every time someone adds a new matching algorithm.
"""

from __future__ import annotations

from atominterface.builders.base import InterfaceBuilder


class BuilderRegistry:
    """Registry of named builder factories."""

    def __init__(self) -> None:
        self._builders: dict[str, type[InterfaceBuilder]] = {}

    def register(self, name: str, builder_cls: type[InterfaceBuilder]) -> type[InterfaceBuilder]:
        # name may come from a class attribute (e.g. cls.name), in
        # which case the decorator is invoked as @register_builder(cls)
        # and the intended name is on the class itself.
        resolved_name: str
        if isinstance(name, type) and issubclass(name, InterfaceBuilder):
            resolved_name = getattr(name, "name", name.__name__)
            builder_cls = name
        else:
            resolved_name = str(name)

        if resolved_name in self._builders:
            raise ValueError(f"Builder already registered: {resolved_name}")
        if not hasattr(builder_cls, "build"):
            raise TypeError(f"{builder_cls} does not implement build()")
        self._builders[resolved_name] = builder_cls
        return builder_cls

    def get(self, name: str) -> type[InterfaceBuilder]:
        if name not in self._builders:
            raise KeyError(f"Unknown builder '{name}'. Available: {sorted(self._builders)}")
        return self._builders[name]

    def list(self) -> dict[str, type[InterfaceBuilder]]:
        return dict(self._builders)


builder_registry = BuilderRegistry()


def register_builder(name: str):
    """Decorator form: ``@register_builder('my-builder')``."""

    def deco(cls: type[InterfaceBuilder]):
        return builder_registry.register(name, cls)

    return deco


def get_builder(name: str) -> type[InterfaceBuilder]:
    return builder_registry.get(name)


def autodiscover() -> int:
    """Execute builder registration modules.

    Registration is a module-level side effect, so we must import
    *through the package* (``from atominterface.builders.coherent
    import CoherentInterfaceBuilder``) rather than via importlib, which
    would load a second, independent module object when the package
    has already been partially imported.

    Returns the number of builder classes now registered.
    """
    before = len(builder_registry.list())
    for mod in ("atominterface.builders.coherent",):
        try:
            __import__(mod, fromlist=["__all__"])
            # Force re-execution of the module body only if it was
            # skipped due to a previous failed optional import.
            import importlib
            import sys

            if mod in sys.modules:
                importlib.reload(sys.modules[mod])
        except Exception:  # noqa: BLE001
            # Optional dependency missing (e.g. pymatgen) -- skip.
            pass
    return len(builder_registry.list()) - before
