"""CHGNet backend adapter (optional dependency)."""

from __future__ import annotations

from typing import Any

from .base import Capability
from .registry import backend_registry


def _capability() -> Capability:
    return Capability(
        name="chgnet",
        supports_stress=True,
        supports_uncertainty=False,
        energy_unit="eV",
        force_unit="eV/A",
        stress_unit="eV/A^3",
        license="MIT",  # model weights have their own license
        notes="Wraps the pretrained CHGNet model as an ASE calculator.",
    )


def _load_pretrained(cls: Any) -> Any:
    """Load CHGNet with its bundled pretrained weights.

    ``CHGNet.load()`` with no arguments loads the default pretrained
    checkpoint. Releases differ in which extra keyword arguments are
    required, so the plain no-arg call is tried first and the legacy
    ``verbose=False`` spelling second.

    Args:
        cls: The ``CHGNet`` class.

    Returns:
        A loaded CHGNet model instance.

    Raises:
        TypeError: If no supported ``load`` spelling is accepted.
    """
    try:
        return cls.load()
    except TypeError:
        return cls.load(verbose=False)


def _factory(**kwargs):  # noqa: D401
    try:
        from chgnet.model.dynamics import CHGNetCalculator  # type: ignore[import-untyped]
        from chgnet.model.model import CHGNet  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001
        raise ImportError("chgnet is not installed; install it with `pip install chgnet`") from exc

    device = kwargs.get("device")
    device = str(device) if device is not None else None

    def _calc(atoms):  # noqa: A002
        model = _load_pretrained(CHGNet)
        if device is None:
            return CHGNetCalculator(model=model)
        try:
            return CHGNetCalculator(model=model, use_device=device)
        except TypeError:
            # releases without the use_device kwarg: keep the model device
            return CHGNetCalculator(model=model)

    return _calc


backend_registry.register("chgnet", _factory, _capability(), requires="chgnet")
