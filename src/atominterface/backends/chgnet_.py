"""CHGNet backend adapter (optional dependency)."""

from __future__ import annotations

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
        notes="Wraps CHGNetPretrained as an ASE calculator.",
    )


def _factory(**kwargs):  # noqa: D401
    try:
        from chgnet.model.model import CHGNet  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001
        raise ImportError("chgnet is not installed") from exc

    def _calc(atoms):  # noqa: A002
        model = CHGNet.load(**kwargs)
        return model.get_calculator()

    return _calc


backend_registry.register("chgnet", _factory, _capability(), requires="chgnet")
