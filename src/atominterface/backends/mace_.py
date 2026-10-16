"""MACE backend adapter (optional dependency)."""

from __future__ import annotations

from .base import Capability
from .registry import backend_registry


def _capability() -> Capability:
    return Capability(
        name="mace",
        supports_stress=True,
        supports_uncertainty=False,
        energy_unit="eV",
        force_unit="eV/A",
        stress_unit="eV/A^3",
        license="MIT",  # model weights have their own license
        notes="Wraps mace-torch MACECalculator as an ASE calculator.",
    )


def _factory(**kwargs):  # noqa: D401
    try:
        from mace.calculators import MACECalculator  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001
        raise ImportError("mace-torch is not installed") from exc

    def _calc(atoms):  # noqa: A002
        return MACECalculator(model_path=kwargs["model_path"], device=kwargs.get("device", "cpu"))

    return _calc


backend_registry.register("mace", _factory, _capability(), requires="mace")
