"""Backend registry and the built-in dummy backend.

Real backends (MACE, CHGNet, MatterSim, ...) are lazily imported so
the package remains importable in CI without heavy optional deps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Capability


@dataclass
class _DummyState:
    energy: float = 0.0
    forces: np.ndarray | None = None
    stress: np.ndarray | None = None
    seed: int | None = None


def _dummy_calc(atoms, state: _DummyState | None = None):  # noqa: D401
    """Return an ASE-compatible calculator-like object (pure numpy)."""
    st = state or _DummyState()

    class _Dummy:  # noqa: D401
        def __init__(self, atoms_ref, state_ref):
            self._atoms = atoms_ref
            self._state = state_ref

        def get_potential_energy(self, atoms, force_consistent=False):  # noqa: A003
            # ASE 3.29's Optimizer probes
            # atoms.get_potential_energy(force_consistent=True) and
            # forwards the kwarg here; rejecting it made every relaxation
            # die in the optimizer constructor.  The dummy backend has a
            # single energy surface, so the kwarg is accepted and ignored.
            n = len(atoms)
            return float(st.energy if n == 0 else st.energy + 0.001 * n)

        def get_forces(self, atoms):
            if st.forces is not None:
                return np.asarray(st.forces, dtype=float)
            rng = np.random.default_rng(st.seed)
            return rng.standard_normal((len(atoms), 3)) * 0.01

        def get_stress(self, atoms):
            if st.stress is not None:
                return np.asarray(st.stress, dtype=float)
            return np.zeros(6)

        @property
        def results(self):
            """Results mapping, so ASE writers can serialize the state.

            The relaxers only call the ``get_*`` methods, but writing a
            relaxed structure (``ase.io.write``) reads ``calc.results``;
            without this the documented ``--relax`` demo crashes.
            """
            atoms = self._atoms
            return {
                "energy": self.get_potential_energy(atoms),
                "free_energy": self.get_potential_energy(atoms),
                "forces": self.get_forces(atoms),
                "stress": self.get_stress(atoms),
            }

    return _Dummy(atoms, st)


class BackendRegistry:
    """Maps backend names to a factory that returns an ASE calculator."""

    def __init__(self) -> None:
        self._factories: dict[str, tuple] = {}

    def register(self, name: str, factory, capability: Capability, requires: str = "") -> None:
        self._factories[name] = (factory, capability, requires)

    def list(self) -> dict[str, Capability]:
        out: dict[str, Capability] = {}
        for name, (_, cap, _) in self._factories.items():
            out[name] = cap
        return out

    def get(self, name: str, **kwargs):  # noqa: D401
        if name not in self._factories:
            self._raise_unknown(name)
        factory, _, requires = self._factories[name]
        if requires:
            try:
                __import__(requires)  # noqa: F401
            except Exception as exc:  # noqa: BLE001
                raise ImportError(
                    f"backend {name!r} requires {requires!r} which is not installed"
                ) from exc
        return factory(**kwargs)

    def _raise_unknown(self, name: str) -> None:
        raise ValueError(f"unknown backend {name!r}; available: {sorted(self._factories)}")


backend_registry = BackendRegistry()


def _capability_dummy() -> Capability:
    return Capability(
        name="dummy",
        supports_stress=True,
        supports_uncertainty=False,
        energy_unit="eV",
        force_unit="eV/A",
        stress_unit="eV/A^3",
        license="MIT",
        notes="Zero-dependency reference backend for CI and demos; not physical.",
    )


def _factory_dummy(**kwargs):  # noqa: D401
    seed = kwargs.pop("seed", None)
    state = _DummyState(seed=seed)
    return lambda atoms: _dummy_calc(atoms, state)


backend_registry.register("dummy", _factory_dummy, _capability_dummy(), requires="")


def get_calculator(name: str = "dummy", **kwargs):  # noqa: D401
    """Resolve a backend name to an ASE calculator."""
    return backend_registry.get(name, **kwargs)


def list_backends() -> dict[str, Capability]:
    """Return capability table for all registered backends."""
    return backend_registry.list()


#: Import errors from :func:`autodiscover`, keyed by backend module name
#: (``"mace_"``, ``"chgnet_"``). Empty when every optional backend loaded.
discovery_failures: dict[str, str] = {}


def autodiscover():  # noqa: D401
    """Import optional backend modules so their register calls fire.

    Import errors are recorded in :data:`discovery_failures` instead of
    being silently swallowed. Silence is what made this hard to diagnose
    on CI: an optional backend that failed to import simply never got
    registered, and the only symptom was a much later
    ``unknown backend 'chgnet'`` error with no explanation, in a test
    whose ``importorskip`` had already passed.
    """
    import importlib

    for mod in ("mace_", "chgnet_"):
        try:
            importlib.import_module(f"atominterface.backends.{mod}")
        except Exception as exc:  # noqa: BLE001 - optional dependency
            discovery_failures[mod] = f"{type(exc).__name__}: {exc}"
