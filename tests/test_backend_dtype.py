"""dtype plumbing for the MACE backend (mocked, no mace-torch required).

These tests pin the float64 reference-run entry point: ``dtype`` must
reach ``mace_mp`` (and the manual-checkpoint fallback as
``default_dtype``) unchanged, the default must stay ``"float32"`` so
existing numbers do not move, and unsupported values must raise
``ValueError`` before any heavy dependency is touched.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
from ase.build import bulk


def _fake_mace_modules(monkeypatch, mace_mp):
    """Install a fake ``mace.calculators`` module exposing ``mace_mp``.

    Returns the fake calculators module so callers can also attach a
    ``MACECalculator`` stand-in when exercising the factory path.
    """
    fake_mace = ModuleType("mace")
    fake_mace.__path__ = []  # type: ignore[attr-defined]
    fake_calculators = ModuleType("mace.calculators")
    fake_calculators.mace_mp = mace_mp  # type: ignore[attr-defined]
    fake_mace.calculators = fake_calculators  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mace", fake_mace)
    monkeypatch.setitem(sys.modules, "mace.calculators", fake_calculators)
    return fake_calculators


def _recording_mace_mp(record: dict):
    def mace_mp(**kwargs):
        record.update(kwargs)
        return object()

    return mace_mp


def test_load_default_passes_float32_when_dtype_omitted(monkeypatch):
    from atominterface.backends import mace_

    record: dict = {}
    _fake_mace_modules(monkeypatch, _recording_mace_mp(record))

    mace_._load_default(object, "cpu")

    assert record["model"] == "medium"
    assert record["device"] == "cpu"
    assert record["dtype"] == "float32"


def test_load_default_forwards_float64(monkeypatch):
    from atominterface.backends import mace_

    record: dict = {}
    _fake_mace_modules(monkeypatch, _recording_mace_mp(record))

    mace_._load_default(object, "cpu", dtype="float64")

    assert record["dtype"] == "float64"


def test_load_default_rejects_unknown_dtype(monkeypatch):
    from atominterface.backends import mace_

    record: dict = {}
    _fake_mace_modules(monkeypatch, _recording_mace_mp(record))

    with pytest.raises(ValueError, match="unsupported dtype 'float16'"):
        mace_._load_default(object, "cpu", dtype="float16")
    # Validation must fire before the helper is called at all.
    assert record == {}


def test_instantiate_forwards_default_dtype():
    from atominterface.backends import mace_

    record: dict = {}

    class RecordingCalculator:
        def __init__(self, **kwargs):
            record.update(kwargs)

    mace_._instantiate(RecordingCalculator, "cpu", model_path="weights.model")
    assert record["default_dtype"] == "float32"

    record.clear()
    mace_._instantiate(RecordingCalculator, "cpu", "float64", model_path="weights.model")
    assert record["default_dtype"] == "float64"


def test_instantiate_rejects_unknown_dtype():
    from atominterface.backends import mace_

    with pytest.raises(ValueError, match="supported values"):
        mace_._instantiate(object, "cpu", "bfloat16", model_path="weights.model")


def test_factory_rejects_unknown_dtype_before_importing_mace(monkeypatch):
    """A bad dtype raises ValueError even where mace-torch is absent."""
    from atominterface.backends import mace_

    monkeypatch.delitem(sys.modules, "mace", raising=False)
    monkeypatch.delitem(sys.modules, "mace.calculators", raising=False)

    with pytest.raises(ValueError, match="unsupported dtype"):
        mace_._factory(dtype="float16")


def test_registry_passes_dtype_through_to_mace_mp(monkeypatch):
    """``get_calculator("mace", dtype="float64")`` reaches the helper."""
    import atominterface.backends.mace_  # noqa: F401  (fires backend registration)
    from atominterface.backends import get_calculator

    record: dict = {}
    fake_calculators = _fake_mace_modules(monkeypatch, _recording_mace_mp(record))

    class FakeMACECalculator:
        def __init__(self, **kwargs):
            raise AssertionError("default-model path must use mace_mp, not MACECalculator")

    fake_calculators.MACECalculator = FakeMACECalculator  # type: ignore[attr-defined]

    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    get_calculator("mace", dtype="float64")(atoms)

    assert record["dtype"] == "float64"
    assert record["model"] == "medium"
