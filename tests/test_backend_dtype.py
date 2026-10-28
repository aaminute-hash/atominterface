"""dtype plumbing for the MACE backend (mocked, no mace-torch required).

These tests pin the float64 reference-run entry point: ``dtype`` must
reach ``mace_mp`` as its ``default_dtype`` (and the manual-checkpoint
fallback the same way), the default must stay ``"float32"`` so existing
numbers do not move, and unsupported values must raise ``ValueError``
before any heavy dependency is touched.

The ``mace_mp`` stand-in below deliberately mirrors the *real* helper's
signature instead of accepting ``**kwargs``.  An earlier version of this
file used ``def mace_mp(**kwargs)`` and asserted on a ``dtype`` key, which
pinned the wrong contract: the production code passed ``dtype=`` to a
helper whose parameter is spelled ``default_dtype``, the typo landed in
the helper's own ``**kwargs`` and was forwarded to ``MACECalculator``,
which ignored it -- so every "float64" run silently built a float32
model while the tests stayed green.  A mock that accepts anything cannot
catch a misspelled keyword, so this one cannot.
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
    """``mace_mp`` stand-in with the real helper's signature.

    Mirrors ``mace.calculators.mace_mp`` (mace-torch 0.3.x): the precision
    knob is ``default_dtype``.  There is intentionally **no** ``**kwargs``
    catch-all, so a call that spells the keyword any other way raises
    ``TypeError`` here exactly as it would against the real helper -- which
    is the whole point, since the real helper absorbs a stray ``dtype=``
    into its own ``**kwargs`` and fails silently instead.
    """

    def mace_mp(
        model=None,
        device="",
        default_dtype="float32",
        dispersion=False,
        dispersion_damping="bj",
        dispersion_xc="pbe",
        dispersion_cutoff=21.167088422553647,
        dispersion_coord_cutoff=21.167088422553647,
        return_raw_model=False,
    ):
        record.update(
            {
                "model": model,
                "device": device,
                "default_dtype": default_dtype,
                "dispersion": dispersion,
                "dispersion_damping": dispersion_damping,
                "dispersion_xc": dispersion_xc,
                "dispersion_cutoff": dispersion_cutoff,
                "dispersion_coord_cutoff": dispersion_coord_cutoff,
                "return_raw_model": return_raw_model,
            }
        )
        return object()

    return mace_mp


def test_load_default_passes_float32_when_dtype_omitted(monkeypatch):
    from atominterface.backends import mace_

    record: dict = {}
    _fake_mace_modules(monkeypatch, _recording_mace_mp(record))

    mace_._load_default(object, "cpu")

    assert record["model"] == "medium"
    assert record["device"] == "cpu"
    assert record["default_dtype"] == "float32"


def test_load_default_forwards_float64(monkeypatch):
    from atominterface.backends import mace_

    record: dict = {}
    _fake_mace_modules(monkeypatch, _recording_mace_mp(record))

    mace_._load_default(object, "cpu", dtype="float64")

    assert record["default_dtype"] == "float64"


def test_load_default_spells_the_helper_kwarg_default_dtype(monkeypatch):
    """Regression: the helper's knob is ``default_dtype``, not ``dtype``.

    ``_load_default`` used to call ``mace_mp(..., dtype=dtype)``.  The real
    helper takes ``default_dtype`` and swallows the stray ``dtype=`` in its
    own ``**kwargs``, handing it to ``MACECalculator`` -- which reads its
    ``default_dtype`` (still "float32") and returns a float32 model.  No
    exception, no warning: every "float64" run was quietly float32.

    The stand-in's signature has no ``**kwargs``, so the old spelling now
    raises ``TypeError``, which drops ``_load_default`` onto its tier-2
    manual-checkpoint fallback.  Guarding that fallback turns the
    regression into an immediate, network-free failure instead of a slow
    download attempt.
    """
    from atominterface.backends import mace_

    record: dict = {}
    _fake_mace_modules(monkeypatch, _recording_mace_mp(record))

    def unreachable():
        raise AssertionError("tier-1 must succeed; tier-2 fallback was reached")

    monkeypatch.setattr(mace_, "_download_default_weights", unreachable)

    mace_._load_default(object, "cpu", dtype="float64")

    assert record["default_dtype"] == "float64"
    # The misspelled key must not appear at all.
    assert "dtype" not in record


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

    assert record["default_dtype"] == "float64"
    assert record["model"] == "medium"


def test_manual_checkpoint_fallback_still_spells_default_dtype(monkeypatch):
    """Tier-2 (manual weights) keeps forwarding ``default_dtype``.

    ``_load_default`` falls back to downloading the canonical checkpoint
    and constructing the calculator directly when the ``mace_mp`` helper
    is unavailable.  That path was always correct, so pin it here too: a
    later "unify the two tiers" refactor must not break the one that
    worked while fixing the one that did not.
    """
    from atominterface.backends import mace_

    record: dict = {}
    monkeypatch.setattr(mace_, "_download_default_weights", lambda: "weights.model")
    monkeypatch.setitem(sys.modules, "mace", None)
    monkeypatch.setitem(sys.modules, "mace.calculators", None)

    class RecordingCalculator:
        def __init__(self, **kwargs):
            record.update(kwargs)

    mace_._load_default(RecordingCalculator, "cpu", dtype="float64")

    assert record["default_dtype"] == "float64"
    assert record["model_paths"] == ["weights.model"]
