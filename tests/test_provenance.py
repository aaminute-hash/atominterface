"""Tests for run provenance metadata."""

import numpy as np

import atominterface.provenance as provenance_module
from atominterface.provenance import (
    RunMeta,
    _collect_env,
    build_meta,
    current_seed,
    hash_bytes,
    write_metadata,
)


def test_hash_bytes_deterministic():
    assert hash_bytes(b"atominterface") == hash_bytes(b"atominterface")
    assert hash_bytes(b"x") != hash_bytes(b"y")


def test_run_meta_roundtrip_dict():
    m = RunMeta(backend="dummy", model_hash="abc123")
    d = m.to_dict()
    assert d["backend"] == "dummy"
    assert d["model_hash"] == "abc123"
    assert "env" in d


def test_write_metadata_yaml(tmp_path):
    m = build_meta(backend="dummy", model_hash="deadbeef", seed=42)
    p = write_metadata(m, tmp_path / "run_meta.yaml")
    assert p.exists()
    text = p.read_text()
    assert "deadbeef" in text
    assert "dummy" in text


def test_write_metadata_json_when_yaml_missing(tmp_path, monkeypatch):
    # simulate PyYAML missing
    monkeypatch.setitem(__import__("sys").modules, "yaml", None)
    m = RunMeta(backend="mace")
    p = write_metadata(m, tmp_path / "run_meta")
    assert p.suffix == ".json"
    assert p.exists()


def test_collect_env_values_are_plain_str():
    """Every env version must be exactly ``str``, not a str subclass.

    torch exposes ``__version__`` as ``TorchVersion`` (a str subclass);
    ``yaml.safe_dump`` looks representers up by exact type and refuses
    the subclass, which silently downgraded ``metadata.yaml`` to the
    JSON fallback whenever torch was importable.
    """
    env = _collect_env()
    assert env  # python/platform are always collected
    for key, value in env.items():
        assert type(value) is str, f"{key} is {type(value).__name__}"


# ------------------------------------------------------------------
# current_seed: state-fingerprint semantics (0.5.1)
# ------------------------------------------------------------------
def test_current_seed_deterministic_after_np_random_seed():
    """Same numpy state -> same fingerprint; different state -> different."""
    np.random.seed(42)
    first = current_seed()
    assert first is not None
    assert first > 0
    np.random.seed(42)
    assert current_seed() == first
    np.random.seed(43)
    assert current_seed() != first


def test_current_seed_none_when_numpy_unavailable(monkeypatch):
    """The pre-existing graceful path: no numpy state -> None."""
    monkeypatch.setattr(provenance_module, "np_state", lambda: None)
    assert current_seed() is None


def test_current_seed_survives_malformed_state(monkeypatch):
    """Any unexpected state shape degrades to None instead of raising."""
    monkeypatch.setattr(provenance_module, "np_state", lambda: {"state": "not-a-tuple"})
    assert current_seed() is None
    # A tuple whose first element is not the 'MT19937' algorithm name.
    monkeypatch.setattr(
        provenance_module,
        "np_state",
        lambda: {"state": (12345, b"words", 7, False, 0.0)},
    )
    assert current_seed() is None


def test_build_meta_auto_populates_seed_fingerprint():
    """build_meta() without a seed records the current state fingerprint."""
    np.random.seed(7)
    meta = build_meta(backend="dummy")
    assert meta.seed is not None
    assert meta.seed == current_seed()


def test_build_meta_explicit_seed_wins():
    """The explicit-seed contract is untouched by the auto-fill."""
    meta = build_meta(backend="dummy", seed=42)
    assert meta.seed == 42
