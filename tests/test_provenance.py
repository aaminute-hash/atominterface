"""Tests for run provenance metadata."""

from atominterface.provenance import (
    RunMeta,
    _collect_env,
    build_meta,
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
