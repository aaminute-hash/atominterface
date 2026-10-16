"""Tests for run provenance metadata."""

from atominterface.provenance import (
    RunMeta,
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
