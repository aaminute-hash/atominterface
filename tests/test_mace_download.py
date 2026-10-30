"""Tests for the MACE default-weights download integrity check.

All network access is mocked: no test touches the real release asset.
"""

from __future__ import annotations

import io

import pytest

import atominterface.backends.mace_ as mace_module
from atominterface.backends.mace_ import _download_default_weights
from atominterface.provenance import hash_bytes


class _FakeResponse(io.BytesIO):
    """BytesIO dressed as a urlopen context-manager response."""


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    """Redirect the mace cache to a throwaway directory."""
    cache_dir = tmp_path / "mace-cache"
    cache_dir.mkdir()
    target = cache_dir / "2023_12_03mace128L1_epoch199.model"
    monkeypatch.setattr(mace_module, "mace_cache_path", lambda: target)
    return target


def _fake_urlopen(payload: bytes):
    def _open(url, timeout=120):  # noqa: ARG001
        return _FakeResponse(payload)

    return _open


def test_download_verifies_sha256_and_rejects_mismatch(fake_cache, monkeypatch):
    """Corrupt bytes + a set anchor -> RuntimeError and no cached leftover."""
    monkeypatch.setattr(mace_module, "MACE_MP0_MEDIUM_SHA256", hash_bytes(b"genuine-weights"))
    monkeypatch.setattr(mace_module, "urlopen", _fake_urlopen(b"corrupt-bytes"))

    with pytest.raises(RuntimeError, match="SHA-256 integrity check"):
        _download_default_weights()

    # Neither the target nor the .part sibling may survive.
    assert not fake_cache.exists()
    assert not fake_cache.with_name(fake_cache.name + ".part").exists()


def test_download_skips_verification_when_anchor_unset(fake_cache, monkeypatch):
    """Anchor None: the pre-verification behaviour (download succeeds)."""
    assert mace_module.MACE_MP0_MEDIUM_SHA256 is None  # current state of the anchor
    payload = b"some-checkpoint-bytes"
    monkeypatch.setattr(mace_module, "urlopen", _fake_urlopen(payload))

    result = _download_default_weights()

    assert result == fake_cache
    assert result.read_bytes() == payload


def test_cached_hit_bypasses_verification(fake_cache, monkeypatch):
    """A pre-existing cache file is returned untouched, never hashed."""
    fake_cache.write_bytes(b"pre-existing-cache")
    # Anchor that could never match the cached bytes: a cache hit must
    # not care.
    monkeypatch.setattr(mace_module, "MACE_MP0_MEDIUM_SHA256", "0" * 64)

    called = False

    def _unexpected_open(url, timeout=120):  # noqa: ARG001
        nonlocal called
        called = True
        raise AssertionError("urlopen must not be called on a cache hit")

    monkeypatch.setattr(mace_module, "urlopen", _unexpected_open)

    result = _download_default_weights()

    assert result == fake_cache
    assert result.read_bytes() == b"pre-existing-cache"
    assert called is False
