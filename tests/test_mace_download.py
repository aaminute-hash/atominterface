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
    """Anchor None: the pre-verification behaviour (download succeeds).

    The shipped default is a real hash, so this exercises the skip branch
    by forcing the anchor to ``None`` -- the state a build without a
    recorded anchor would be in.  Asserting on the live constant instead
    would turn a legitimate anchor update into a test failure.
    """
    monkeypatch.setattr(mace_module, "MACE_MP0_MEDIUM_SHA256", None)
    payload = b"some-checkpoint-bytes"
    monkeypatch.setattr(mace_module, "urlopen", _fake_urlopen(payload))

    result = _download_default_weights()

    assert result == fake_cache
    assert result.read_bytes() == payload


def test_shipped_anchor_is_a_full_lowercase_sha256():
    """The committed anchor must be a usable 64-char hex digest."""
    anchor = mace_module.MACE_MP0_MEDIUM_SHA256

    assert anchor is not None, "the anchor must ship with a real value"
    assert len(anchor) == 64
    assert set(anchor) <= set("0123456789abcdef"), "expected lowercase hex"


def test_download_rejects_bytes_that_do_not_match_the_shipped_anchor(fake_cache, monkeypatch):
    """A payload that is not the real checkpoint is rejected by the real anchor."""
    payload = b"definitely-not-the-real-mace-checkpoint"
    monkeypatch.setattr(mace_module, "urlopen", _fake_urlopen(payload))

    with pytest.raises(RuntimeError, match="SHA-256 integrity check"):
        _download_default_weights()

    assert not fake_cache.exists()


def test_shipped_anchor_matches_the_local_checkpoint_when_available():
    """End-to-end guard: hash the real cache file if this machine has one.

    Skipped on clean CI hosts.  This is the only test that can catch an
    anchor that is self-consistent inside the repository but wrong about
    the actual upstream artifact.
    """
    from atominterface.backends.mace_ import mace_cache_path

    cached = mace_cache_path()
    if not cached.is_file():
        pytest.skip(f"no cached MACE checkpoint at {cached}")

    assert hash_bytes(cached.read_bytes()) == mace_module.MACE_MP0_MEDIUM_SHA256


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
