"""Shared pytest configuration for the atominterface test suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers (mirrors the pyproject.toml declarations).

    Registering here as well keeps the markers recognised even when the
    suite runs from a checkout whose ``pyproject.toml`` is not used as
    the pytest config file.
    """
    for marker, description in (
        (
            "needs_mace",
            "requires mace-torch; downloads the MACE-MP-0 medium weights on first use",
        ),
        ("needs_gpu", "requires a CUDA-capable device"),
    ):
        config.addinivalue_line("markers", f"{marker}: {description}")
