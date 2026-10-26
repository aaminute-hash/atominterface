"""Smoke tests for the real, optional-dependency backends.

Every case that needs a heavy optional dependency begins with
``pytest.importorskip`` so the suite stays green where mace-torch, chgnet
or torch are absent: the tests skip instead of failing. The first MACE
run downloads the ~44 MB MACE-MP-0 medium checkpoint into the mace-torch
cache directory; later runs reuse it.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import pytest
from ase.build import bulk

from atominterface.backends import model_id
from atominterface.backends.model_id import MACE_MP0_MEDIUM_URL


def _copper_cell():
    """A minimal fcc Cu cell (4 atoms) exercising a pure-metal model.

    Two details make this a useful smoke fixture:

    * ``cubic=True``: ASE >= 3.23 builds the primitive cell by default,
      which for fcc is a single atom.
    * one atom is displaced by 0.05 A: atoms on ideal fcc sites carry
      the full cubic point-group symmetry, so a perfect cell returns
      *exactly* zero forces at any lattice constant. The displacement
      breaks that symmetry and lets the test tell real forces apart
      from a zero-filled array.
    """
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    atoms.positions[0] += [0.05, 0.0, 0.0]
    return atoms


def _assert_energy_and_forces(atoms, n_atoms: int) -> None:
    """Pin the shared smoke contract: finite non-zero energy, finite forces."""
    energy = atoms.get_potential_energy()
    assert math.isfinite(energy)
    assert energy != 0.0

    forces = atoms.get_forces()
    assert forces.shape == (n_atoms, 3)
    assert all(math.isfinite(value) for value in forces.flat)
    assert max(abs(float(value)) for value in forces.flat) > 1e-8


# ------------------------------------------------------------------
# dummy backend (always available, no heavy dependencies)
# ------------------------------------------------------------------
def test_dummy_backend_semantics():
    """Pin the documented dummy-backend contract."""
    from atominterface.backends import get_calculator

    atoms = _copper_cell()
    calc = get_calculator("dummy")(atoms)

    # _Dummy convention: 0.001 eV per atom, independent of the structure
    assert calc.get_potential_energy(atoms) == pytest.approx(0.001 * len(atoms))

    forces = calc.get_forces(atoms)
    assert forces.shape == (len(atoms), 3)
    assert all(math.isfinite(value) for value in forces.flat)

    stress = calc.get_stress(atoms)
    assert stress.shape == (6,)
    assert all(value == 0.0 for value in stress)

    # results mapping keeps ase.io writers working
    assert set(calc.results) >= {"energy", "forces", "stress"}


# ------------------------------------------------------------------
# MACE (needs mace-torch; first run downloads the default weights)
# ------------------------------------------------------------------
@pytest.mark.needs_mace
def test_mace_cpu_small_cell():
    """MACE-MP-0 medium on CPU: finite energy and non-trivial forces.

    The default weights are the MACE-MP-0 medium foundation model; the
    first invocation downloads the ~44 MB checkpoint into the mace-torch
    cache, subsequent runs reuse it.
    """
    pytest.importorskip("mace")

    from atominterface.backends import get_calculator

    atoms = _copper_cell()
    atoms.calc = get_calculator("mace")(atoms)
    _assert_energy_and_forces(atoms, len(atoms))


@pytest.mark.needs_gpu
def test_mace_gpu_small_cell():
    """MACE on CUDA when a device is available (same contract as CPU)."""
    pytest.importorskip("mace")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA-capable device available")

    from atominterface.backends import get_calculator

    atoms = _copper_cell()
    atoms.calc = get_calculator("mace", device="cuda")(atoms)
    _assert_energy_and_forces(atoms, len(atoms))


@pytest.mark.needs_mace
def test_mace_running_calculator_identity():
    """A live MACE run's identity pins model name and a 12-char hash."""
    pytest.importorskip("mace")

    from atominterface.backends import get_calculator

    atoms = _copper_cell()
    atoms.calc = get_calculator("mace")(atoms)
    identity = model_id.model_identity_for_meta("mace")
    assert identity.startswith("MACE-MP-0/medium@mace-torch:")
    assert ":sha256:" in identity
    assert len(identity.rsplit(":", 1)[1]) == 12


# ------------------------------------------------------------------
# CHGNet (needs chgnet; loads bundled pretrained weights)
# ------------------------------------------------------------------
def test_chgnet_cpu_small_cell():
    """Pretrained CHGNet on CPU: finite energy and non-trivial forces."""
    pytest.importorskip("chgnet")

    from atominterface.backends import autodiscover, backend_registry, get_calculator
    from atominterface.backends.registry import discovery_failures

    # ``importorskip`` only proves the *package* imports. The backend is
    # registered by importing ``atominterface.backends.chgnet_``, which can
    # fail independently -- and when it does the error used to be swallowed,
    # so this test failed with a bare "unknown backend 'chgnet'" even though
    # chgnet was installed. Skip on what is actually needed, and report why
    # registration did not happen.
    autodiscover()
    if "chgnet" not in backend_registry.list():
        reason = discovery_failures.get("chgnet_", "not registered")
        pytest.skip(f"chgnet backend unavailable: {reason}")

    atoms = _copper_cell()
    atoms.calc = get_calculator("chgnet")(atoms)
    _assert_energy_and_forces(atoms, len(atoms))


# ------------------------------------------------------------------
# provenance model identity (no heavy dependencies required)
# ------------------------------------------------------------------
def test_model_meta_dummy_has_empty_hash():
    """The dummy backend has no verifiable weights."""
    meta = model_id.resolve_model_meta("dummy")
    assert meta["backend"] == "dummy"
    assert meta["model_name"] == "dummy/reference"
    assert meta["model_hash"] == ""


def test_model_meta_unknown_backend_has_empty_hash():
    """Unknown backends degrade gracefully instead of raising."""
    meta = model_id.resolve_model_meta("not-a-backend")
    assert meta["backend"] == "not-a-backend"
    assert meta["model_hash"] == ""
    assert meta["model_source"] == ""


def test_model_meta_mace_hashes_url_when_weights_absent(monkeypatch):
    """Without cached weights the record still pins the exact artifact."""
    monkeypatch.setattr(
        model_id, "mace_cache_path", lambda: Path("/nonexistent/mace/cache/weights.model")
    )
    meta = model_id.resolve_model_meta("mace")
    assert meta["model_source"] == "url"
    assert meta["model_hash"] == (hashlib.sha256(MACE_MP0_MEDIUM_URL.encode()).hexdigest()[:12])


def test_model_meta_mace_hashes_cached_weights_when_present(monkeypatch, tmp_path):
    """When the default checkpoint is cached, its bytes are hashed.

    The cache directory is redirected to ``tmp_path`` via mace-torch's
    own ``get_cache_dir`` hook and a fake checkpoint file is planted
    under the sanitized URL basename, so the test exercises the real
    cache-path replication logic without depending on (or waiting for)
    a genuine ~44 MB download.
    """
    pytest.importorskip("mace")
    from mace.tools import utils as mace_utils

    monkeypatch.setattr(mace_utils, "get_cache_dir", lambda: str(tmp_path))
    stem = "".join(c for c in os.path.basename(MACE_MP0_MEDIUM_URL) if c.isalnum() or c in "_")
    payload = b"not-a-real-checkpoint"
    (tmp_path / stem).write_bytes(payload)

    meta = model_id.resolve_model_meta("mace")
    assert meta["model_source"] == MACE_MP0_MEDIUM_URL
    assert meta["model_hash"] == hashlib.sha256(payload).hexdigest()[:12]


def test_model_identity_string_matches_meta():
    """The convenience identity string mirrors resolve_model_meta."""
    meta = model_id.resolve_model_meta("mace")
    identity = model_id.model_identity_for_meta("mace")
    assert identity == (
        f"{meta['model_name']}@mace-torch:{meta['backend_version']}:sha256:{meta['model_hash']}"
    )


def test_model_identity_chgnet_string_matches_meta():
    """CHGNet identity mirrors its meta record (installed or not)."""
    meta = model_id.resolve_model_meta("chgnet")
    identity = model_id.model_identity_for_meta("chgnet")
    assert identity == (
        f"{meta['model_name']}@chgnet:{meta['backend_version']}:sha256:{meta['model_hash']}"
    )
