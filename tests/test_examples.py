"""Smoke tests for the examples (no heavy deps required)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
EX01 = ROOT / "examples" / "01_build_and_relax.py"


@pytest.mark.skipif(not EX01.exists(), reason="example missing")
def test_example_01_runs_with_dummy(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(EX01), "--backend", "dummy", "--outdir", str(tmp_path / "out")],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    # The script may exit non-zero if pymatgen is absent (graceful message),
    # or succeed if present.  In either case it must not import-crash.
    combined = proc.stdout + proc.stderr
    assert ("AtomInterface quickstart" in combined) or ("pymatgen" in combined)
    if proc.returncode == 0:
        assert (tmp_path / "out" / "candidates.csv").exists()
