# Contributing

## Getting started

```bash
git clone https://github.com/<you>/atominterface.git
cd atominterface
pip install -e ".[dev]"
pre-commit install   # optional but recommended
```

Run the test suite:

```bash
pytest --basetemp=/tmp/pt-run -q
ruff check src tests examples
mypy src/atominterface
```

`--basetemp` pointing somewhere other than the default temporary directory is
not optional on every machine: a sandbox hook blocks bulk deletion there, which
interrupts the run and returns a non-zero exit code even when every test
passed. If pytest dies during cleanup with no failing test, this is why.

If behaviour does not match the source you just edited, an editable install
from a previous checkout is probably shadowing it. Either reinstall from the
current checkout or put the source tree first:

```bash
PYTHONPATH=<repo>/src python -m atominterface build ...
```

## What we value

1. **Physical correctness over feature count.** A new builder with a
   rotation-invariance test is worth more than three untested features.
2. **Lazy imports.** Optional dependencies (pymatgen, mace, chgnet)
   must never break `import atominterface` on a minimal install.
3. **No new data structures.** Structures are `ase.Atoms`, tensors are
   numpy arrays or torch tensors. Do not invent a new container.
4. **Clear failure modes.** When something goes wrong, tell the user
   what to install or which parameter to relax — no raw stack traces.

## Adding a backend

1. Create `src/atominterface/backends/<name>.py`.
2. Define a `Capability` and a factory returning an ASE calculator.
3. Register it with `backend_registry.register(...)`.
4. The module is imported lazily by `autodiscover()` — do not import
   the heavy library at module level.
5. Add a test that the backend is listed when its dependency is
   present, and skipped gracefully when it is not.

## Adding a watcher

A watcher is a callable `(atoms, forces) -> WatchResult`. It must:

- be stateless (all state lives in the relaxer)
- return a score in `[0, 1]`
- run in O(N) or better
- have a test with a hand-constructed pathological structure

## Commit messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat(relax): add committee watcher`
- `fix(builders): handle 2x2 lattice in fallback path`
- `test(physics): rotation invariance of stress`
- `docs: clarify Li/NaCl is a demo system`

## Tests must be falsifiable — a case in point

This rule came out of a real defect, and it is worth reading before you write a
mock.

The MACE backend accepted a `dtype` argument. A test mocked the underlying
factory and asserted the argument arrived:

```python
def _recording_mace_mp(record):
    def mace_mp(**kwargs):          # accepts any keyword name
        record.update(kwargs)
```

The assertion was `record["dtype"] == "float64"`. Six tests passed. The feature
was dead: the real factory's parameter is spelled `default_dtype`, so our
`dtype=` fell into `**kwargs`, was ignored, and a float64 request silently
returned a float32 model. Nothing errored, and the run metadata still recorded
`dtype: float64` — so anyone reproducing that run would believe they had run
double precision.

The mock could not have caught it, because **it accepted every keyword name**.
It was verifying that we sent *something called* `dtype`, not that we sent what
the callee actually reads.

The fix included rewriting the mock to spell the real signature, with no
`**kwargs`:

```python
def mace_mp(*, model, device, default_dtype, dispersion, ...):
    ...
```

Now a wrong name raises `TypeError` and fails loudly.

**The rule:** when you mock a third-party function whose signature has named
parameters, the mock must reproduce those names. A `**kwargs` catch-all
permanently exempts "we spelled the argument wrong" from ever failing a test —
which is exactly the bug class this project exists to prevent.

## Red lines

Do not change these without an explicit decision:

- `src/atominterface/relax/` — the guarded relaxation layer. Freeze it or get
  an exemption; changes there alter what "converged" and "safe" mean.
- dependency ranges in `pyproject.toml`
- `LICENSE`

## Documentation and archived data

- **Numbers in docs must be reproducible from the archived data.** Prefer
  values computed by script over values copied by hand. A hand-written summary
  table once disagreed with the per-candidate data by 0.0001 because some cells
  were truncated and others rounded; the docs now use the recomputed values and
  say so.
- **Archived data is copied verbatim.** Verify `sha256` against what the run
  recorded. Do not edit artifacts to make a narrative tidier.
- **Unmeasured is not passing.** If a check did not run (no committee mounted,
  a backend missing), say it did not run. An `ood_score` of `0.000` with
  `committee_enabled=False` means no measurement, not a clean result.
- **State the limits.** Known limitations belong in the document a reader will
  actually open, not only in a future changelog.

## License

By contributing you agree that your contributions are licensed under
the project's MIT license.
