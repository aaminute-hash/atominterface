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
pytest
ruff check src tests examples
mypy src/atominterface
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

## License

By contributing you agree that your contributions are licensed under
the project's MIT license.
