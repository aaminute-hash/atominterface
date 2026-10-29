# AtomInterface

> **Reliable interface modeling for AI4S.**
> Coherent interface construction · uMLIP-guarded relaxation · interface thermodynamics · CZM export.

AtomInterface is a **middleware**, not a foundation model. It wraps the
tools you already use (ASE, pymatgen, MACE, CHGNet, ...) and adds the
unglamorous but critical layer: *is this interface candidate physical,
is the MLIP inside its training distribution, and where should I spend
my next DFT calculation?*

**What it is not:**

- not a new universal MLIP (we use MACE / CHGNet / MatterSim as backends)
- not an LLM agent that writes VASP input files
- not a notebook dump — `pip install -e .` and import it

## Install

```bash
# minimal: ASE + dummy backend, enough to run the quickstart guard demo
pip install -e .

# with pymatgen (needed for real coherent interface building)
pip install -e ".[pymatgen]"

# full, including optional MLIP backends and dev tools
pip install -e ".[all]"
```

## 5-minute quickstart

```bash
python examples/01_build_and_relax.py --backend dummy
python examples/01_build_and_relax.py --backend dummy --relax
```

This builds Li (bcc) | NaCl (rocksalt) interface candidates with
pymatgen's `CoherentInterfaceBuilder`, filters by von Mises strain,
then runs a guarded relaxation with the built-in dummy backend.
Outputs land in `outputs/quickstart_li_nacl/`.

> **Heads up:** Li and NaCl do **not** form a stable coherent
> interface — this example is a *code-path demonstration*, not a
> physical result. If ZSL finds no match, the script falls back to a
> manual slab stack and tells you so. For a real interface, use a
> low-mismatch pair (Li/LiH, Mg/MgO, Cu/Ni, ...).

## CLI

```bash
atominterface --version
atominterface builders list-builders

# Enumerate coherent interfaces for a substrate/film pair.
atominterface build examples/data/Cu_fcc.vasp examples/data/Ni_fcc.vasp \
    --sub-miller 0,0,1 --film-miller 0,0,1 \
    --max-strain 0.05 --max-area 100 \
    --max-candidates 5 --outdir outputs/build
```

`build` writes one structure file per candidate (`extxyz` by default, see
`--format`) plus a `candidates.csv` summary. Mismatch is reported as
`max(stretch, shear)` of the in-plane strain; the substrate cell is the
reference, so the whole mismatch sits on the film.

If every candidate is rejected, the command reports the closest lattice
matches it found and the `--max-strain` value that would keep them,
instead of just exiting empty.

With `build --relax`, each CSV/JSON row includes `valid`, `converged`,
finite-energy/force checks, structured calculator errors,
`committee_enabled`, and `ood_monitoring`. Any invalid relaxation is still
written for diagnosis, but the command exits non-zero. A score of zero with
`committee_enabled=false` means only geometry/force heuristics ran; it is not
proof that the structure is in-distribution.

The complete Cu/Ni energy ledger is also fail-closed:

```bash
# Wiring-only success path (dummy energies are not physical).
python examples/03_interface_energy.py --backend dummy --fmax 0.05

# Deliberate failure: candidates.csv is retained, gamma cells are empty,
# valid=False, and the process exits non-zero.
python examples/03_interface_energy.py --backend dummy --max-steps 1 --fmax 1e-6
```

Surface/interface energies are emitted only when the pristine and coherent
bulk reference energies are finite and both reference slabs plus the interface
pass convergence, OOD, exception, and finite energy/force checks.

Calculator construction failures (missing weights, unusable device) are
recorded as an invalid result with an explicit reason; the driver never
substitutes a stand-in calculator and never aborts on a bare traceback.

## Python API

```python
from atominterface.builders import CoherentInterfaceBuilder, BuilderConfig
from atominterface.backends import get_calculator
from atominterface.relax import GuardedRelaxer
from atominterface.thermo import interface_energy, fit_cohesive_zone

cfg = BuilderConfig(max_strain=0.05, gap=2.0)
builder = CoherentInterfaceBuilder(
    substrate="Li.cif",
    film="LiH.cif",
    substrate_miller=(0, 0, 1),
    film_miller=(0, 0, 1),
    config=cfg,
)
candidates = builder.build()  # list[Candidate]

calc = get_calculator("dummy")
relaxer = GuardedRelaxer(calculator=calc, ood_threshold=0.8)
result = relaxer.run(candidates[0].to_atoms())

if result.ood_triggered:
    print("OOD at step", result.ood_step, "->", result.quarantine_path)
elif not result.valid:
    print("invalid energy result:", result.exception or result.reasons)
else:
    print("finite converged energy:", result.final_energy)
```

## Architecture

```
               ASE Atoms (two bulks)
                        │
                        ▼
        ┌───────────────────────────────────┐
        │  Builders                         │
        │   coherent (pymatgen ZSL + match) │
        │   strain filter · termination log │
        └───────────────┬───────────────────┘
                        │  list[Candidate]
                        ▼
        ┌───────────────────────────────────┐
        │  Relax + Guard (ALWAYS ON)        │
        │   NeighborWatcher                 │
        │   ForceNormWatcher                │
        │   CommitteeWatcher (optional)     │
        │   quarantine → DFT single-point   │
        └───────────────┬───────────────────┘
                        │  relaxed Atoms + OOD report
                        ▼
        ┌───────────────────────────────────┐
        │  Thermo                           │
        │   interface energy                │
        │   work of separation              │
        │   CZM fitting                     │
        └───────────────┬───────────────────┘
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼               ▼
      report.md     DFT input      CZM → FEM
      candidates.csv (VASP/QE)    (MOOSE/FEniCSx)
      metadata.yaml
```

The key design choice: **the guard is not something you opt into**.
Every `GuardedRelaxer.run()` checks finite energies and forces and runs its
configured watchers. A result is `valid` only after finite evaluation and
convergence without OOD or calculator exceptions. Committee-based
distributional monitoring remains optional and is reported explicitly; the
default geometry/force heuristics must not be interpreted as an
in-distribution certificate. When a watched local atomic environment crosses
the OOD threshold, the guard pauses, cuts out the offending cluster, and
writes a DFT single-point input.

## Testing

```bash
pytest
pytest --cov=atominterface
```

Physical invariance tests are first-class:

- energy invariance under rigid rotation
- forces transform as vectors, stress as a rank-2 tensor
- energy derivative w.r.t. displacement equals negative force
- overlapping atoms are flagged by the neighbor watcher

## Documentation

| question | where |
|---|---|
| Get a number in five minutes | [`docs/quickstart.md`](docs/quickstart.md) |
| What do the outputs mean, and how much do I trust them? | [`docs/interpretation.md`](docs/interpretation.md) |
| Something failed, or a result looks wrong | [`docs/troubleshooting.md`](docs/troubleshooting.md) |
| How were these numbers validated? | [`docs/validation_cu_ni_001.md`](docs/validation_cu_ni_001.md) |
| Raw data to reproduce it myself | [`docs/validation/cu_ni_001_multi_candidate/`](docs/validation/cu_ni_001_multi_candidate/) |
| Contributing (read the testing rule before you write a mock) | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

One caveat worth repeating: **γ_int from the shipped defaults is biased by about
+0.005 J/m².** Add `--fmax 0.002` before quoting it. The measurement behind
that is in the validation note.

## Roadmap

| Phase | What | Status |
|-------|------|--------|
| 0.1 | Builder + guard + dummy backend + CLI + report | shipped (v0.1.0) |
| 0.2 | Charge balance / polar termination handling | shipped (v0.2.0) |
| 0.3 | Interface thermodynamics on a real system (Cu/Ni) | shipped (v0.3.0) |
| 0.4 | MD → stress/strain → FEniCSx constitutive pipeline | planned |

See `docs/roadmap.md` for the detailed plan.

## License

MIT. Model weights shipped by individual backends carry their own
licenses — check each backend's documentation before redistribution.
