# Troubleshooting

This note is about a specific failure mode this project takes seriously: **a
result that looks valid but was never actually measured**. Most of the entries
below distinguish three states that are easy to confuse:

| what you see | what it actually means |
|---|---|
| a number was produced | the computation ran |
| `valid = true` | the computation ran *and* every required quantity was finite and converged |
| a check reported "nothing" | often **no measurement happened**, not "nothing was found" |

---

## `ood_score = 0.000` and `committee_enabled = False`

**Symptom.** A run finishes cleanly and reports
`OOD: monitoring=heuristics_only committee_enabled=False score=0.000 triggered=False`.

**What it means.** The out-of-distribution check did **not run**. With a single
model there is nothing for a committee to disagree with, so the score is a
placeholder, not a passing grade. Treating `0.000` as "my structure is
in-distribution" is the most common misreading of this output.

**What to do.** Mount a second, independently trained model so the committee
has something to compare against:

```bash
atominterface build --relax --backend mace --committee-backend chgnet \
    Cu.cif Ni.cif --outdir outputs/build
```

The primary backend drives the relaxation; the committee member only votes.
Without `--committee-backend`, `ood_score` stays a placeholder by design.

**Related.** `docs/validation_cu_ni_001.md` §10 (Committee / OOD).

---

## `ood_score` is non-zero but nothing triggered

**Symptom.** The committee is enabled and reports a non-zero score (say 0.4)
yet `triggered = False`.

**What it means.** The committee and the heuristic watchers use **different
thresholds**, because they report **different units**. The committee's score is
a force disagreement in **eV/Å** and has its own threshold (0.15 eV/Å by
default); the heuristic watchers use a dimensionless score with
`ood_threshold` (0.8). A disagreement of 0.4 eV/Å is well above the
distribution-in baseline (0.03–0.07) but below the committee threshold only if
you lowered it — check which value your run used.

**What to do.** Read the run metadata for `committee_state` and the threshold
actually used. If you want a different sensitivity, set it explicitly; do not
assume `ood_threshold` also governs the committee.

---

## A candidate is `valid = false`, but the structure files were still written

**Symptom.** `candidates.csv` has a row with `status=failed` or `valid=false`
and empty gamma cells, yet `.extxyz` files for that candidate exist on disk.

**What it means.** This is deliberate, not a bug. Results are **fail-closed**:
when a node in the energy chain fails, the row and its structures are kept for
diagnosis but every gamma cell is left empty, and the driver exits non-zero. A
missing gamma is meant to be visibly different from a computed zero.

**What to do.** Open the candidate's structures and the run log to find which
node failed (bulk reference, slab, or interface). Re-running with a looser
`--fmax` or a larger `--max-steps` is a hypothesis, not a fix — confirm which
node failed first.

---

## The run exited non-zero, but the CSV has rows

**Symptom.** Non-zero exit code, yet `candidates.csv` was written.

**What it means.** Same fail-closed contract as above: the file is a diagnosis
artifact, not a result set. The exit code is the signal that at least one gamma
is missing.

**What to do.** Do not read the gamma columns of that run. Fix the failing
node, then re-run.

---

## An OOD trigger fired — where is the evidence?

**Symptom.** `triggered=True`, `reasons` mentions `committee_disagreement` or a
neighbour/force-norm reason.

**What it means.** The relaxation was stopped and the offending structure was
quarantined rather than silently continued.

**What to do.** Look for the quarantine directory under the output root: it
contains a `POSCAR` and an `INCAR`, i.e. a ready-made single-point input you can
hand to DFT. That is the intended next step — the point of the check is to tell
you *which* calculation deserves your DFT budget, not to stop you.

---

## `--dtype float64` changed nothing (v0.2.1)

**Symptom.** On **v0.2.1**, requesting float64 produced numbers identical to
float32.

**What it means.** A defect in that release: the backend passed `dtype=` to
`mace_mp()`, whose parameter is spelled `default_dtype`. The wrong name was
absorbed by that helper's `**kwargs`, so a float64 request silently built a
float32 model while the run metadata still recorded `dtype: float64`. Nothing
errored.

**What to do.** Upgrade to **v0.3.0 or later**, where the kwarg is correct and
the test that guards it uses the real signature. If you have results produced
with v0.2.1's float64 flag, treat them as float32.

---

## The first MACE run takes much longer than later ones

**Symptom.** First run downloads and compiles; subsequent runs are faster.

**What it means.** The pretrained weights (MACE-MP-0 medium, ~44 MB) are
fetched once and cached.

**What to do.** Nothing. If the machine is offline, run once on a networked
machine and copy the cache, or use `--backend dummy` for pipeline testing.

---

## `chgnet` backend is unavailable

**Symptom.** `unknown backend 'chgnet'`, or an import error even after
installing it.

**What it means.** Registration requires importing the backend module, and the
`chgnet` release compatible with ASE 3.29 may not install cleanly on some
Windows environments (older `chgnet` releases reference APIs that current ASE
removed).

**What to do.** Use another committee member that is installed (a second MACE
model of a different size counts — they are independently trained). Check
`discovery_failures` in the registry if you want the precise reason a backend
was skipped.

---

## A relaxation seems to hang

**Symptom.** No convergence, very long runtime, no error.

**What it means.** `max_steps` is a **total optimizer-step budget**, not a
per-call limit. Any backend whose residual-force floor sits above your `fmax`
will consume the whole budget before stopping.

**What to do.** Set `--max-steps` explicitly and check the reported terminal
`|F|max`. If it plateaus above your `fmax`, the backend cannot reach that
tolerance; loosen `fmax` rather than raising the step budget.

---

## Running the test suite locally on Windows

**Symptom.** `pytest` is interrupted mid-run with a non-zero exit code but no
failed test; or the run dies during cleanup.

**What it means.** A sandbox hook blocks bulk deletion in the default
temporary directory. The exit code is not a test failure.

**What to do.** Point pytest's temp dir at a drive that is not affected:

```bash
pytest --basetemp=/tmp/pt-run -q
```

---

## Imports resolve to an old copy of the package

**Symptom.** Behavior does not match the source you just edited; `version` in
the output metadata is stale.

**What it means.** An editable install pointing at a previous checkout is
shadowing your working tree.

**What to do.** Either reinstall the editable package from the current
checkout, or run with the source tree first on the path:

```bash
PYTHONPATH=<repo>/src python -m atominterface build ...
```
