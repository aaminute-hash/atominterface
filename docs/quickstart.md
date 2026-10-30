# Quickstart

Three steps: prove the pipeline works (20 s), compute a real number (1 min),
and learn why the default parameters need one flag changed (1 min).

Every command below was run against **v0.3.0** on a Windows CPU machine with
`torch 2.14.1+cpu` and `mace-torch 0.3.17`. The timings and outputs shown are
real, not illustrative.

---

## 0. Install

```bash
pip install https://github.com/aaminute-hash/atominterface/releases/download/v0.3.0/atominterface-0.3.0-py3-none-any.whl
```

The MLIP backends are optional extras. For the MACE path below:

```bash
pip install mace-torch
```

The first MACE run downloads the pretrained weights (~44 MB) and caches them.

---

## 1. Prove the chain works — the dummy backend (≈ 20 s)

The dummy backend returns numbers designed to exercise the pipeline, not
physics. It answers one question only: *is the chain wired up?*

```bash
python examples/03_interface_energy.py \
    --backend dummy --fmax 0.05 --max-candidates 1 \
    --outdir outputs/quickstart_dummy
```

Real output (18.3 s):

```
E_slab  sub     :     0.0080 eV (valid=True)
E_slab film     :     0.0100 eV (valid=True)
E_interface     :     0.0180 eV (valid=True)
gamma_surf Cu   :     0.0000 eV/A^2 =   0.000 J/m^2
gamma_surf Ni   :     0.0000 eV/A^2 =   0.000 J/m^2
gamma_int       :     0.0000 eV/A^2 =   0.000 J/m^2
OOD             : monitoring=heuristics_only committee_enabled=False score=0.000 triggered=False

Wrote .../candidates.csv, metadata.yaml and structures to .../quickstart_dummy/
ledger   : 1/1 candidate(s) valid
note: dummy energies are not physical; the numbers only prove the chain.
```

Two things to read here:

- `1/1 candidate(s) valid` — the fail-closed ledger accepted the candidate.
- `committee_enabled=False score=0.000` — **no OOD measurement happened**. That
  is a placeholder, not a passing grade. See
  [`troubleshooting.md`](troubleshooting.md) before you interpret any
  `ood_score` from a single-model run.

---

## 2. A real number — MACE (≈ 50 s)

```bash
python examples/03_interface_energy.py \
    --backend mace --fmax 0.01 --max-candidates 1 \
    --outdir outputs/quickstart_mace
```

Real output (50.5 s):

```
gamma_surf Cu   :     0.0935 eV/A^2 =   1.498 J/m^2
gamma_surf Ni   :     0.1197 eV/A^2 =   1.918 J/m^2
gamma_int       :     0.0038 eV/A^2 =   0.062 J/m^2
ledger   : 1/1 candidate(s) valid
```

γ_Cu and γ_Ni are the numbers this project trusts. **γ_int is not, at this
setting** — read the next step before quoting it.

---

## 3. Why you should add `--fmax 0.002` (≈ 57 s)

Run the same command with a tighter force gate:

```bash
python examples/03_interface_energy.py \
    --backend mace --fmax 0.002 --max-candidates 1 \
    --outdir outputs/quickstart_mace_tight
```

Real output (56.9 s):

```
gamma_surf Cu   :     0.0935 eV/A^2 =   1.498 J/m^2
gamma_surf Ni   :     0.1197 eV/A^2 =   1.918 J/m^2
gamma_int       :     0.0035 eV/A^2 =   0.057 J/m^2
```

Compare:

| | `--fmax 0.01` | `--fmax 0.002` |
|---|---|---|
| γ_Cu (J/m²) | 1.498 | 1.498 |
| γ_Ni (J/m²) | 1.918 | 1.918 |
| **γ_int (J/m²)** | **0.062** | **0.057** |
| wall time | 50.5 s | 56.9 s |

The surface energies do not move. **γ_int moves by 0.005 J/m² — about 9% — and
it moves downward**, because at the looser gate the smallest supercells stop
short of convergence. Thirteen percent more time buys the trustworthy number.

This is the single most useful thing to know about this tool: γ_int is a small
difference between large terms, so it is far more sensitive to relaxation
convergence than the surface energies are. The measured basis for this is in
[`validation_cu_ni_001.md`](validation_cu_ni_001.md) §8, and the practical
rules are in [`interpretation.md`](interpretation.md).

**Quote 0.0567 ± 0.0002 J/m²** (the converged ten-candidate value), not the
0.062 you get from the defaults.

---

## 4. Enumerating candidates — the CLI (≈ 6 s)

The example script computes energies for one candidate set. To *search* for
candidates first, use the builder:

```bash
python -m atominterface build \
    --builder coherent --backend dummy --relax --max-area 40 \
    --outdir outputs/quickstart_build \
    examples/data/Cu_fcc.vasp examples/data/Ni_fcc.vasp
```

Real output (6.2 s) — ten candidates written, with a report:

```
Enumeration stopped at --max-candidates; the report covers attempted candidates.
Wrote 10 structure file(s), .../candidates.csv and JSON reports to .../quickstart_build/
```

Products: `build_report.json`, `candidates.csv`, `run_config.yaml`, and one
`.extxyz` per candidate. `run_config.yaml` records the effective parameters —
pass it back with `--config` to replay a run exactly.

To mount a second model so the OOD check actually measures something:

```bash
python -m atominterface build --relax \
    --backend mace --committee-backend chgnet \
    examples/data/Cu_fcc.vasp examples/data/Ni_fcc.vasp
```

---

## Where to go next

| question | document |
|---|---|
| What does this output actually mean? Is it trustworthy? | [`interpretation.md`](interpretation.md) |
| Something failed, or a result looks wrong | [`troubleshooting.md`](troubleshooting.md) |
| How were these numbers validated, and against what? | [`validation_cu_ni_001.md`](validation_cu_ni_001.md) |
| I want the raw data to reproduce it myself | [`validation/cu_ni_001_multi_candidate/`](validation/cu_ni_001_multi_candidate/) |
| I want the (111)/(110) orientations | [`validation/cu_ni_multi_orientation/`](validation/cu_ni_multi_orientation/) |
