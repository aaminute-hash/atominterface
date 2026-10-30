# Quickstart

Three steps: prove the pipeline works (20 s), compute a real number (1 min),
and learn why the default parameters need one flag changed (1 min).

Every command below was run on a Windows CPU machine with `torch 2.14.1+cpu`
and `mace-torch 0.3.17`. The timings and outputs shown are real, not
illustrative. They were captured on **v0.3.0** and **re-run and confirmed on
v0.5.0**: all nine numbers below still reproduce.

Two things about what "reproduce" does and does not mean, both measured on
v0.5.0 rather than asserted:

- **The (001) path is unchanged by v0.5.0.** v0.5.0 adds a `registry` field,
  default `"auto"`, which places the film on a named site instead of inheriting
  wherever the cut left it. On (001) the requested target is `(0, +½)` and the
  as-built site is `(0, −½)` — the same site modulo a lattice vector — so the
  builder moves the film by exactly zero (max |Δ| = 0.000e+00 Å on the
  unrelaxed structure, `--min-thickness` 10.8 to 21.6 Å). `registry=auto` and
  `registry=none` produce bit-identical *built* structures.
- **Relaxed outputs are not bit-reproducible, and never were.** Re-running the
  identical command twice yields structures differing by ~5e-6 Å (float32 on a
  multi-threaded CPU); even `bulk_sub.extxyz` changes hash between runs. The
  numbers below are stable to the digits shown, but the `.extxyz` files are not
  byte-identical run to run. Do not use file hashes as a regression gate.

Both caveats are (001)-specific. **On (110) v0.5.0 is a breaking change**:
`auto` moves the film from the ridge to the trough, and γ_int goes from 0.595
to 0.088 J/m². Read the [CHANGELOG](../CHANGELOG.md) before upgrading if you
have (110) numbers.

---

## 0. Install

```bash
pip install https://github.com/aaminute-hash/atominterface/releases/download/v0.5.0/atominterface-0.5.0-py3-none-any.whl
```

The MLIP backends are optional extras. For the MACE path below:

```bash
pip install mace-torch
```

The first MACE run downloads the pretrained weights (~44 MB) and caches them.

---

## 1. Prove the chain works — the dummy backend (≈ 13 s)

The dummy backend returns numbers designed to exercise the pipeline, not
physics. It answers one question only: *is the chain wired up?*

```bash
python examples/03_interface_energy.py \
    --backend dummy --fmax 0.05 --max-candidates 1 \
    --outdir outputs/quickstart_dummy
```

Real output (13.0 s):

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

## 2. A real number — MACE (≈ 58 s)

```bash
python examples/03_interface_energy.py \
    --backend mace --fmax 0.01 --max-candidates 1 \
    --outdir outputs/quickstart_mace
```

Real output (58.0 s):

```
gamma_surf Cu   :     0.0935 eV/A^2 =   1.498 J/m^2
gamma_surf Ni   :     0.1197 eV/A^2 =   1.918 J/m^2
gamma_int       :     0.0038 eV/A^2 =   0.062 J/m^2
ledger   : 1/1 candidate(s) valid
```

γ_Cu and γ_Ni are the numbers this project trusts. **γ_int is not, at this
setting** — read the next step before quoting it.

---

## 3. Why you should add `--fmax 0.002` (≈ 71 s)

Run the same command with a tighter force gate:

```bash
python examples/03_interface_energy.py \
    --backend mace --fmax 0.002 --max-candidates 1 \
    --outdir outputs/quickstart_mace_tight
```

Real output (71.2 s):

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
| wall time | 58.0 s | 71.2 s |

(Timings are from one machine under load — treat them as an order of magnitude,
not a benchmark. The energies are reproducible; the wall clock is not.)

The surface energies do not move. **γ_int moves by 0.005 J/m² — about 9% — and
it moves downward**, because at the looser gate the smallest supercells stop
short of convergence. Twenty-three percent more time buys the trustworthy
number.

This is the single most useful thing to know about this tool: γ_int is a small
difference between large terms, so it is far more sensitive to relaxation
convergence than the surface energies are. The measured basis for this is in
[`validation_cu_ni_001.md`](validation_cu_ni_001.md) §8, and the practical
rules are in [`interpretation.md`](interpretation.md).

**Quote 0.0567 ± 0.0002 J/m²** (the converged ten-candidate value), not the
0.062 you get from the defaults.

---

## 4. Enumerating candidates — the CLI (≈ 7 s)

The example script computes energies for one candidate set. To *search* for
candidates first, use the builder:

```bash
python -m atominterface build \
    --builder coherent --backend dummy --relax --max-area 40 \
    --outdir outputs/quickstart_build \
    examples/data/Cu_fcc.vasp examples/data/Ni_fcc.vasp
```

Real output (7.4 s) — ten candidates written, with a report:

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

## 5. Where the film sits — the `registry` field

v0.5.0 added `registry`, which names the lateral site the film's join layer
lands on. Before v0.5.0 the film inherited wherever the cut happened to leave
it — on (110) that is the ridge, a symmetry-locked saddle rather than the
scanned minimum.

| value | meaning |
|---|---|
| `auto` (default) | place on the site that continues the substrate stacking |
| `on-top` | directly above a substrate top-layer atom |
| `hcp` | the hollow that does *not* continue the stacking |
| `u,v` | explicit fractional coordinates in the substrate in-plane basis |
| `none` | pre-0.5.0 behaviour: leave the film where the cut put it |

On (001) `auto` is a no-op, which is why every number above is unchanged. On
(110) it moves the film 1.2781 Å, from ridge to trough. Whatever you ask for,
each candidate reports what it actually got — `registry_as_built_frac`,
`registry_target_frac`, `registry_lateral_offset` — so a request can be checked
against the result instead of trusted.

**This document re-measured (001) only.** No (110) or (111) number is printed
here, because none was produced for this page. For the full site table see
[`design/registry_field.md`](design/registry_field.md); for the measured
four-site scan on (110) see
[`validation/cu_ni_multi_orientation/`](validation/cu_ni_multi_orientation/).

---

## Where to go next

| question | document |
|---|---|
| What does this output actually mean? Is it trustworthy? | [`interpretation.md`](interpretation.md) |
| Something failed, or a result looks wrong | [`troubleshooting.md`](troubleshooting.md) |
| How were these numbers validated, and against what? | [`validation_cu_ni_001.md`](validation_cu_ni_001.md) |
| I want the raw data to reproduce it myself | [`validation/cu_ni_001_multi_candidate/`](validation/cu_ni_001_multi_candidate/) |
| I want the (111)/(110) orientations | [`validation/cu_ni_multi_orientation/`](validation/cu_ni_multi_orientation/) |
| Which lateral site does the film sit on, and how do I change it? | [`design/registry_field.md`](design/registry_field.md) |
| What changed in v0.5.0, and will it move my numbers? | [`CHANGELOG`](../CHANGELOG.md) |
