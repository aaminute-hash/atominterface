# Validation note 001 — Cu(001) \| Ni(001) interface energetics

**Status:** measured, not modelled. Every number below comes from
`examples/03_interface_energy.py` on the shipped defaults.

**Question this note answers:** does the energy ledger in AtomInterface
produce interface and surface energies that *converge* with slab
thickness, and are they *close* to accepted DFT values?

Short answer: yes to convergence (spread ≤ 0.003 J/m² across 4 → 12 Cu
layers), and yes to accuracy at the ~5–15 % level expected of a
general-purpose foundation uMLIP — but the interface energy is a small
difference of two large numbers, so its *relative* uncertainty is an
order of magnitude worse than either surface energy's. That is
quantified in [§8](#8-why-gamma_int-is-the-weakest-number-here).
[§7](#7-vasp-cross-check-co-computed-reference) replaces the
literature-only comparison in §6 with a co-computed plane-wave
reference, and shows the gap is almost entirely model error, not a
defect in our estimator.

---

## 1. Setup

| item | value |
|---|---|
| driver | `examples/03_interface_energy.py` |
| backend | MACE-MP-0 **medium**, `mace-torch` 0.3.16, `sha256:01bfe2210013` |
| device | CPU |
| seed | 0 (fixed; recorded in `metadata.yaml`) |
| relaxed structures | FIRE, `fmax = 0.01` eV/Å, `max_steps = 200` |
| vacuum | 15 Å (thickness scan), 10–25 Å (vacuum scan) |
| gap | 2.0 Å |
| cell | 1×1 coherent (001), area **6.5341 Å²**, one atom per layer |
| in-plane lattice | 2.5562 Å = a_Cu/√2, a_Cu = 3.6150 Å (from `examples/data/Cu_fcc.vasp`) |
| mismatch | Cu **0.000000**, Ni **+0.025823** |
| environment | Python 3.13.14, ASE 3.29.0, torch 2.14.1+cpu, Windows 11 |

Unit conversion: 1 eV/Å² = 16.02176634 J/m² (`atominterface.physics.EV_A2_TO_J_M2`).

## 2. The ledger

```
E_term(A) = (E_slab(A) - n_A * E_bulk(A)) / (2 * area)        = gamma_A
E_term(B) = (E_slab(B) - n_B * E_bulk(B)) / (2 * area)        = gamma_B
gamma_int = (E_int - E_slab(A) - E_slab(B)) / area + gamma_A + gamma_B
          = gamma_A + gamma_B - W_ad / area
```

`n_surfaces = 2` for both reference slabs (asymmetric slab, two exposed
faces). The bulk term cancels between the interface and the two
reference slabs, so `gamma_int` is independent of *which* bulk
reference is used **only if the same reference is used on both sides
and the reference slabs carry the same layer counts as the interface**.

The driver now treats that dependency graph as a validity contract, not just
an equation. `valid=true` requires finite energies for both pristine and
coherent bulk references; all three guarded relaxations (both reference slabs
and the interface) must have finite energies and forces and must converge
without OOD or a calculator exception. If any node fails, the row and structures remain available for
diagnosis, but all γ cells are empty and the process exits non-zero. A missing
γ is therefore distinct from a physically computed value of zero.

## 3. Thickness convergence (the headline table)

`--min-thickness` swept, vacuum fixed at 15 Å:

| min-thickness (Å) | Cu layers | Ni layers | γ_Cu (J/m²) | γ_Ni (J/m²) | γ_int (J/m²) | converged |
|---|---|---|---|---|---|---|
| 7.2 | 4 | 6 | 1.4984 | 1.9210 | 0.0562 | yes |
| 10.8 | 6 | 8 | 1.4977 | 1.9181 | 0.0573 | yes |
| 14.4 | 8 | 10 | 1.4977 | 1.9183 | 0.0617 | yes |
| 18.0 | 10 | 12 | 1.4978 | 1.9184 | 0.0620 | yes |
| 21.6 | 12 | 14 | 1.4984 | 1.9187 | 0.0581 | yes |

| quantity | mean | sd | spread (max − min) |
|---|---|---|---|
| γ_Cu | 1.4980 | 0.0004 | **0.0008** |
| γ_Ni | 1.9189 | 0.0012 | **0.0030** |
| γ_int | 0.0590 | 0.0026 | **0.0058** |

All three are flat and non-monotonic — the signature of a converged
quantity with residual noise, not of a systematic drift.

## 4. Vacuum convergence

Slab thickness fixed at 14.4 Å, vacuum swept:

| vacuum (Å) | γ_Cu (J/m²) | γ_Ni (J/m²) | γ_int (J/m²) |
|---|---|---|---|
| 10 | 1.4977 | 1.9183 | 0.0617 |
| 15 | 1.4977 | 1.9183 | 0.0617 |
| 20 | 1.4977 | 1.9183 | 0.0617 |
| 25 | 1.4977 | 1.9183 | 0.0617 |

Identical to the written precision: 10 Å of vacuum is already enough
for this system, so the 15 Å default carries margin.

## 5. The defect this run exposed (and fixed)

The first scan produced a γ_Ni that climbed monotonically with layer
count instead of converging — 1.9996 → 2.0221 → 2.0492 → 2.0785 →
2.1026 J/m². Two separate causes, both fixed:

**(a) The bulk reference was the wrong crystal.** A coherent slab is
built on the *interface* in-plane lattice, so the film side carries the
mismatch strain (+2.58 % for Cu/Ni) while the substrate does not.
Referencing a strained slab against the *pristine* bulk leaves
`n × ΔE_strain` inside the termination energy — a term that grows with
layer count, which is exactly the observed drift. The fix
(`_coherent_bulk_energy`) locks the reference to the slab's in-plane
strain and relaxes the out-of-plane response. Measured strain
self-energies: Cu **−0.00035** eV/atom (essentially none; it carries no
strain), Ni **+0.01057** eV/atom.

The out-of-plane minimum is found with a **local three-point parabola**
around the discrete grid minimum, not a fit across the whole grid. Over
a 10 % window E(s_z) is strongly anharmonic: the wide fit reported
+0.57 meV/atom of relaxation gain for Cu where a 61-point dense scan
finds +0.35. The local fit reproduces the dense-scan minimum to within
0.008 meV/atom on both metals.

**(b) `fmax = 0.05` eV/Å was too loose.** The force driving a Cu slab's
out-of-plane expansion is itself ≈ 0.05 eV/Å, i.e. right at the
tolerance, so the interior never reached the spacing the reference
assumes. Measured central interlayer spacing in the Cu reference slab:
1.8075 Å at `fmax = 0.05` (stuck at the pristine value) versus
1.8136–1.8183 Å at `fmax = 0.01`, against a bulk optimum of 1.8219 Å.
Ni goes to 1.686–1.690 Å against a coherent optimum of 1.6880 Å. The
demo default is now `--fmax 0.01`.

| configuration | γ_Cu spread | γ_Ni spread | γ_int spread |
|---|---|---|---|
| pristine reference, `fmax 0.05` (before) | 0.0012 | **0.1031** | **0.1183** |
| coherent reference, `fmax 0.05` | 0.0046 | 0.0039 | 0.0178 |
| coherent reference, `fmax 0.01` (shipped) | **0.0008** | **0.0030** | **0.0058** |

Note the middle row: fixing the reference alone makes γ_Ni converge but
leaves γ_Cu drifting, because the slab and the reference now disagree
about the out-of-plane spacing. Both fixes are needed.

## 6. Comparison with DFT

| quantity | this work | DFT literature | ratio |
|---|---|---|---|
| γ_Cu(001) | **1.498 ± 0.001** | 1.29 – 1.44 J/m² | +4 % to +17 % |
| γ_Ni(001) | **1.919 ± 0.002** | 1.98 – 2.23 J/m² | −3 % to −14 % |

Literature band from four independent PBE/plane-wave sources: Cu(100)
1.34, 1.44, 1.36 and 1.285 J/m²; Ni(100) 2.23, 2.22, 1.98 and
2.147 J/m².

Both offsets point in a physically explicable direction. MACE-MP-0 is a
general-purpose foundation model whose training set (MPtrj) contains few
surfaces, so ~10 % surface-energy error is the expected accuracy rather
than a bug. The Ni value is additionally *not* the quantity in the
literature table: our Ni is stretched +2.58 % onto Cu's lattice, and
biaxial tension lowers a surface energy, so a value below the unstrained
DFT band is the expected sign.

> **§7 qualifies both statements above.** The co-computed VASP reference
> reproduces γ_Cu to 0.3 % but γ_Ni only to 14 %, and it places VASP's
> γ_Ni *inside* this literature band with MACE outside it. See
> [§7.3](#73-where-does-the-14--on-γ_ni-come-from-two-hypotheses-both-killed)
> and [§7.6](#76-what-this-changes-about-6).

## 7. VASP cross-check (co-computed reference)

§6 compares against *published* DFT. This section compares against DFT
computed on **the same geometries**, which removes every difference in
slab construction, strain state and reference definition and leaves only
the potential itself.

### 7.1 Setup

| item | value |
|---|---|
| code | VASP 6.4.2 (intelmpi 2017, `optcell_smart` build) |
| pseudopotentials | PAW_PBE **Cu 22Jun2005**, **Ni 02Aug2007** |
| plane-wave cutoff | ENCUT 520 eV, PREC = Accurate |
| SCF | EDIFF 1e-6, LREAL = .FALSE., ALGO = Normal |
| spin | ISPIN = 2 (control: ISPIN = 1) |
| smearing | ISMEAR = 1, SIGMA = 0.1 (primary); ISMEAR = −5 tetrahedron (control) |
| k-mesh | Γ-centred 12×12×1 for slabs/interface, 12×12×12 for bulk |
| dipole correction | LDIPOL = .TRUE., IDIPOL = 3 (interface only) |
| parallelism | KPAR = 4, NCORE = 4, 64 cores |
| geometries | the MACE-relaxed structures, **re-used verbatim** — no re-relaxation |

23 single points: interface, Cu slab, Ni slab, Cu bulk, Ni bulk, and
9-point normal-strain grids for both coherent bulk references.

**Energy convention.** Every number is VASP's `energy(sigma->0)`, *not*
the `free energy TOTEN` line. The two differ by −T·S, and that term is
**not constant across these cells** — it ranges from −1.9 meV (Cu
substrate slab) to +0.2 meV (Ni coherent cells). Mixing the two
definitions inside one energy difference injects ≈ 0.002 J/m² into
γ_Cu, which is half the entire γ_Cu discrepancy being measured. Only
`sigma->0` is comparable with the tetrahedron set, which has no entropy
term at all.

### 7.2 Results

| quantity | VASP | MACE | Δ | Δ% |
|---|---|---|---|---|
| γ_Cu (J/m²) | **1.5023** | 1.4980 | +0.0043 | **+0.3 %** |
| γ_Ni (J/m²) | **2.1933** | 1.9189 | +0.2744 | **+14.3 %** |
| W_ad (J/m²) | **3.4018** | 3.3579 | +0.0439 | +1.3 % |
| γ_int (J/m²) | **0.2938** | 0.0590 | +0.2348 | +398 % |
| strain self-energy, Cu (eV/atom) | −0.00098 | −0.00035 | −0.00063 | — |
| strain self-energy, Ni (eV/atom) | **+0.01081** | +0.01057 | +0.00024 | **+2.3 %** |

The coherent-reference machinery transfers cleanly to plane-wave DFT:
VASP and MACE agree on the Ni strain self-energy to 2.3 %, and the
optimal out-of-plane strain agrees to within one grid step
(−0.030 vs −0.030 for Ni). **The estimator is not in question.**

### 7.3 Where does the 14 % on γ_Ni come from? Two hypotheses, both killed

| hypothesis | test | shift in γ_Ni | share of the 0.2744 gap |
|---|---|---|---|
| Fermi-smearing error (Ni has high DOS at E_F) | `ISMEAR=-5` tetrahedron, entire Ni branch re-run (11 jobs) | **−0.0019** | **−0.7 %** |
| MACE is spin-unpolarised, VASP is not | `ISPIN=1` on the same geometries | +0.0156 | +5.7 % |
| **residual = MACE model error** | — | **+0.2607** | **95.0 %** |

Neither explanation survives. The tetrahedron method moves γ_Ni by
0.002 J/m² — two orders of magnitude less than the gap — so the
smearing choice is exonerated. Removing spin *raises* VASP's γ_Ni to
2.2089, i.e. magnetism pushes VASP **towards** MACE, not away from it;
it cannot be the source of a gap in the opposite direction.

The verdict is therefore that **MACE-MP-0 under-predicts the Ni(001)
surface energy by ~14 %**, and our VASP value of 2.1933 J/m² sits
inside the literature band quoted in §6 (1.98–2.23), while MACE's
1.9189 sits *below* it. This upgrades §6's "expected uMLIP error" from
a plausible story to a measured one, and it moves the burden of proof:
the number to distrust is MACE's γ_Ni, not our pipeline's.

*Caveat on the controls:* `ISMEAR=-5` and `ISPIN=1` were run on the Ni
branch only. W_ad and γ_int are deliberately **not** reported for them,
because the interface energy comes from the primary setting and mixing
the two Hamiltonians inside one difference would be meaningless.

### 7.4 The γ_int gap is inherited, not interfacial

γ_int = γ_Cu + γ_Ni − W_ad, so the three error contributions add
directly:

```
d(γ_Cu)   +0.0043
d(γ_Ni)   +0.2744     <-- 117 % of the total gap
−d(W_ad)  −0.0439
────────────────────
sum       +0.2348   ==  direct gap +0.2348   (exact)
```

**W_ad — the actual Cu–Ni bonding term — agrees to 1.3 %.** The 5×
discrepancy in γ_int is not a failure to describe the interface; it is
the Ni surface error passing straight through a near-cancelling
difference. This is the operational content of §8, now measured rather
than argued: anyone quoting γ_int from this pipeline must carry the
γ_Ni error budget with it, because it arrives undamped.

### 7.5 Numerical convergence of the reference

| item | value |
|---|---|
| interface SCF at NELM = 200 | **not converged**, \|dE\| = 1.5 × 10⁻⁴ eV |
| interface SCF at NELM = 600 | converged, DAV step 178, \|dE\| = 1.8 × 10⁻⁷ eV |
| energy difference between the two | 0.63 meV |
| effect on γ_int | **0.0016 J/m² (0.53 %)** |
| dipole correction | **required** — without it the asymmetric interface slab diverges to −30863 eV with `BRMIX: very serious problems` |

The reported γ_int uses the converged (NELM = 600) value. The residual
is 0.5 % of γ_int — negligible next to the 0.23 J/m² model term.

### 7.6 What this changes about §6

- §6 attributed the low γ_Ni partly to the +2.58 % coherent strain.
  That attribution is now **weakened**: the strain self-energy is
  +0.0108 eV/atom in VASP and +0.0106 in MACE, i.e. both codes handle
  the strain the same way. Strain is a property of the *state* being
  computed, not a disagreement between the two potentials.
- §6's literature band remains the right sanity check, but our VASP
  value is now inside it and MACE's is outside it. The correct wording
  is "MACE is 3–14 % below the DFT band", not "our value is below the
  band because it is strained".
- **A caveat that cuts the other way, and should not be hidden:** on
  **Cu** the two codes agree to 0.3 %, yet *both* sit +4 % to +17 %
  **above** the literature band in §6 (VASP 1.5023 vs 1.29–1.44). Where
  MACE and plane-wave PBE agree with each other and disagree with the
  published range, the suspicious quantity is more likely the
  literature band than either code — those values come from thinner
  slabs and older conventions. The §6 band should be treated as a
  ±0.1–0.2 J/m² context, not a target. We have not chased this further;
  a thickness-converged VASP γ_Cu (4 → 14 layers) would settle it.

## 8. Why γ_int is the weakest number here

`gamma_int = gamma_Cu + gamma_Ni − W_ad/area`, and the terms nearly
cancel:

| min-thickness | γ_Cu + γ_Ni | W_ad/area | γ_int | γ_int / (γ_Cu+γ_Ni) |
|---|---|---|---|---|
| 7.2 | 3.4195 | 3.3633 | 0.0562 | 1.64 % |
| 10.8 | 3.4158 | 3.3585 | 0.0572 | 1.68 % |
| 14.4 | 3.4159 | 3.3542 | 0.0617 | 1.81 % |
| 18.0 | 3.4163 | 3.3543 | 0.0620 | 1.81 % |
| 21.6 | 3.4171 | 3.3590 | 0.0581 | 1.70 % |

γ_int is **~1.7 %** of the quantities it is subtracted from. A 1 % error
in either the surface energies or W_ad therefore produces ≈ 0.034 J/m²
of error in γ_int — more than half of γ_int itself. The 0.0058 J/m²
scatter across the scan understates the real uncertainty by roughly an
order of magnitude; treat γ_int = **0.06 ± 0.03 J/m²** (model error),
not ± 0.003 (scan scatter).

Anyone quoting γ_int from this pipeline should carry the surface-energy
error budget with it.

## 9. Committee / OOD

With a second real model mounted (MACE-MP-0 **small** as committee
member, medium as primary) on the 18-atom interface:

| signal | value |
|---|---|
| committee disagreement, built structure | 0.0306 eV/Å |
| committee disagreement, relaxed structure | **0.0327 eV/Å** |
| max \|F\| on the relaxed structure | 0.0099 eV/Å |
| OOD trigger | no (threshold 0.8) |

The disagreement between two MACE-MP-0 variants is **3× larger than the
residual force** the relaxation converged to. Relaxing to 0.01 eV/Å is
therefore numerically clean but physically below the model-to-model
noise floor; the force tolerance is not the accuracy-limiting step, the
model is.

The run also exercises the watcher chain that `880ce76` fixed. With
three watchers where the two leading ones return an empty reason and
`CommitteeWatcher` wins the argmax at index 2:

```
NeighborWatcher    score=0.000000  reason=''
ForceNormWatcher   score=0.000000  reason=''
CommitteeWatcher   score=0.032737  reason='committee_disagreement'
argmax -> CommitteeWatcher (index 2), reason='committee_disagreement'
```

The reason belongs to the winning watcher; no `relax_exception`. Before
the fix this shape (empty-reason prefix ahead of the argmax) raised
`IndexError` inside the guard and aborted the relaxation.

**Reading OOD fields:** the single-model rows in the tables above have
`committee_enabled=false` and `ood_monitoring=heuristics_only`. Their
`ood_score=0.000000` says that the geometry and force-magnitude watchers did
not fire; it is not evidence that the structures are in-distribution. Only a
run with `committee_enabled=true` queried model disagreement. The explicit
fields prevent the old ambiguous zero from being mistaken for a committee
certificate.

## 10. Reproducibility

Two independent processes, same command, `--min-thickness 14.4`:

| artifact | result |
|---|---|
| `candidates.csv` | **byte-identical** |
| `interface_relaxed.extxyz` | not byte-identical; max Δposition 5.3 × 10⁻⁶ Å |
| `slab_sub_relaxed.extxyz` | not byte-identical; max Δposition 1.2 × 10⁻⁷ Å |
| `slab_film_relaxed.extxyz` | not byte-identical; max Δposition 5.7 × 10⁻⁷ Å |
| recorded energies | identical to the 6 decimals written |

The extxyz column *order* is byte-stable (`canonicalize_arrays`); what
drifts is the last written digit of the coordinates, from torch's CPU
thread-reduction order. Reruns inside one process are bit-identical. Do
not build equality assertions on extxyz bytes.

`atominterface build` with the dummy backend is **not** reproducible:
`cli_build.py` calls `get_calculator(backend, device=…)` without a seed,
and the dummy calculator draws from an unseeded RNG. The demo driver
always passes `seed=0`. This has no effect on the test suite, which
never asserts dummy forces.

## 11. Limitations

- **One VASP candidate, at one geometry.** §7 verifies the 1×1 coherent
  cell only, and it is still one entry, not the 5–10 the roadmap asks
  for. The cross-check is also single-point: no VASP relaxation was
  performed, so the comparison tests the *energy estimator* on MACE's
  geometries, not the relaxation path itself. A VASP relaxation would
  test both, and would also expose any error in the MACE geometries
  that a single point cannot see.
- **γ_Ni is MACE-limited, not pipeline-limited.** §7.3 shows 95 % of the
  14 % gap is model error. This is now a characterisation of MACE-MP-0,
  not an open defect in AtomInterface — but it does mean γ_Ni, and
  therefore γ_int, cannot be quoted to better than ~14 % / ~0.23 J/m²
  with this backend regardless of how carefully the pipeline is run.
- **One candidate, not 5–10.** The roadmap asks for a candidate table.
  The demo caps `max_area` at 100 Å² and `max_candidates` at 1, so it
  emits the single 1×1 cell. Widening that is a knob, not a code
  change, but it has not been run.
- **MACE runs in float32.** `mace_mp()` defaults to float32 ("faster but
  less accurate … use float64 for geometry optimization"). We did not
  override it, so every number above carries float32 round-off. This is
  a candidate follow-up, not a known defect in the measurements.
- **γ_Cu and γ_Ni are orientation- and strain-specific.** The Ni value
  is for Ni stretched onto Cu, and neither number is the
  experimentally-quoted (orientation-averaged) surface energy.
- **No spin polarisation, no temperature.** Ni is ferromagnetic; MACE-MP-0
  is trained on spin-unpolarised PBE (except Fe, Ni, Co and Mn in some
  variants) and all energies here are 0 K static-lattice values.

## 12. Reproducing this note

```bash
python examples/03_interface_energy.py --backend mace --device cpu \
    --min-thickness 14.4 --outdir outputs/energy_cu_ni
```

A fail-closed wiring check is reproducible without model weights:

```bash
python examples/03_interface_energy.py --backend dummy --max-steps 1 \
    --fmax 0.000001 --outdir outputs/energy_invalid
```

That command must exit non-zero while retaining `candidates.csv` with
`valid=False`, explicit `invalid_reasons`, and empty
`gamma_surf_sub_eV_A2`, `gamma_surf_film_eV_A2`, `gamma_eV_per_A2`, and
`gamma_J_per_m2` cells.

14.4 Å is also the *default*: the demo used to default to 6.0 Å, which
cuts only 4 Cu / 4 Ni layers and reports γ_Ni = 1.979 J/m² and γ_int =
0.041 J/m² — 3 % and 30 % away from the plateau above. Slabs thinner
than ~6 Ni layers are not on the plateau, so the flag must not be
lowered without re-running this table.

`--min-thickness` 7.2 / 10.8 / 14.4 / 18.0 / 21.6 for the thickness
scan, `--vacuum` 10 / 15 / 20 / 25 at `--min-thickness 14.4` for the
vacuum scan. Regression tests live in
`tests/test_validation_cu_ni.py`; the cheap cases run on synthetic
calculators and the MACE-marked ones re-run the driver end to end.
