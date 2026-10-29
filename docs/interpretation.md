# Interpreting the numbers

This note is about what the outputs mean and how much to trust them. It exists
because the three quantities this tool reports do **not** carry the same
confidence, and quoting them as if they did is the easiest way to be wrong.

All values below are measured on Cu(001)|Ni(001) with MACE-MP-0 medium,
float32, using the coherent bulk reference. The underlying data is in
[`validation/cu_ni_001_multi_candidate/`](validation/cu_ni_001_multi_candidate/)
and the analysis in [`validation_cu_ni_001.md`](validation_cu_ni_001.md).

---

## The three quantities are not equally trustworthy

γ_int is computed as a difference of large terms:

```
gamma_int = gamma_Cu + gamma_Ni - W_ad / area
```

With γ_Cu + γ_Ni ≈ 3.42 J/m² and γ_int ≈ 0.057 J/m², the result is **1.7% of
the terms it is subtracted from**. A relative error of 0.5% in the surface
energies becomes a ~30% relative error in γ_int. That single arithmetic fact
explains most of what follows.

Measured consequence, over ten candidates:

| quantity | spread across candidates |
|---|---|
| γ_Cu | essentially bit-identical |
| γ_Ni | essentially bit-identical |
| γ_int | 0.0049 J/m² at `fmax=0.01` (≈ 9% of its value) |

So: **trust γ_Cu and γ_Ni as they come. Treat γ_int as a number that needs a
stated convergence setting.**

---

## Three error sources, and what each one costs

| source | size | what to do |
|---|---|---|
| relaxation convergence (the `fmax` gate) | **+0.005 J/m²** at `fmax=0.01`; ±0.0002 at `fmax=0.002` | tighten `fmax` — cheap and effective |
| float32 round-off | **0.0001 J/m²** | nothing. Not worth fixing |
| the MLIP model itself | γ_Ni is **+14.3%** against VASP; γ_Cu +0.3%; the interface bonding term only +1.3% | change model or apply a DFT correction |

### Convergence — the one worth spending on

At `fmax = 0.01` the ten candidates split into clusters by supercell size
(0.0617 / 0.0568 / 0.0573 J/m²). Tightening to `fmax = 0.002` collapses that
spread to **0.0002 J/m²**, a 25× reduction, and each candidate moves by an
amount proportional to its own residual force: the least-converged candidate
falls the most. That proportionality is the signature of residual convergence
error, and it rules out physics, model error and code defects.

Two independent observables agree: γ_int spread narrows 25×, per-cell interface
energy spread narrows 22×.

Cost: about 13% more wall time (50.5 s → 56.9 s for one candidate).

### Precision — the one that is not worth spending on

A float64 control run at the same `fmax=0.01` gives 0.0616 versus 0.0617 J/m²:
**0.0001**. Double precision roughly doubles relaxation time for a change two
orders of magnitude smaller than the convergence error.

Rule of thumb: **you almost never need float64.** Use it only when you want a
geometry-optimised reference value for its own sake. Reproducing this benchmark
in float32 with `--fmax 0.002` is both sufficient and cheaper.

(If you are on v0.2.1, `--dtype float64` is silently a no-op there — see
[`troubleshooting.md`](troubleshooting.md).)

### The model — the one you cannot fix from inside

Against VASP on the same structures, γ_Ni comes out 14.3% high. The error
budget closes almost entirely through that surface term: the interface bonding
term itself agrees to 1.3%. In other words, **the interface chemistry is fine;
the Ni surface energy is where the model is wrong**, and γ_int inherits it.

No flag fixes this. Either accept it with a stated uncertainty, or correct the
surface term from DFT.

---

## Geometry settings that do and do not matter

| setting | effect | recommendation |
|---|---|---|
| slab thickness | 10.8 → 14.4 Å steps **+0.0044** J/m²; 14.4 → 18.0 Å only +0.0005 | **14.4 Å is converged.** Thinner is genuinely wrong, not just imprecise |
| vacuum | 10 / 15 / 20 Å produce **bit-identical** results | 15 Å has ample margin; do not spend here |
| per-side thickness | γ_Cu tracks the substrate layer count only, γ_Ni the film's | safe to vary per side; γ_int's thickness dependence is almost entirely the Ni side |

---

## How to quote these numbers

**Do:**

> γ_int(Cu(001)|Ni(001)) = 0.0567 ± 0.0002 J/m², MACE-MP-0 medium, float32,
> coherent bulk reference, `fmax = 0.002` eV/Å, ten candidates.

**Do not:**

> γ_int = 0.062 J/m²

That is the value you get from the shipped defaults at `fmax = 0.01`, and it
carries a known **+0.005 J/m²** bias. Quoting it without the convergence
setting is how a number becomes misleading.

**Also state the model.** The 14.3% Ni surface error is a property of
MACE-MP-0, not of this package. A γ_int quoted without its model is not
comparable to anyone else's.

---

## Reading an OOD score

An `ood_score` from a single-model run is **not a measurement** — there is no
second opinion to disagree with. Mount a second model
(`--committee-backend`) before reading anything into it, and remember the
committee's threshold is in eV/Å (0.15 by default) while the heuristic
watchers' threshold is dimensionless (0.8). They are not interchangeable.

Distribution-in baselines for this system measured 0.030–0.068 eV/Å; genuinely
out-of-distribution inputs measured 0.4–0.66 eV/Å. The threshold is calibrated
on three structures so far — treat it as a starting point, not a law.
