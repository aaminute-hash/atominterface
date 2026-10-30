# Roadmap

## Phase 0.1 — Interface construction + guard (done — v0.1.0, 2026-10-17)

Scope lock: **five things, no more.**

1. `CoherentInterfaceBuilder` wrapping pymatgen's `CoherentInterfaceBuilder` + `ZSLGenerator`
2. Strain filter (von Mises) + termination bookkeeping
3. `GuardedRelaxer` with `NeighborWatcher` and `ForceNormWatcher`
4. Quarantine: cut OOD cluster → write VASP single-point input
5. `metadata.yaml` provenance for every run

Explicitly **out of scope** for 0.1: fine-tuning, NEB, CZM codegen,
LLM agent.

## Phase 0.2 — Polar terminations and charge balance (done — ships in v0.2.0)

- Slab dipole correction / symmetric slab construction
- Termination energy ranking with a vacuum reference
- Better error messages when pymatgen's interface builder disagrees
  with the user's chemical intuition

## Phase 0.3 — A real interface, end to end (current)

Pick **one** system and publish both a technical note and a full
candidate table:

- Li / LLZO (solid electrolyte), or
- Cu / Ni (low-mismatch fcc/fcc)

Cu / Ni is the chosen system, at the 1x1 coherent (001) cell. The
technical note is `docs/validation_cu_ni_001.md`: thickness and vacuum
convergence scans, a comparison against published DFT surface energies,
and an uncertainty budget for gamma_int (which turns out to be the
weakest number in the ledger, at ~1.7 % of the quantities it is
subtracted from). A co-computed plane-wave reference was added later,
as §7 of the same note.

Both deliverables below are now closed.
**The VASP single point is computed** — 23 single points on the
MACE-relaxed geometries (VASP 6.4.2, PAW_PBE, energies taken as
`energy(sigma->0)` throughout), giving γ_Cu 1.5023 vs 1.4980 J/m²
(+0.3 %), γ_Ni 2.1933 vs 1.9189 (+14.3 %) and W_ad 3.4018 vs 3.3579
(+1.3 %). The γ_int difference closes exactly through γ_int = γ_Cu +
γ_Ni − W_ad, so it is inherited from the Ni surface term rather than
from interfacial bonding, which agrees to 1.3 %; ~95 % of the γ_Ni gap
is MACE-MP-0 model error, not smearing (which shifts γ_Ni by
0.002 J/m²) or missing spin (which moves VASP *towards* MACE). Record
and controls: [validation note §7](docs/validation_cu_ni_001.md#7-vasp-cross-check-co-computed-reference).

**The candidate table is closed too** — 10 low-strain candidates with a
per-candidate formation energy and OOD trace, plus thickness/vacuum
scans and a float64 control, archived in
[`docs/validation/cu_ni_001_multi_candidate/`](docs/validation/cu_ni_001_multi_candidate/).

**(111) and (110) are archived as well**, in
[`docs/validation/cu_ni_multi_orientation/`](docs/validation/cu_ni_multi_orientation/):
γ_int = 0.081 J/m² and 0.088 J/m², both from explicit registry scans and both
shown to be independent of the entry point (initial gap 2.0 → 2.4 Å moves
γ_int by ≤ 1e-4 J/m²). Two things came out of that work:

- **A geometry bug, now fixed** (PR #25): the builder read `matrix[:2]` as
  the in-plane lattice, which is only true on (001). Area, layer count and
  in-plane period were all wrong on (111)/(110) — Cu(111) reported
  3.2671 Å² against a true 5.6587 Å². (001) comes out bit-identical
  across the fix.
- **A design gap, still open: the initial lateral registry is a function
  of `--min-thickness`**, and every registry is a symmetry-locked
  stationary point that a relaxation never leaves. On Cu/Ni(111) the
  thickness steps 10.8 / 14.4 / 18.0 / 21.6 Å start at fcc-hollow /
  on-top / fcc-hollow / hcp-hollow and give γ_int 0.081 / 0.898 / 0.081 /
  0.081 — an 11× jump from the thickness knob alone, which a reader of a
  thickness scan would otherwise mistake for non-convergence. PR #25
  downgrades this from *silently wrong* to *warned*: a non-(001) run now
  warns that its registry has not been scanned, and prints the measured
  lateral offset so two runs can be compared. **The registry problem
  itself is not fixed** — the builder still does not sample registries.

Both orientation archives carry the same caveat: (111) and (110) have
**no DFT cross-check** — every number there has exactly one source
(MACE-MP-0). Only the (001) values rest on the plane-wave reference above.

The energy-credible core — real uMLIP backends with model provenance,
`interface_formation_energy` + reference slabs, committee OOD trace,
and the CLI / example 03 end-to-end driver — lands with v0.2.0.

Deliverables:

- 5–10 low-strain candidates with formation energy + OOD trace
- At least one cross-checked with a VASP single-point (done — the 1x1
  cell, validation note §7)
- A short paper: *"Where uMLIPs lie about interfaces"*

## Phase 0.4 — AutoConstitutive data pipeline

- Extract Green-Lagrange strain and Cauchy stress from guarded MD
  trajectories (Virial → Cauchy, with Pint unit enforcement)
- Invariance tests: rigid rotation leaves stress invariant, small-strain
  elastic tensor vs. independent relaxation
- Export a NumPy constitutive table consumed by a FEniCSx unit-cell
  example
- CZM codegen (UMAT / VUMAT / MOOSE) comes **after** the data pipeline
  is trusted

## Phase 0.5 — Community

- Publish a software paper (JOSS or CPC)
- Submit PRs / start Discussions in ase, pymatgen, matgl, torch-sim
- Establish a citation format and a "paper using AtomInterface" list
