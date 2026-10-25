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
subtracted from).

Two deliverables below are still open. **No VASP single point** was
computed — this environment has no plane-wave code, so the cross-check
in the note is against published DFT values rather than a co-computed
reference. And the candidate table is **one 1x1 cell, not 5–10**: the
demo caps `max_area` at 100 A^2 and `max_candidates` at 1. Widening
that is a knob, not a code change, but it has not been run.

The energy-credible core — real uMLIP backends with model provenance,
`interface_formation_energy` + reference slabs, committee OOD trace,
and the CLI / example 03 end-to-end driver — lands with v0.2.0.

Deliverables:

- 5–10 low-strain candidates with formation energy + OOD trace
- At least one cross-checked with a VASP single-point
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
