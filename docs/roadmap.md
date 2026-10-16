# Roadmap

## Phase 0.1 — Interface construction + guard (current)

Scope lock: **five things, no more.**

1. `CoherentInterfaceBuilder` wrapping pymatgen's `CoherentInterfaceBuilder` + `ZSLGenerator`
2. Strain filter (von Mises) + termination bookkeeping
3. `GuardedRelaxer` with `NeighborWatcher` and `ForceNormWatcher`
4. Quarantine: cut OOD cluster → write VASP single-point input
5. `metadata.yaml` provenance for every run

Explicitly **out of scope** for 0.1: fine-tuning, NEB, CZM codegen,
LLM agent.

## Phase 0.2 — Polar terminations and charge balance

- Slab dipole correction / symmetric slab construction
- Termination energy ranking with a vacuum reference
- Better error messages when pymatgen's interface builder disagrees
  with the user's chemical intuition

## Phase 0.3 — A real interface, end to end

Pick **one** system and publish both a technical note and a full
candidate table:

- Li / LLZO (solid electrolyte), or
- Cu / Ni (low-mismatch fcc/fcc)

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
