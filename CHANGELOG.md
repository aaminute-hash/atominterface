# Changelog

## [Unreleased]

### Fixed

- **The film no longer lands exactly on top of the substrate on (111).**
  Assembly had no in-plane placement step, so the lateral registry was
  whatever the slab cut left behind: `n_Cu mod 3` decided whether the
  film's joining layer coincided with the substrate's top layer, and when
  it did the interface began on a symmetry-locked stationary point that
  relaxation could not leave — γ_int came out about 11× too large on
  Cu/Ni(111). It was periodic, not a single value: it occurred at
  `--min-thickness` 8.0, 13.0/14.0/14.4 and 19.0/20.0 Å. The builder now
  shifts the film in-plane by the substrate's own stacking vector when,
  and only when, the as-built join is exactly opposed. (001) and (110)
  build bit-identically, and the cell never changes.
- **The non-(001) registry warning no longer quotes 0.081 / 0.898 / 11×.**
  That sequence was a symptom of the defect above and is no longer
  reproducible. The warning now states what is still true: the registry is
  inherited from the two slab cuts, is never scanned across candidate
  sites, and there is no configuration field to request one.
- The (111)/(110) validation archive is marked as partially superseded: its
  on-top value 0.8980 J/m² is a pre-fix record that current code does not
  reproduce, and it has not been re-measured. The 0.5953 J/m² (110) ridge
  value still reproduces — (110) was deliberately left untouched.

## [0.4.0] - 2026-10-30

Closes the non-(001) gap: orientations other than (001) now build correct
geometry, and the one remaining way they can still be wrong is at least
reported instead of silently producing a credible number.

A note on numbering: **the version no longer tracks the phase number.**
Phase 0.4 in `docs/roadmap.md` (the MD -> constitutive pipeline) is still
planned; v0.4.0 ships the Cu/Ni (111) and (110) work below.

### Added

- **Miller indices for both sides.** `--sub-miller` / `--film-miller` on
  example 03 and `substrate_miller` / `film_miller` on
  `CoherentInterfaceConfig`, so an orientation other than (001) can be
  requested at all. The run banner now names the orientation instead of
  implying (001).
- **A geometry gate on the assembled interface.** Assembly refuses an
  interface whose area is not conserved (`GeometryGateError`), and warns
  when the shortest in-plane period falls below the nearest-neighbour
  distance or when a cross-interface pair is closer than the contact scale.
  The proximity warning names on-top registry when the lateral offset is
  exactly zero, and reports the measured lateral offset in A so two runs
  can be compared.
- **A warning that a non-(001) registry has not been scanned.** Any run
  whose Miller index is not (001) now warns that the registry it was handed
  may sit on a symmetry-locked stationary point, quoting the measured
  Cu/Ni evidence (6.8x on (110), 5.9x on (111)). (001) is exempt: the
  published protocol enumerates ten candidates itself.
- **The Cu/Ni (111) and (110) archive.**
  `docs/validation/cu_ni_multi_orientation/` holds the registry scans, the
  registry-vs-thickness map, the entry-point control and the fmax
  comparison. gamma_int = 0.081 J/m^2 on (111) and 0.088 J/m^2 on (110),
  both from explicit registry scans. Both orientation archives carry the
  caveat that they have **no DFT cross-check** — every number there has
  exactly one source (MACE-MP-0).

### Fixed

- **Non-(001) slab geometry was wrong.** The builder read `matrix[:2]` as
  the in-plane lattice, which is only true on (001): on (111) and (110) the
  pymatgen slab does not have both a and b in the interface plane. Area,
  layer count and in-plane period were all wrong as a result — Cu(111)
  reported 3.2671 A^2 against a true 5.6587 A^2. Slabs are now oriented so
  the surface normal is exactly +z (symmetric slabs included) and the four
  `abs(det(v[:, :2]))` area computations use `|a x b|`. **(001) is
  bit-identical across this change**: the area is locked at its full
  precision 6.5341125 A^2 and a re-run reproduces every published (001)
  energy exactly.
- **The coherent bulk reference was strained on the wrong plane.** It
  applied `base * [1+e, 1+e, 1+sz]`, which is equibiaxial on (001) only:
  on (111) that gives 3.72e-2 of in-plane anisotropy and flips the sign of
  the normal strain. It now uses the deformation gradient
  `F = I + e(I - nn^T) + sz*nn^T` built from the (hkl) normal.
- **A published document contradicted itself on the sign of the fmax
  effect.** `docs/validation_cu_ni_001.md` used `+` in three places and
  `-` in section 7 for the same quantity. Unified on
  `delta = gamma(fmax=0.01) - gamma(fmax=0.002)` ("the default batch sits
  above the converged value"), and section 7 now gives drops with both
  endpoints so the direction cannot be misread.

### Known limitations

- **The registry problem itself is not fixed.** The initial lateral
  registry is a function of `--min-thickness`, and every registry is a
  stationary point a relaxation never leaves: on Cu/Ni(111) the thickness
  steps 10.8 / 14.4 / 18.0 / 21.6 A give gamma_int 0.081 / 0.898 / 0.081 /
  0.081 — an 11x jump from the thickness knob alone, which a reader of a
  thickness scan would otherwise mistake for non-convergence. The builder
  still does not sample registries; v0.4.0 only downgrades this from
  silently wrong to warned.
- `gamma_int(111) = 0.898 J/m^2` and `gamma_int(110) = 0.5953 J/m^2` are
  **not** values. They are symmetry-locked artefacts, documented as
  failure-mode examples only and marked not-for-citation.

## [0.3.0] - 2026-10-28

Closes Phase 0.3.2: the Cu/Ni(001) benchmark is reproducible end to end, and
every number it reports carries a measured uncertainty instead of a hope.

### Added

- **A working OOD committee.** `build --relax --committee-backend <backend>`
  mounts a second uMLIP beside the primary one through
  `GuardedRelaxer.from_calculators`, so `ood_score` measures disagreement
  between two independently trained models rather than being a constant zero.
  The flag is empty by default: without it nothing changes.
- **A threshold the committee can actually fire at.** The committee reports
  eV/Å while `ood_threshold` was calibrated for `ForceNormWatcher`'s
  dimensionless score, so both the measured distribution-in baselines
  (0.030-0.068 eV/Å) and real out-of-distribution cases (0.4-0.66 eV/Å) sat
  below it. The committee now has its own threshold,
  `DEFAULT_COMMITTEE_THRESHOLD = 0.15` eV/Å, with the calibration basis
  recorded beside the constant, and the winning watcher is named in the
  metadata so a trigger can be attributed.
- **Per-side slab thicknesses.** `min_thickness_substrate` /
  `min_thickness_film` on `CoherentInterfaceConfig`, `--min-thickness-substrate`
  / `--min-thickness-film` on the builder and on example 03, each falling back
  to the shared `min_thickness` when unset. The resolved values are recorded
  per row as `min_thickness_substrate_A` / `min_thickness_film_A`, so the layer
  counts in the same row stop contradicting the thickness column.
- **A dtype choice for the MACE backend.** `get_calculator("mace",
  dtype="float64")` and example 03's `--dtype`; the default stays `float32`.
- **The Cu/Ni(001) benchmark archive.**
  `docs/validation/cu_ni_001_multi_candidate/` holds the 10-candidate run, the
  `fmax = 0.002` convergence batch, the float64 control and the
  thickness/vacuum scan as per-candidate CSVs plus structures, with
  `environment.lock.yml` and `reproduce.md` (a five-step checklist for
  confirming that a reproduction matched).

### Fixed

- **MACE float64 was silently a no-op.** `mace_mp()`'s parameter is spelled
  `default_dtype`; passing `dtype=` fell into its `**kwargs`, so a float64
  request built a float32 model with no error and no warning while the run
  metadata still recorded `dtype: float64`. The test that should have caught
  it mocked `mace_mp` as `def mace_mp(**kwargs)` -- accepting any keyword name,
  it pinned the wrong contract and stayed green over a dead feature. The mock
  now spells the real signature, so a wrong name raises instead of passing
  silently. **v0.2.1 carries the broken entry point**: do not use
  `--dtype float64` there (documented in `docs/validation_cu_ni_001.md` and in
  the archive README).
- `mypy src examples tests`: 22 errors to 0. `examples/03` had 16 unique error
  sites; all were fixed by annotation and narrowing, with no `type: ignore`.

### Science

Cu/Ni(001), MACE-MP-0 medium, float32, coherent bulk reference:

- 10/10 candidates valid; gamma_Cu 1.4977, gamma_Ni 1.9183 J/m2.
- gamma_int = **0.0567 +/- 0.0002 J/m2** at `fmax = 0.002`, against 0.0617 for
  the same candidates at `fmax = 0.01` -- a +0.005 J/m2 bias from residual
  relaxation convergence, not from physics, model error or a code defect. Each
  candidate's displacement scales with its residual force, and two independent
  observables agree (gamma_int spread narrows 25x, per-cell energy spread 22x).
- float64 control: 0.0616 versus 0.0617. **Precision is not the bottleneck; the
  force gate is.**
- 14.4 A is a converged thickness (10.8 to 14.4 steps +0.0044 J/m2, 14.4 to
  18.0 only +0.0005); vacuum from 10 to 20 A changes nothing.
- Per-side thicknesses decompose as expected: gamma_Cu tracks the substrate
  layer count, gamma_Ni the film's, and gamma_int's thickness dependence is
  almost entirely the Ni side.

### Known limitations

- The 0.15 eV/A committee threshold is calibrated on three structures; it needs
  re-calibration on more material systems before being relied on.
- The 10 candidates share one in-plane strain, so the batch is a
  supercell-consistency check rather than a strain scan.
- Without `--committee-backend`, `ood_score = 0` means "not measured", not
  "in distribution".

## [0.2.1] - 2026-10-26

### Added
- **Energy results fail closed.** `candidates.csv` now carries `valid`,
  `committee_enabled` and `ood_monitoring`. `valid=true` requires finite
  energies for both bulk references, and finite energies and forces plus
  convergence for the two reference slabs and the interface. When any node
  fails, the row and its structures are still written for diagnosis, but
  every gamma cell is left empty and the driver exits non-zero -- a missing
  gamma is now distinct from a physically computed zero.
- `atominterface.relax.proximity`: minimum-image distance helpers
  (`minimum_image_distances`, `nearest_image_vectors`,
  `pairwise_distance_matrix`, `pairwise_min_distance`) built on
  `ase.geometry.find_mic`, so skewed cells and mixed `pbc` masks are
  handled.
- `NeighborWatcher` results carry `per_atom` and `meta["threshold"]` /
  `meta["min_pair"]`.
- `CommitteeWatcher` results carry `meta["committee_state"]`
  (`disabled` / `unavailable` / `degraded` / `ok`) together with
  `committee_requested` / `committee_members` / `committee_failed`, and
  `_record_committee_state()` resolves the worst state observed into the
  run metadata.
- `cut_cluster()` accepts `record=` and fills
  `QuarantineRecord.cluster_indices`; the kept indices are also written to
  `cluster.info["cluster_indices"]`.
- Calculator construction failures are recorded as row failures instead of
  aborting the whole energy run.

### Fixed
- **`NeighborWatcher` ignored periodic boundaries.** Distances were bare
  cartesian, so a contact straddling the cell edge of a slab cell was
  invisible. Distances are now minimum-image.
- **`NeighborWatcher`'s threshold drifted with the vacuum.** It was
  `mean(|a_i|) / 10 * 0.6`, so thickening a slab's vacuum from 10 A to 40 A
  raised it from 0.6 A to 1.2 A and could flip a clean structure to
  out-of-distribution with no change to the geometry. The threshold is now
  `min_dist_frac * (r_i + r_j)` over the covalent radii of the bonded pair.
- **`cut_cluster()` ignored its own `include_images` flag.** The neighbour
  across the periodic boundary was dropped from the cluster cut intended to
  investigate it. Images are now honoured, and each source atom is emitted at
  the image nearest the centre so the result is one contiguous ball.
- **`GuardedRelaxer` evaluated the watcher chain only between optimization
  calls.** It now walks `irun()`, so a transient excursion inside one
  `opt.run()` can no longer hide behind benign final forces.
- **`_aggregate()` dropped `per_atom` and `meta`.** The quarantine cluster was
  therefore always anchored on atom 0 and the committee state was invisible:
  an outage scored 0.0, could never win the argmax, and never surfaced. The
  winning watcher's signals now propagate, and a winner without per-atom
  signal raises `quarantine_center_degraded` instead of silently centring on
  atom 0.
- **A committee that could not run was indistinguishable from one that
  agreed.** Both scored `0.0`; an unusable committee now reports
  `committee_state="unavailable"` (or `"degraded"` when some members were
  dropped) with `reason="committee_unavailable"`, and overrides
  `ood_monitoring` so a monitoring outage is never readable as consensus.
- `QuarantineRecord.cluster_indices` was never assigned.
- `cut_cluster()` accepted an out-of-range `center_index` silently; it now
  raises `IndexError`.
- **Model-loading failures were swallowed.** `mace_.py` silently fell back
  through every tier, and `GuardedRelaxer._wrap()` fabricated a zero-force
  calculator for anything it could not construct, so a backend that never
  loaded produced plausible-looking zeros. Both now raise with the full
  chain of causes.
- **`ood_score=0` was readable as "in distribution".** The CSV and the run
  metadata now carry `committee_enabled` and `ood_monitoring` explicitly, so
  a zero that merely means "no committee was asked" can no longer be
  mistaken for a certificate.

### Changed
- `docs/validation_cu_ni_001.md`: the Cu(001) | Ni(001) note documents the
  validity contract behind gamma, explains how to read the new `valid`,
  `committee_enabled` and `ood_monitoring` fields instead of the old
  ambiguous `ood_score=0.000000`, and records a fail-closed wiring check that
  reproduces without model weights -- `python
  examples/03_interface_energy.py --backend dummy --max-steps 1 --fmax
  0.000001` must exit non-zero while keeping `candidates.csv` with
  `valid=False` and empty gamma cells.

## [0.2.0] - 2026-10-25

### Added
- `atominterface build` now works end to end: loads a substrate/film pair,
  enumerates coherent matches, writes one structure file per candidate plus
  `candidates.csv`, and supports `python -m atominterface`.
- `CoherentInterfaceBuilder.preview_matches()` reports the best lattice
  matches ignoring the strain cut-off, so an empty result comes with the
  `--max-strain` value that would keep them instead of a dead end.
- `in_plane_mismatch()` / `max_stretch_strain()` / `max_shear_strain()`:
  2D engineering strain for surface cells.
- Example input structures `examples/data/{Cu,Ni}_fcc.vasp`.
- `atominterface build` gains `--backend` / `--device` / `--relax`: the
  real uMLIP backends (MACE-MP-0 medium, CHGNet) load pretrained weights
  by default, and the resolved model identity is recorded in
  `provenance.RunMeta.model_hash`.
- `examples/03_interface_energy.py`: end-to-end Cu(001) | Ni(001)
  interface formation energy — bulk single points, reference slabs,
  guarded relaxation, termination energies, gamma_int.
- `candidates.csv` now reports both gamma unit conventions (eV/A^2 and
  J/m^2) and the OOD trace (`ood_score`, `ood_step`, `ood_triggered`).
- `CommitteeWatcher` is now wired into the default watcher chain of
  `GuardedRelaxer` whenever `committee_calculators` is passed -- it was
  listed under 0.1.0 but was not enabled by default until now.
- `atominterface.io.canonicalize_arrays()`: sorts `atoms.arrays` keys so
  extxyz column order (and file bytes) no longer drift with
  `PYTHONHASHSEED`.
- `docs/validation_cu_ni_001.md`: thickness/vacuum convergence scans,
  literature comparison and an uncertainty budget for gamma_int.

### Changed
- **Strain metric is now two-dimensional.** The old code folded the vacuum
  axis into a 3D deviatoric decomposition, which reported a pure 5 %
  isotropic in-plane stretch as 3.4 % tensile *plus* 3.4 % shear that does
  not exist in the crystal, so well-matched pairs were rejected. `tensile`
  is now half the trace of the in-plane strain tensor and `shear` is the
  in-plane principal spread. `compute_mismatch()` keeps the 3D behaviour
  for genuinely 3D cells.
- Lattice matching runs on the **slab** in-plane cells instead of the bulk
  cells. For any surface other than (001) the bulk's first two vectors are
  not in the interface plane, and the reported `match_area` disagreed with
  the assembled geometry.
- `max_area` default 500 -> 100 A^2. The ZSL enumeration is super-linear in
  area: on a typical 2D cell 100 A^2 takes < 1 s while 500 A^2 takes
  minutes. `max_area_ratio` now drives ZSL's area tolerance, and
  `zsl_max_area_ratio` is a real bound on the matched supercell area ratio
  instead of an implicit `-1.0` fudge.
- `Candidate.atoms` is now the ASE `Atoms` view, as its docstring always
  claimed; it was leaking a pymatgen `Structure` and every exporter raised.
- **Python floor is now 3.12** (`requires-python >= 3.12`, CI matrix
  3.12/3.13, ruff `target-version = py312`, mypy `python_version = 3.12`).
  The old 3.9-3.12 matrix could not stay green: pymatgen 2026.10 requires
  >= 3.11 and numpy 2.5 requires >= 3.12, while the mypy 1.x resolved on
  3.9/3.10 cannot parse the PEP 695 `type` statements in the numpy stubs.
- `examples/03_interface_energy.py` defaults to `--min-thickness 14.4`
  (8 Cu / 10 Ni layers), the plateau point of the thickness scan in
  `docs/validation_cu_ni_001.md`. The previous 6.0 A default cut only 4 Ni
  layers and reported gamma_Ni = 1.979 J/m^2 and gamma_int = 0.041 J/m^2
  against converged values of 1.9189 and 0.0590 -- 3 % and 30 % off, so the
  demo contradicted its own validation note out of the box.

### Fixed
- `make_supercell` was mutating the source slab (newer pymatgen defaults to
  `in_place=True`), compounding candidates from 6.5 to 1104 to 16564 A^2.
- Match ranking was decided by float noise in the strain, so a pair with
  identical cell shapes produced million-atom "best" candidates; equally
  good matches are now ranked by area.
- `interface_id` was identical for every supercell of the same surface pair.
- `strain_film` reported the shear strain instead of the stretch.
- `vacuum_thickness` was ignored during assembly (the code looked for a
  non-existent `vacuum` field).
- `_fallback_matches` scaled the film but reported the substrate as scaled,
  and computed the area from the wrong cell.
- `<supercell>` factors returned a bogus `(m, n)` pair instead of the
  integer transformation matrix.
- The quickstart example took over 5 minutes because it asked pymatgen's
  ZSL for 600 A^2 interfaces; it is now 8 seconds and exposes `--max-area`.
- **Termination energies were referenced to the pristine bulk.** A
  coherent slab carries the interface's in-plane lattice, so the film
  side is strained (Cu/Ni: +2.58 %) while the substrate is not;
  referencing it against the unstrained bulk left `n * dE_strain` in the
  termination energy, a term that grows with layer count. gamma_Ni
  drifted 2.00 -> 2.10 J/m^2 across the thickness scan instead of
  converging. The reference is now the *coherent* bulk (in-plane locked
  to the slab strain, out-of-plane relaxed by a local three-point
  parabola fit — a fit across the whole grid is biased by anharmonicity
  and reported +0.57 meV/atom of spurious Cu relaxation gain where a
  dense scan finds +0.35). gamma_Ni spread over 6 -> 14 Ni layers:
  0.1031 -> 0.0030 J/m^2.
- `examples/03_interface_energy.py --fmax` default 0.05 -> 0.01 eV/A. At
  0.05 the force driving a Cu slab's out-of-plane expansion sits at the
  tolerance itself, so the slab interior never reached the spacing the
  bulk reference assumes (measured central interlayer spacing 1.8075 A
  vs a bulk optimum of 1.8219 A).
- `GuardedRelaxer._aggregate` reported the wrong watcher's reason. It
  indexed a reasons list compacted to non-empty entries, so any
  empty-reason watcher ahead of the argmax shifted the index
  (misattributing the reason), and an all-empty prefix raised
  `IndexError`, which surfaced as `relax_exception`. The reason now comes
  from the watcher that won the argmax.
- The dummy backend rejected ASE 3.29's `force_consistent=True` kwarg.
- `provenance` could emit a non-plain `str` for the torch version, which
  PyYAML refused to represent.
- **`GuardedRelaxer` could relax forever.** `max_steps` only bounded the
  guard loop, while `opt.run()` was called without `steps` and so ran on
  ASE's 100,000,000-step default: any calculator whose residual force
  plateaus above `fmax` never returned. The dummy backend reached it as
  soon as the demo tolerance dropped to 0.01 eV/A (its force floor is
  ~0.018), and a real backend is not safe either -- the Cu/Ni note
  measures a MACE residual of 0.0099 eV/A against that same 0.01 eV/A
  target. The budget now caps the *total* optimizer steps across guard
  iterations, and an exhausted budget is reported as a
  `step_budget_exhausted` reason rather than a silent `converged=False`.

## [0.1.0] - 2026-10-17

### Added
- `CoherentInterfaceBuilder` wrapping pymatgen's `CoherentInterfaceBuilder` + `ZSLGenerator`
- Strain filtering (von Mises) and termination bookkeeping
- `GuardedRelaxer` with `NeighborWatcher`, `ForceNormWatcher`, `CommitteeWatcher`
- Quarantine: OOD-triggered cluster extraction + VASP single-point input writer
- `interface_energy`, `work_of_separation`, `fit_cohesive_zone` thermodynamics helpers
- `export_czm_moose` for FEM-ready CZM parameter snippets
- `metadata.yaml` provenance via `atominterface.provenance`
- CLI: `atominterface build coherent`, `atominterface builders list-builders`
- Example: `01_build_and_relax.py` (Li bcc | NaCl rocksalt, dummy backend)
- GitHub Actions CI (3.9–3.12, ruff, mypy, pytest, build)
- Physical invariance tests: rotation, force consistency, stress symmetry
