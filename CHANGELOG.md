# Changelog

## [Unreleased]

### Added
- `atominterface.relax.proximity`: minimum-image distance helpers
  (`minimum_image_distances`, `nearest_image_vectors`,
  `pairwise_distance_matrix`, `pairwise_min_distance`) built on
  `ase.geometry.find_mic`, so skewed cells and mixed `pbc` masks are
  handled.
- `NeighborWatcher` results now carry `per_atom` (each atom's relative
  bond shortfall) and `meta["threshold"]` / `meta["min_pair"]`.
- `CommitteeWatcher` results now carry `meta["committee_state"]`
  (`disabled` / `unavailable` / `degraded` / `ok`) plus
  `committee_requested` / `committee_members` / `committee_failed`.
- `cut_cluster()` accepts `record=` and fills
  `QuarantineRecord.cluster_indices`; the kept indices are also on
  `cluster.info["cluster_indices"]`.

### Fixed
- **`NeighborWatcher` ignored periodic boundaries.** Distances were bare
  cartesian, so a contact straddling the cell edge of a slab cell was
  invisible. Distances are now minimum-image.
- **`NeighborWatcher`'s threshold drifted with the vacuum.** It was
  `mean(|a_i|) / 10 * 0.6`, so thickening a slab's vacuum from 10 A to
  40 A raised it from 0.6 A to 1.2 A and could flip a clean structure
  to out-of-distribution with no change to the geometry. The threshold
  is now `min_dist_frac * (r_i + r_j)` over the covalent radii of the
  bonded pair -- a local length scale.
- **`cut_cluster()` ignored its own `include_images` flag.** The
  neighbour across the periodic boundary was dropped from the cluster
  cut to investigate it. Images are now honoured, and each source atom
  is emitted at the image nearest the centre so the result is one
  contiguous ball.
- **`QuarantineRecord.cluster_indices` was never assigned.**
- **A committee that could not run was indistinguishable from one that
  agreed.** Both scored `0.0`; an unusable committee now reports
  `committee_state="unavailable"` (or `"degraded"` when some members
  were dropped) and `reason="committee_unavailable"`.
- **`cut_cluster()` accepted an out-of-range `center_index` silently.**
  It now raises `IndexError` instead of cutting around index 0.

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
