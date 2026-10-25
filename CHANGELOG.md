# Changelog

## [Unreleased]

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
