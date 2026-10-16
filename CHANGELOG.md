# Changelog

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
