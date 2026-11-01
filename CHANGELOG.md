# Changelog

## [Unreleased]

### Added

- `registry="scan"` builds one candidate per distinguishable high-symmetry
  site of the surface (per match/termination), so a non-(001) gamma_int can
  finally be quoted from an explicit scan rather than a single requested site.
- `--max-registry-sites` caps how many sites a scan enumerates (`None` scans
  all); it is the CLI spelling of `CoherentInterfaceConfig.max_registry_sites`.
- `enumerate_registry_sites(sub_coords, sub_2d)` enumerates the
  distinguishable in-plane sites of a surface in deterministic order.
- **`physics/stress_strain.py` — the Phase 0.4 (AutoConstitutive) data
  pipeline.** `extract(trajectory, *, reference_cell=None)` turns any ASE
  trajectory-like input (a file path, an `ase.io.Trajectory`, or a sequence
  of `Atoms`) into a `StressStrainTable` of per-frame Green-Lagrange strain,
  Cauchy stress (eV/Å³, plus a GPa copy) and temperature. It is a pure,
  side-effect-free pipeline that never imports or touches
  `atominterface.relax`.
- `green_lagrange_strain`, `strain_to_voigt`, `virial_to_cauchy` and
  `cauchy_to_voigt` — the frame-level physics helpers behind `extract`, in
  the Voigt order `[xx, yy, zz, yz, xz, xy]` with **tensor** shear
  components (no 2× engineering factor), matching `transform_stress_voigt`
  and ASE's `get_stress(voigt=True)`.
- `export_constitutive_table(table, path)` writes a `StressStrainTable` to a
  NumPy `.npz` constitutive table — one array per channel plus a `units`
  JSON string key — for downstream fitting.
- `examples/02_md_stress_strain.py` — the Phase 0.4 pipeline demo: a
  synthetic stretched Cu supercell becomes an extracted table and a `.npz`
  constitutive table, pure ase/numpy (no network, no MACE).
- **`physics/md_sources.py` — real-MD bridges.** `read_lammps_md(dump, log)`
  and `read_cp2k_md(stress, pos, cell, ener=...)` turn a LAMMPS classical run
  and a CP2K AIMD run into `list[ase.Atoms]` frames carrying the full Cauchy
  stress (tension positive, eV/A³) as a `SinglePointCalculator`. The three
  source conventions are pinned and unit-tested: pressure is negated to
  stress (`sigma = -P`), `bar -> eV/A^3` via `1e-4 / EV_A3_TO_GPA`, and the
  kinetic term is kept by reading the software's *full* pressure (LAMMPS
  `pxx..pyz`, CP2K `.stress`) rather than rebuilding a virial from
  coordinates. LAMMPS velocities are converted A/ps -> ASE via
  `sqrt(mvv2e)`. See `docs/validation_md_virial_cauchy.md` for the
  frame-by-frame real-data cross-check.

## [0.5.2] - 2026-10-31

### Fixed

- **`rotation_matrix(axis, angle)` returned a matrix of the wrong handedness.**
  The quaternion's vector part was negated, producing `R(-theta)` -- the
  mirror-transpose of what ASE builds. `ase.Atoms.rotate(90, "z")` maps `+x`
  onto `+y`; `rotation_matrix("z", 90)` mapped `+x` onto `-y`.

  This never produced a wrong number inside the package, because
  `rotate_positions` consumes the matrix as `positions @ R.T` and the two sign
  flips cancelled. The defect lived entirely at the boundary: passing an ASE
  rotation (for example the ZSL `match.rotation`) into `rotate_positions`,
  `transform_forces` or `transform_stress_voigt` would have mirrored the cell
  silently.

  Migration: if you used the returned matrix as a column-vector operator
  (`R @ v`), it now means what a textbook rotation means. If you compensated
  for the old behaviour by transposing it yourself, drop that compensation.
  `rotate_positions(p, R)` yields the same coordinates as before for the same
  `R`, so code that only used the helpers is unaffected.

  `test_rotation_matrix_matches_ase_handedness` now pins agreement with
  `ase.Atoms.rotate` on all three axes, plus `det(R) = +1`.

## [0.5.1] - 2026-10-31

### Breaking

- **`transform_forces(f, R)` now returns `f @ R.T` (previously `f @ R`).**
  Forces are polar vectors: in the row-vector convention the code already
  uses (`rotate_positions(p, R) = p @ R.T`) a force must co-rotate with the
  coordinates, so the same `R` maps `+x` onto `-y` for both. The old
  `forces @ R` rotated forces in the *opposite* direction -- a rigid body's
  forces did not follow its coordinates. Migration: external consumers that
  compensated by passing `R.T` themselves can simply drop that
  compensation. There are no internal callers (verified by grep at the time
  of the change).

### Fixed

- **`check_force_consistency` is a real finite-difference check.** It used
  to be a stub that returned `0.0` unconditionally, so "forces consistent
  with the energy" was never actually verified anywhere. It now computes
  `max_i |F_i - (-dE/dr_i)|` by a central finite difference of a caller
  supplied energy function and never mutates the input positions. The
  signature changes accordingly: the first argument is now an energy
  function handle `energy_fn(positions) -> float` instead of a scalar
  energy. The old stub had no real consumers (verified by grep).
- **`current_seed` reads a real fingerprint of the numpy RNG state.** It
  used to run `int(state[0])` on numpy's `get_state()` tuple, whose first
  element is the algorithm name string `'MT19937'` -- the call always
  raised `ValueError` and was silently swallowed into `None`, so the
  field was dead. It now returns a deterministic SHA-256 fingerprint of
  the 624-word MT19937 state plus stream position (state A semantics:
  a *state* fingerprint, explicitly not a seed for restoration --
  documented in the docstring). `build_meta()` now fills `seed` from it
  when no explicit seed is passed, so `metadata.yaml` always records the
  randomness a run actually used.
- **`EV_A3_TO_GPA` can no longer drift from `EV_A3_TO_PA`.** The two
  constants were independent literals in the same file and had already
  disagreed in their last four digits (160.21766208 vs the 160.2176634
  implied by the exact 1.602176634e11 Pa). `EV_A3_TO_GPA` is now derived
  as `EV_A3_TO_PA / 1e9`, so a single truth feeds both. The top level no
  longer keeps its own copy of any physics constant either:
  `atominterface.*` constants are forwards of `atominterface.physics.*`
  by identity (verified by test), and the top-level `__all__` now covers
  the full physics export surface (`rotation_matrix`, `transform_forces`,
  `transform_stress_voigt`, `ANG_TO_BOHR`, `ev_per_a3_to_gpa`,
  `gpa_to_ev_per_a3`, ...). Numeric impact of the drift fix is e-8 in
  relative terms and only meaningful in the same environment
  (float-identity conclusions are environment-bound as always).

### Added

- **The default MACE-MP-0 medium download is now integrity-checked.**
  After `_download_default_weights` moves the checkpoint into place it is
  hashed with `provenance.hash_bytes` and compared against the
  `MACE_MP0_MEDIUM_SHA256` anchor; a mismatch deletes the downloaded
  file and raises with the expected/actual hashes and recovery guidance,
  so a corrupt or substituted checkpoint can never be silently used.
  The check deliberately covers only the *current* download: a cache
  hit from a previous run (or the official mace helper) is returned
  without verification, so existing working setups never break. The
  anchor is currently `None` -- the implementation environment could not
  reach the release asset on 2026-10-31 (proxy refused the object-store
  tunnel), and a hash is only ever recorded from a genuine download,
  never hand-entered. Until a trusted environment fills it, the check
  skips gracefully and behaviour is unchanged; the fill procedure is
  documented beside the constant.

### Documentation

- `docs/roadmap.md` records two facts for the next cycle: the
  `pydantic` dependency is declared in `pyproject.toml` but referenced
  nowhere in `src/` (per the user decision the declaration stays, the
  fact is recorded, and disposal is deferred to a release window), and
  the monolith split candidates for `builders/coherent.py` (2700+
  lines) and `builders/charge.py` (932 lines) with their cut lines --
  including the one silent fallback in `coherent.py` (the single-match
  rotation `suppress(Exception)`) that is deliberately left for that
  split rather than widened in this cycle.

## [0.5.0] - 2026-10-30

**This release changes the numbers a (110) build produces.** Read "Changed"
before upgrading: every γ_int you have computed on (110) with 0.4.1 or
earlier moves, and the old values are now reachable only on request.

### Added

- **A `registry` field that names the lateral site of the join.**
  `CoherentInterfaceConfig.registry` (and `--registry` on `atominterface
  build`) accepts `"auto"` (default), `"on-top"`, `"hcp"`, an explicit
  `(u, v)` in-plane fractional offset (CLI: `--registry 0.5,0.5`), or
  `None`. The join is now *placed on* a site instead of inheriting
  whatever the two slab cuts left behind. Placement is targeted: the
  site is named first, and the translation that reaches it is solved for
  (`shift = d − target`), not added incrementally.
- **Three-part provenance for the placement.** Every candidate carries
  `registry_requested` (what you asked for), `registry_target_frac` (what
  that resolved to, in fractional coordinates) and
  `registry_as_built_frac` / `registry_as_built_lateral_A` (what the
  builder actually produced). `registry_lateral_offset_ang` is unchanged
  and still consumed downstream. A request and its result are separate
  fields on purpose: collapsing them hides a request that was not
  honoured.
- **`measure_registry(atoms)`**, public, re-measures the join site of a
  relaxed or re-read structure carrying the `SLAB_ID` array. After
  relaxation the CLI writes `registry_final_frac` /
  `registry_final_lateral_A` beside the relax record, so "it relaxed
  away from the site I asked for" is a readable fact rather than
  something a lateral distance has to be guessed from.
- **A placement self-check.** After the film is moved the join is
  re-measured and compared with the request in fractional coordinates;
  a miss beyond `REGISTRY_PLACEMENT_TOL` (1e-6) raises `ValueError`
  naming the requested site, the actual site, the deviation and the
  sites that exist on that surface. A site that cannot be *measured*
  raises too — it never falls back silently to the as-built position.

### Changed

- **The default is `registry="auto"`, and on (110) it moves the film.**
  `auto` asks for the site that continues the substrate's own stacking
  sequence. Measured on Cu/Ni:

  | orientation | as-built site (`registry=None`) | `auto` target | effect |
  |---|---|---|---|
  | (001) 10.8/14.4/18.0/21.6 Å | (0, −½), 1.8075 Å | (0, +½) — the same site | **no move, all four thicknesses** |
  | (111) 10.8/14.4/18.0 Å | fcc hollow (−⅓, −⅓), 1.4758 Å | (−⅓, −⅓) | no move |
  | (111) 21.6 Å | hcp hollow (+⅓, +⅓), 1.4758 Å | fcc hollow (−⅓, −⅓) | moves 1.4758 Å |
  | (110) 10.8/14.4/18.0/21.6 Å | ridge (½, 0), 1.8075 Å | trough (½, ½), 2.2137 Å | moves 1.2781 Å |

  The fractions are what `registry_as_built_frac` reports, and they are
  read before relaxation. **Which physical site a given `(u, v)` names
  depends on the slab, not only on the surface**: on (111) the cut lands
  on the fcc hollow at 10.8/14.4/18.0 Å and on the hcp hollow at 21.6 Å,
  so `(u, v) = (⅓, ⅓)` is *hcp* here while `(−⅓, −⅓)` is *fcc*. That is
  why the vocabulary has no site names — `auto` and `hcp` resolve the
  site from the board's own stacking, and a `(u, v)` is a fraction you
  have to have measured, not a name you can carry between slabs.

  **(001) is bit-identical across this change** — verified with
  `np.array_equal` on all four thicknesses, not with `allclose`. It is
  the only orientation this project has a DFT cross-check for, and it
  had to stay untouched.
- **Every (110) number from 0.4.1 and earlier changes.** The as-built
  (110) ridge site reads γ_int 0.5952 J/m²; the trough that `auto` now
  asks for reads 0.0881 J/m², 6.8× lower and the scanned minimum of
  `docs/validation/cu_ni_multi_orientation/registry_110_placement.md`.
  If you need the old numbers, ask for them: `registry=None` reproduces
  the 0.4.1 cut verbatim and `(0.5, 0.0)` names the ridge.
- **The non-(001) warning now names the site you asked for.** It used to
  say the registry was inherited from the slab cuts and that there was
  no configuration field to request one; that half is no longer true.
  The warning still fires, and still must: requesting one site is not a
  scan of the others, and only `registry="scan"` would retire it.

### Fixed

- **The in-plane placement extrapolated instead of aiming.** v0.4.1's
  `_off_on_top_shift` *added* the substrate stacking step to whatever
  the cut produced, which is correct only where the cut leaves the film
  on a sublattice site. It is right on fcc(111) (on-top + step = fcc
  hollow) and wrong on fcc(110), where the cut leaves the film on the
  ridge and ridge + step = short bridge: γ_int 0.5952 → 1.5189 J/m²,
  2.55× *worse* than the defect it was fixing, still `valid=True`, still
  converged. Targeted placement gives 0.0881 on the same board.
- **A cartesian in-plane difference was being wrapped as if it were
  fractional.** Wrapping is only meaningful after converting with
  `inv(sub_2d)`; on a non-cubic surface lattice the shortcut does not
  raise, it places the film 0.31 of a cell away from the requested site
  with every downstream check green.

### Validation

- **`registry` reproduces the archive's own four-site scan.** Cu/Ni(110)
  at 14.4 Å, MACE-MP-0, the protocol of
  `docs/validation/cu_ni_multi_orientation` (vacuum 15 Å, fmax 0.002,
  400 steps, 1 candidate). Every column is a value the archive measured
  independently, by shifting the film by hand:

  | `--registry` | site | E_interface (eV) | γ_int (J/m²) | archive |
  |---|---|---|---|---|
  | `none` | ridge | −115.0994 | 0.595 | −115.099365 / 0.5952 |
  | `0,0.5` | short bridge | −114.5666 | 1.519 | −114.566643 / 1.5189 |
  | `auto` | trough | −115.3919 | 0.088 | −115.391869 / 0.0881 |
  | `on-top` | on-top | −114.2700 | 2.033 | −114.270020 / 2.0332 |

  `auto` lands on the scanned minimum, not merely on a different site.
- **`(111)` at 18.0 Å: `auto` gives −87.0464 eV**, the same number the
  archive records for the fcc hollow, and `hcp` gives −87.0465 eV —
  lower by 1e-4 eV, the direction the archive's 0.0810 vs 0.0812 J/m²
  implies. The two hollows are 1.4758 Å from a substrate atom either
  way, so a lateral distance cannot tell them apart; site labels were
  assigned from the board's own plane geometry (an atom directly below
  the join in the second plane is hcp, in the third is fcc).
- **`(111)` at 21.6 Å is the thickness that used to start on hcp** and
  now starts on fcc, which is what makes `--min-thickness` a single
  variable again.

### Not implemented

- `registry="scan"` — the explicit enumeration of every site of the
  surface, which is what would actually retire the non-(001) warning.
  The name is reserved and rejected with a message saying so.

## [0.4.1] - 2026-10-30

A patch release, and it ships two things: the (111) on-top fix itself, which
changed numerical output and had been sitting on `main` unreleased, and the
measurement of what that fix is actually worth.

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
  sites, and there is no configuration field to request one. (The last
  clause is only true of 0.4.1: v0.5.0 adds the `registry` field and the
  warning names the requested site.)
- The (111)/(110) validation archive is marked as partially superseded: its
  on-top value 0.8980 J/m² is a pre-fix record that current code does not
  reproduce. Its 0.5953 J/m² (110) ridge value still reproduced here —
  (110) was deliberately left untouched — but **v0.5.0 supersedes it too**,
  so the archive now records both as reachable only on request.

### Validation

- **The superseded value was re-measured: 0.8981 → 0.0812 J/m²** on Cu/Ni(111)
  at 14.4 Å. One re-run cannot answer this, because the archive was produced
  with mace-torch 0.3.17 / pymatgen 2026.10.24 and re-running today would
  change the code and the environment together. Both arms share one process and
  one protocol and differ only in that the pre-fix arm replaces
  `_off_on_top_shift` with a zero-valued function. The control arm reproduces
  the archived 0.8981 and its surface terms (1.2994 / 1.8237 J/m²), and 18.0 Å
  builds bit-identically under both arms — `E_interface` agrees to six decimals
  — so the 14.4 Å delta is attributable to the shift alone. The arithmetic
  closes: ΔE_interface = −0.288513 eV over 5.658707 Å² is 0.8169 J/m², and
  0.8981 − 0.0812 = 0.8169. Recorded in
  `docs/validation/cu_ni_multi_orientation/on_top_remeasured.md`.
- **The site is confirmed independently of the number.** Post-fix the joining
  layer sits at fractional (+1/3, +1/3), 1.4758 Å from on-top and 0.0000 Å from
  the ideal position, with a void in the plane immediately below it: the fcc
  hollow, the same offsets the archive's own four-site scan records at 18.0 Å.
- **One limit is recorded rather than left implicit:** the film does not move
  in plane in either arm, because a hollow is also a three-fold stationary
  point. This fixes the *starting placement*, not the search — which is why
  `CoherentInterfaceConfig` still has no `registry` field and a user still
  cannot ask for a site. (Superseded in v0.5.0, which lets a user name a
  site. What v0.5.0 still does not do is the *search*: `registry="scan"`
  is reserved, not implemented.)
- The 14.4 Å fmax comparison that was skipped because the on-top lock would
  have swamped it has now been run: 0.002 → 0.0812, 0.01 → 0.0817 J/m², a
  difference of +0.0005 — an order of magnitude below the (001) cluster's
  +0.0050 and three below the 0.817 the registry defect was worth.

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
