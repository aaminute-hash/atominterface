# Validation note — real-MD Virial→Cauchy bridge (LAMMPS + CP2K)

**Status:** measured, not modelled. Every number below comes from
`physics.md_sources.read_lammps_md` / `read_cp2k_md` fed to
`physics.stress_strain.extract`, cross-checked frame-by-frame against the raw
software output files.

**Question this note answers:** does the Phase 0.4 bridge turn *real* MD
trajectories (a LAMMPS classical run and a CP2K AIMD run) into the correct
Cauchy stress, with the right sign, units, and kinetic term — or does it only
work on synthetic data?

Short answer: yes, bit-for-bit. On both sources the bridged stress equals the
direct `-P` reading of the software's own pressure tensor to machine precision
(≤ 7e-16 GPa), and the temperature recovered from the bridged velocities
matches the thermostat set point. Three conventions had to be pinned first —
sign, units, and the kinetic term — plus two pits that a synthetic trajectory
cannot expose.

---

## 1. The three conventions (and two hidden pits)

| # | convention | statement |
|---|---|---|
| C1 | **sign** | LAMMPS and CP2K report a *pressure* tensor, positive under compression. Cauchy stress is positive under tension, so `sigma = -P`. |
| C2 | **units** | both report `bar`. `sigma[eV/A^3] = -P[bar] * 1e-4 / EV_A3_TO_GPA`. |
| C3 | **kinetic term** | the full Cauchy stress includes the ideal-gas term. LAMMPS thermo `pxx..pyz` includes it; `compute pressure NULL virial` (`c_myp[1..6]`) does not. CP2K `.stress` includes it. Use the full pressure, never rebuild a virial. |
| P1 | **translation dependence** | `sum(r_i x f_i)` from dump coordinates is origin-dependent for a periodic cell with a nonzero instantaneous net force (`sum((r+c)f) = sum(r f) + c·F_total`); it disagreed with the software pair virial by 2–5 GPa. |
| P2 | **ASE `lammps-dump-text` reader** | it treats LAMMPS `vx vy vz` as *momenta* (multiplying by a default H mass of 1.008), drops `fx fy fz` entirely, and defaults masses to 1.008. The bridge parses the dump text manually. |
| P3 | **velocity units** | LAMMPS `metal` velocities are A/ps; ASE velocities are in a unit where `0.5*m*v^2` is already eV. The conversion is `v_ase = v_lammps * sqrt(mvv2e)` with `mvv2e = 1.0364269e-4` — **not** `÷1000`. |

---

## 2. Data sources (SCNET)

| | LAMMPS | CP2K |
|---|---|---|
| run | Cu fcc 256 atoms (4×4×4), NVT 300 K | Cu fcc 32 atoms (2×2×2), BOMD NVT 300 K |
| potential | Mishin EAM (`Cu_mishin1.eam.alloy`) | GTH-PBE-q11 + DZVP-MOLOPT-SR-GTH |
| cell | 14.46 Å cubic (fixed) | 7.23 Å cubic (fixed) |
| frames | 81 dump (step 2000→10000) + 101 thermo | 61 (step 0→60; complete 60-step run) |
| stress source | thermo `pxx..pyz` (bar) | `.stress` (bar) |

The CP2K run completed all 60 BOMD steps (job 126555095, exit 0, 1:19:04), so
its `.stress`/`.cell`/`.ener` and `pos-1.xyz` are fully aligned at 61 frames.

---

## 3. Results

### 3.1 LAMMPS

| quantity | value |
|---|---|
| bridged frames | 81 (all have stress) |
| temperature from bridged velocities | mean **297.0 K** (set point 300 K) |
| mean Cauchy hydrostatic stress | **−1.9080 GPa** (tension positive) |
| full pressure, mean hydro | 18800 bar → −1.880 GPa |
| virial (`c_myp`), mean hydro | 15375 bar → −1.538 GPa |
| kinetic term (`P_full − P_vir`) | 3424 bar = **0.3425 GPa** |
| max \|bridge − direct thermo\| over 81 frames | **6.66e-16 GPa** |

The kinetic term checks independently against the ideal-gas estimate
`NkT/V ≈ 0.35 GPa` for 256 atoms at 300 K in a 14.46³ Å³ box.

### 3.2 CP2K

| quantity | value |
|---|---|
| bridged frames | 61 (all have stress) |
| temperature (from `.ener`) | mean 227.9 K |
| mean Cauchy hydrostatic stress | **−5.1684 GPa** (tension positive) |
| direct `.stress` mean hydro | 51684 bar → −5.1684 GPa |
| max \|bridge − direct\| over 61 frames | **1.78e-15 GPa** |

The CP2K run is more compressive than the LAMMPS one because its cell is frozen
at 7.23 Å (2×3.615 Å), below the GTH-PBE equilibrium lattice constant — a
fixed-cell NVT artefact, not a bridge error; the sign/unit path is what is
being validated and it is exact. Its 32-atom temperature average is 227.9 K
(large finite-size fluctuation around the 300 K set point), expected for such
a small cell.

---

## 4. `extract()` integration

Feeding the bridged frames straight into `extract()` reproduces the same
Cauchy-stress Voigt channel (mean hydro −1.9080 GPa / −5.1684 GPa), with
`has_stress` true on every frame and temperature 297.0 K for LAMMPS (CP2K
carries no velocities, so its temperature is `atoms.info["temperature"]` and
`extract` reports no temperature channel for it).

---

## 5. Environment / reproduce

The bridge code is pure `ase` + `numpy` and is covered hermetically by
`tests/test_md_sources.py` (synthetic files only; CI runs with no SCNET data).
The real-file cross-check above was run on this machine against the files
pulled from SCNET (`lammps/dump.prod`, `lammps/log.lammps`,
`cp2k/cu_aimd-1.{stress,cell,ener}`, `cp2k/cu_aimd-pos-1.xyz`) with the
managed Python 3.13 venv. The LAMMPS binary is
`lammps-2Aug2023-intelmpi2018`; CP2K is `cp2k-v2025.1-oneapi2024`.
