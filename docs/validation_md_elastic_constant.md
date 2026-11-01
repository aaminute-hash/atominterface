# Validation note — elastic constant C11 from small-strain MD (LAMMPS Cu)

**Status:** measured, not modelled. `C11` is extracted from the slope of the
Cauchy stress `sigma_xx` versus a series of applied uniaxial strains, produced
by real LAMMPS NVT runs on SCNET.

**Question this note answers:** does the Phase 0.4 stress/strain pipeline
recover a physically correct elastic constant when fed real MD data — i.e.
`d sigma_xx / d eps_xx = C11` for Cu near 168 GPa?

Short answer: yes. Five uniaxial-strain points give `C11 = 170.01 GPa` with
`R^2 = 0.9996`, within 1.2% of the experimental 298 K value (168 GPa).

---

## 1. Method

- System: Cu fcc, 256 atoms (4×4×4), Mishin EAM (`Cu_mishin1.eam.alloy`).
- Ensemble: NVT at 300 K (Nosé-Hoover), fixed cell (uniaxial strain — only
  `Lx` is changed; `eps_yy = eps_zz = 0`).
- Strains: `eps_xx ∈ {−1.0%, −0.5%, 0, +0.5%, +1.0%}` (five independent jobs,
  independent random seeds).
- Each job: 2000 equilibration steps, then 8000 sampling steps; `thermo` every
  500 steps (16 samples), mean `pxx` taken over the sampling window.
- `sigma_xx = -pxx * 1e-4` GPa (pressure positive under compression → Cauchy
  tension positive).
- `C11 = d sigma_xx / d eps_xx` from a least-squares line through the five
  points. With `eps_yy = eps_zz = 0`, `sigma_xx = C11 * eps_xx`, so the slope
  is exactly `C11`.

## 2. Results

| `eps_xx` | `sigma_xx` (GPa, mean) | std (GPa) | T (K) |
|---|---|---|---|
| −0.0100 | −3.6540 | 0.114 | 302.8 |
| −0.0050 | −2.7930 | 0.095 | 299.5 |
| 0.0000 | −1.9620 | 0.104 | 296.5 |
| +0.0050 | −1.0566 | 0.108 | 301.6 |
| +0.0100 | −0.2721 | 0.113 | 303.7 |

Least-squares fit `sigma_xx = C11 * eps_xx + b`:

| quantity | value |
|---|---|
| **C11** | **170.01 GPa** |
| intercept `b` | −1.948 GPa |
| `R^2` | 0.999648 |
| Cu reference (experiment, 298 K) | ~168 GPa |
| relative deviation | **+1.2%** |

## 3. Interpretation

- The slope `C11 = 170.01 GPa` matches the experimental Cu value (168 GPa) to
  1.2%, i.e. the Mishin EAM potential reproduces the Cu elastic constant to
  within its own accuracy.
- The intercept `b = −1.95 GPa` is a residual *hydrostatic* compression common
  to every strain point: the fixed-cell NVT run leaves a small pressure at
  `eps = 0` (the same ~1.9 GPa seen in the bridge cross-check note). It does
  not affect the slope — `C11` is read from `d sigma / d eps`, not the absolute
  stress — which is why the small-strain differential method is robust to this
  finite-size/fixed-cell artefact.
- Stress std over the sampling window (~0.1 GPa) is ~0.6% of the `C11 * 1%`
  span (~1.7 GPa), consistent with the tight `R^2`.

## 4. Environment / reproduce

LAMMPS `lammps-2Aug2023-intelmpi2018` on SCNET, 8 MPI ranks per job, single
node (`I_MPI_FABRICS=shm`). Inputs: `lammps/in.strain` (strain applied via a
`lattice custom` box with `ax = 3.615*(1+eps)`), `lammps/job_strain.slurm`
(job array 0–4). Each job takes ~10 s. Analysis script:
`p04_md_validation/analyze_c11.py` (reads `log.strain.{0..4}`, fits the slope).
