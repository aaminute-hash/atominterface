# Software support matrix

AtomInterface reads external simulation software output into `ase.Atoms` so the
physics pipeline (`extract` / `virial_to_cauchy`) and the interface builders can
consume it.  Two reader families exist:

* `physics/md_sources.py` — *trajectory* readers (MD), one `Atoms` per frame.
* `physics/dft_sources.py` — *single-point* readers (DFT), one converged `Atoms`.

Uniform conventions: energy in eV (VASP `sigma->0`, not `free energy TOTEN`);
stress as a symmetric 3x3 Cauchy tensor, **tension positive**, eV/A^3
(`sigma = -P` for codes that report pressure); forces in eV/A.

## DFT

| Software | Structure | Energy | Stress | Reader | Status |
|---|---|---|---|---|---|
| VASP | ✅ | ✅ `sigma->0` | ✅ eV/A^3 (tension +) | `read_vasp` (OUTCAR) | **verified** — SCNET OUTCAR, energy −809.06530057 bit-exact |
| Quantum ESPRESSO | ✅ | ✅ Ry→eV | ✅ Ry/bohr³→eV/A^3 (negated) | `read_qe` (pwscf.out) | **verified** — SCNET GaN `pw.x` stdout, energy −302.28381257 Ry bit-exact |
| SIESTA | ✅ (`.fdf`) | ✅ (`E_KS(eV)`) | ⚠ via `.out` | `read_siesta` | **verified** — SCNET `.fdf` (C16H4) + real `RELAX.out` energy −17531.1071 eV bit-exact |
| ABACUS | ✅ (`STRU`) | ✅ eV (direct) | ✅ kbar→eV/A^3 (negated) | `read_abacus` | **verified** — real `running_scf.log`, energy/stress bit-exact |
| CP2K | ✅ | ✅ | ✅ | `read_cp2k_md` (`.stress`) | **verified** (61-frame BOMD, ≤1.8e-15 GPa) |
| Gaussian | ✅ (`.log` orient) | ✅ Hartree→eV | ❌ (molecular) | `read_gaussian` | verified — synthetic `.log`; molecular code, no SCNET install |
| Materials Studio | ✅ (`.car`) | ⚠ `.outmol`/`.castep` | ⚠ | `read_materials_studio` (`.car`) | verified — synthetic DMol3 `.car`; proprietary platform, not on SCNET |

## MD

| Software | Trajectory | Stress | Reader | Status |
|---|---|---|---|---|
| LAMMPS | ✅ dump | ✅ thermo `pxx..pyz` | `read_lammps_md` | **verified** (81-frame NVT, ≤7e-16 GPa) |
| CP2K | ✅ `*-pos-1.xyz` | ✅ `.stress` | `read_cp2k_md` | **verified** (see DFT row) |
| GROMACS | ✅ `.gro` (single frame) | ⚠ `.edr` via `gmx energy` | `read_gromacs_gro` | structure verified; `.trr`/`.xtc` need `gmx trjconv` |
| NAMD | ✅ `.dcd` | ⚠ log | `read_dcd_md` | structure verified (synthetic DCD) |
| AMBER | ✅ `.dcd`/`.mdcrd` | ⚠ mdout | `read_dcd_md` | structure verified (synthetic DCD) |
| CHARMM | ✅ `.dcd` | ⚠ log | `read_dcd_md` | structure verified (synthetic DCD) |
| Forcite (Materials Studio) | ❌ proprietary `.xtd` | ❌ | `read_forcite_md` | `NotImplementedError` — export `.arc`/`.cif`/`.xyz` |

## Notes

* **DCD** carries no element identity and no stress; callers pass `symbols`,
  and stress must be read from each code's log/thermo separately.
* **`.xtd`** (Forcite) is a proprietary binary with no public parser and no
  SCNET installation; Forcite is supported only through the formats it can
  export (`.arc`, `.cif`, `.xyz`).  **Materials Studio** structures are read
  natively from the DMol3 `.car` (via `read_materials_studio`), and
  **Gaussian** `.log` energies/forces via `read_gaussian`; both are commercial
  codes with no SCNET installation, so they are pinned by synthetic tests.
* **Verified** means the reader was run against real (SCNET) output and the
  bridged value matches the software's own reported value to ≤1e-12 in the
  relevant unit.  **verified (synthetic)** means the sign/unit path is pinned
  by a hand-written minimal file but no production run has been read yet.
