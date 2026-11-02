#!/usr/bin/env python
"""FEniCSx unit-cell example: constitutive table -> linear-elastic solve.

Phase 0.4 (AutoConstitutive) closing item.  This example *consumes* the NumPy
constitutive table written by :file:`examples/02_md_stress_strain.py` and closes
the loop from a guarded MD trajectory to a continuum finite-element solve::

    02_md_stress_strain.py   ->  constitutive_table.npz
    04_fenicsx_unit_cell.py  ->  fit Voigt stiffness C  ->  dolfinx solve

The fitting half (:func:`fit_stiffness`) is pure NumPy and runs everywhere; the
FEniCSx half (:func:`solve_unit_cell`) imports dolfinx/ufl lazily and needs a
FEniCSx installation (``conda install -c conda-forge fenicsx``, or a PETSc/
dolfinx build).  Without FEniCSx the script still fits ``C``, prints it, and
exits cleanly with an install hint.

The Voigt convention follows the rest of the package: order
``[xx, yy, zz, yz, xz, xy]`` with **tensor** shear components (no 2x
engineering factor), matching :func:`atominterface.physics.stress_strain`
and ASE's ``get_stress(voigt=True)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")


def load_table(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Read a ``.npz`` constitutive table back into (strain, stress_GPa, units).

    ``strain_voigt`` is dimensionless Green-Lagrange strain; the GPa stress
    channel is returned so that the fitted stiffness is in GPa.
    """
    data = np.load(path)
    strain = np.asarray(data["strain_voigt"], dtype=float)
    stress = np.asarray(data["cauchy_stress_voigt_gpa"], dtype=float)
    units = json.loads(data["units"].item())
    return strain, stress, units


def fit_stiffness(strain_voigt: np.ndarray, stress_voigt: np.ndarray) -> np.ndarray:
    """Least-squares Voigt stiffness ``C`` (6x6): ``stress = strain @ C.T``.

    Both inputs are ``(N, 6)`` in the package Voigt order with tensor shear
    components.  The fitted ``C`` is symmetrised, since the elastic stiffness
    is symmetric by the minor and major symmetries.
    """
    strain = np.asarray(strain_voigt, dtype=float)
    stress = np.asarray(stress_voigt, dtype=float)
    if strain.ndim != 2 or stress.ndim != 2:
        raise ValueError("strain and stress must be (N, 6) arrays")
    if strain.shape[1] != 6 or stress.shape[1] != 6:
        raise ValueError("strain and stress must have 6 Voigt columns")
    if strain.shape[0] != stress.shape[0]:
        raise ValueError("strain and stress must have the same number of rows")
    c = np.linalg.lstsq(strain, stress, rcond=None)[0].T
    return 0.5 * (c + c.T)


def voigt_matrix_to_tensor(sv: np.ndarray) -> np.ndarray:
    """Reassemble a 6-component Voigt vector into a symmetric 3x3 tensor.

    Order ``[xx, yy, zz, yz, xz, xy]`` maps onto the upper triangle as
    ``[[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]``.
    """
    s = np.asarray(sv, dtype=float).reshape(6)
    return np.array(
        [
            [s[0], s[5], s[4]],
            [s[5], s[1], s[3]],
            [s[4], s[3], s[2]],
        ]
    )


def _synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """A small isotropic constitutive dataset (Cu-like) for the no-table path.

    Stress is generated linearly from a known isotropic ``C`` so that
    :func:`fit_stiffness` can recover it exactly.
    """
    lam, mu = 100.0, 50.0  # Lame parameters (GPa)
    c0 = np.zeros((6, 6))
    for i in range(3):
        c0[i, i] = lam + 2.0 * mu
    for i in range(3):
        for j in range(3):
            if i != j:
                c0[i, j] = lam
    for i in range(3, 6):
        c0[i, i] = 2.0 * mu

    rng = np.random.RandomState(0)
    strain = rng.uniform(-0.005, 0.005, size=(24, 6))
    # A uniaxial-dominant sweep so the fit is well-conditioned.
    strain[:, 0] = np.linspace(-0.01, 0.02, 24)
    stress = strain @ c0.T
    return strain, stress


def solve_unit_cell(
    c_voigt: np.ndarray,
    stretch: float = 1.02,
    nx: int = 4,
) -> tuple[object, float]:
    """Solve a unit cube under uniaxial stretch with FEniCSx (dolfinx).

    Requires ``mpi4py``, ``dolfinx`` and ``ufl``.  Returns ``(uh, mean_sxx)``:
    the displacement field and the volume-averaged ``sigma_xx`` (GPa), the
    latter to be compared against ``C[0, 0] * (stretch - 1)`` for the
    side-clamped uniaxial case.

    This function is the FEniCSx 0.7.x flavour; the dolfinx API moved across
    minor versions, so a different installed version may need small adjustments.
    """
    try:
        import dolfinx  # noqa: PLC0415
        import ufl  # noqa: PLC0415
        from mpi4py import MPI  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "FEniCSx is not installed. Install it with "
            "'conda install -c conda-forge fenicsx' (or a dolfinx build) "
            "and retry with --fenicsx."
        ) from exc

    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, nx, nx, nx)
    element = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
    vspace = dolfinx.fem.functionspace(mesh, element)

    c_mat = ufl.as_matrix(np.asarray(c_voigt, dtype=float))

    def eps_fn(w: object) -> object:
        return ufl.sym(ufl.grad(w))

    def sigma_fn(w: object) -> object:
        e = eps_fn(w)
        e_voigt = ufl.as_vector([e[0, 0], e[1, 1], e[2, 2], e[1, 2], e[0, 2], e[0, 1]])
        s_voigt = ufl.dot(c_mat, e_voigt)
        return ufl.as_tensor(
            [
                [s_voigt[0], s_voigt[5], s_voigt[4]],
                [s_voigt[5], s_voigt[1], s_voigt[3]],
                [s_voigt[4], s_voigt[3], s_voigt[2]],
            ]
        )

    u_trial = ufl.TrialFunction(vspace)
    v_test = ufl.TestFunction(vspace)
    a = ufl.inner(sigma_fn(u_trial), eps_fn(v_test)) * ufl.dx
    body = dolfinx.fem.Constant(mesh, np.zeros(3))
    l_form = ufl.inner(body, v_test) * ufl.dx

    disp = float(stretch - 1.0)

    def on_left(x: np.ndarray) -> np.ndarray:
        return np.isclose(x[0], 0.0)

    def on_right(x: np.ndarray) -> np.ndarray:
        return np.isclose(x[0], 1.0)

    def right_value(x: np.ndarray) -> np.ndarray:
        return np.stack(
            (
                np.full(x.shape[1], disp),
                np.zeros(x.shape[1]),
                np.zeros(x.shape[1]),
            )
        )

    bcs = [
        dolfinx.fem.dirichletbc(
            np.zeros(3), dolfinx.fem.locate_dofs_geometrical(vspace, on_left), vspace
        ),
        dolfinx.fem.dirichletbc(
            right_value, dolfinx.fem.locate_dofs_geometrical(vspace, on_right), vspace
        ),
    ]

    uh = dolfinx.fem.Function(vspace)
    problem = dolfinx.fem.petsc.LinearProblem(a, l_form, bcs=bcs, u=uh)
    problem.solve()

    sigma = sigma_fn(uh)
    mean_sxx = dolfinx.fem.assemble_scalar(dolfinx.fem.form(sigma[0, 0] * ufl.dx))
    return uh, float(mean_sxx)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        default=None,
        help=".npz constitutive table (default: synthetic isotropic data)",
    )
    parser.add_argument(
        "--fenicsx",
        action="store_true",
        help="also run the dolfinx unit-cell solve (needs FEniCSx)",
    )
    parser.add_argument("--stretch", type=float, default=1.02, help="uniaxial stretch factor")
    args = parser.parse_args(argv)

    if args.table is not None:
        path = Path(args.table)
        if not path.exists():
            raise SystemExit(f"table not found: {path}")
        strain, stress, _units = load_table(path)
        print(f"loaded {path} ({strain.shape[0]} frames)")
    else:
        strain, stress = _synthetic_data()
        print("using synthetic isotropic data (no --table given)")

    c = fit_stiffness(strain, stress)
    print("fitted Voigt stiffness C (GPa):")
    for i, label in enumerate(VOIGT_LABELS):
        print(f"  C[{label}] = {np.array2string(c[i], precision=2)}")

    if args.fenicsx:
        try:
            _uh, mean_sxx = solve_unit_cell(c, stretch=args.stretch)
        except RuntimeError as exc:
            print(f"\nFEniCSx skipped: {exc}")
            return 1
        predicted = c[0, 0] * (args.stretch - 1.0)
        print(f"\nFEniCSx mean sigma_xx      : {mean_sxx:.4f} GPa")
        print(f"side-clamped prediction   : {predicted:.4f} GPa")
        print(f"relative difference       : {abs(mean_sxx - predicted) / abs(predicted):.3e}")
    else:
        print("\n(no --fenicsx; only the fit was run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
