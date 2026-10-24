"""Interface thermodynamics: formation energy, work of separation, CZM fitting."""

from __future__ import annotations

import numpy as np

from atominterface.thermo.surface import (
    _EV_A2_TO_UNIT,
    TerminationRecord,
    _finite_real,
    rank_terminations,
    slab_formula_units,
    termination_energy,
)

__all__ = [
    "TerminationRecord",
    "fit_cohesive_zone",
    "interface_energy",
    "interface_formation_energy",
    "rank_terminations",
    "slab_formula_units",
    "termination_energy",
    "work_of_separation",
]

# NumPy 2.0 renamed ``trapz`` to ``trapezoid``.  Both names are still in
# circulation depending on the installed NumPy, so resolve once at import
# time to stay compatible with the full ``numpy>=1.21`` support range.
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # type: ignore[attr-defined]


def interface_energy(
    *,
    interface_energy_total: float,
    substrate_energy_per_area: float,
    film_energy_per_area: float,
    area: float,
    unit: str = "eV_A2",
) -> float:
    """Interface energy per unit area.

    ``gamma = (E_interface - E_substrate - E_film) / A``

    The ``*_per_area`` parameter names are historical: every energy
    argument is a **total** energy (eV) of the whole system, and ``A``
    is the interface area (A^2) that total is divided by.  When the two
    surface energies are known, :func:`interface_formation_energy` is
    the physically referenced form: it charges the joint for the
    surfaces that were destroyed to make it.
    """
    if area <= 0:
        raise ValueError("interface area must be positive")
    if unit != "eV_A2":
        raise ValueError(f"unsupported unit {unit!r}")
    return (interface_energy_total - substrate_energy_per_area - film_energy_per_area) / area


def interface_formation_energy(
    *,
    interface_energy_total: float,
    substrate_slab_energy: float,
    film_slab_energy: float,
    substrate_surface_energy: float,
    film_surface_energy: float,
    area: float,
    unit: str = "eV_A2",
) -> float:
    """Interface formation energy per unit area against isolated slabs.

    Two boards that merely touch do not form an interface: the energy
    of the joint is the sum of the two boards **plus** the surfaces
    that were destroyed to make it.  Referencing every side to its own
    bulk, with ``A`` the coherent interface area,

        E_A_slab = n_A * E_bulk_A + 2 * A * gamma_A_surf
        E_B_slab = n_B * E_bulk_B + 2 * A * gamma_B_surf
        E_int    = n_A * E_bulk_A + n_B * E_bulk_B
                    + A * (gamma_A_surf + gamma_B_surf + gamma_int)

    so eliminating the bulk references (``n_A * E_bulk_A`` and
    ``n_B * E_bulk_B`` cancel on both sides) leaves

        gamma_int = (E_int - E_A_slab - E_B_slab) / A
                    + gamma_A_surf + gamma_B_surf.

    ``substrate_slab_energy`` / ``film_slab_energy`` are the **total**
    energies (eV) of the two isolated, relaxed slabs -- not per-area
    values -- while ``substrate_surface_energy`` /
    ``film_surface_energy`` are per-area termination energies, e.g.
    from :func:`termination_energy` with ``n_surfaces=2`` (one
    symmetric board exposes two surfaces, hence the factor 2 above).

    The same expression is the separation-work form: with the work of
    adhesion ``W_ad = (E_A_slab + E_B_slab - E_int) / A`` (see
    :func:`work_of_separation`), the identity

        gamma_int = gamma_A_surf + gamma_B_surf - W_ad

    holds exactly -- the formation energy is the work of adhesion
    discounted by the two surface energies that were given up.  A
    non-positive ``area`` or an unknown ``unit`` raises ``ValueError``.
    """
    if unit not in _EV_A2_TO_UNIT:
        raise ValueError(f"unsupported unit {unit!r}")
    interface_total = _finite_real(interface_energy_total, "interface_energy_total")
    substrate_slab = _finite_real(substrate_slab_energy, "substrate_slab_energy")
    film_slab = _finite_real(film_slab_energy, "film_slab_energy")
    substrate_surface = _finite_real(substrate_surface_energy, "substrate_surface_energy")
    film_surface = _finite_real(film_surface_energy, "film_surface_energy")
    surface_area = _finite_real(area, "area")
    if surface_area <= 0:
        raise ValueError("interface area must be positive")
    energy = (
        (interface_total - substrate_slab - film_slab) / surface_area
        + substrate_surface
        + film_surface
    )
    return energy * _EV_A2_TO_UNIT[unit]


def work_of_separation(
    *,
    interface_energy_total: float,
    separated_energy_total: float,
) -> float:
    """Work required to separate the interface (per the whole interface)."""
    return separated_energy_total - interface_energy_total


def fit_cohesive_zone(
    displacement: np.ndarray,
    traction: np.ndarray,
    *,
    peak_traction: bool = True,
) -> dict:
    """Minimal CZM parameter estimate from a traction-separation curve.

    Returns a dict with:
        max_traction (sigma_max),
        critical_separation (delta_c),
        fracture_energy (G_c, area under curve).
    """
    disp = np.asarray(displacement, dtype=float)
    trac = np.asarray(traction, dtype=float)
    if disp.ndim != 1 or trac.ndim != 1 or disp.size != trac.size:
        raise ValueError("displacement and traction must be 1D arrays of equal length")
    if not np.all(np.diff(disp) >= 0):
        order = np.argsort(disp)
        disp = disp[order]
        trac = trac[order]

    imax = int(np.argmax(trac))
    if 0 < imax < len(trac) - 1:
        # Fit ascending and descending segments with least-squares lines,
        # then intersect them.  Exact for piecewise-linear (bilinear) CZM.
        asc_x = disp[: imax + 1]
        asc_y = trac[: imax + 1]
        desc_x = disp[imax:]
        desc_y = trac[imax:]
        a1, b1 = np.polyfit(asc_x, asc_y, 1)
        a2, b2 = np.polyfit(desc_x, desc_y, 1)
        denom = a1 - a2
        if abs(denom) > 1e-12:
            x_peak = float((b2 - b1) / denom)
            y_peak = float(a1 * x_peak + b1)
        else:
            # Degenerate: take the discrete max directly.
            x_peak = float(disp[imax])
            y_peak = float(trac[imax])
    else:
        x_peak = float(disp[imax])
        y_peak = float(trac[imax])
    sigma_max = y_peak
    delta_c = x_peak
    G_c = float(_trapezoid(trac, disp))
    return {
        "max_traction": sigma_max,
        "critical_separation": delta_c,
        "fracture_energy": G_c,
    }
