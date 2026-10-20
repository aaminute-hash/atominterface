"""Interface thermodynamics: formation energy, work of separation, CZM fitting."""

from __future__ import annotations

import numpy as np

from atominterface.thermo.surface import (
    TerminationRecord,
    rank_terminations,
    slab_formula_units,
    termination_energy,
)

__all__ = [
    "TerminationRecord",
    "fit_cohesive_zone",
    "interface_energy",
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
    """Interface formation energy per unit area.

    gamma = (E_interface - E_substrate - E_film) / A
    """
    if area <= 0:
        raise ValueError("interface area must be positive")
    if unit != "eV_A2":
        raise ValueError(f"unsupported unit {unit!r}")
    return (interface_energy_total - substrate_energy_per_area - film_energy_per_area) / area


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
