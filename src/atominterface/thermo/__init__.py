"""Interface thermodynamics: formation energy, work of separation, CZM fitting."""

from __future__ import annotations

import numpy as np


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
    if unit not in ("eV_A2", "J_m2"):
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

    sigma_max = float(np.max(trac))
    imax = int(np.argmax(trac))
    delta_c = float(disp[imax])
    G_c = float(np.trapz(trac, disp))
    return {
        "max_traction": sigma_max,
        "critical_separation": delta_c,
        "fracture_energy": G_c,
    }
