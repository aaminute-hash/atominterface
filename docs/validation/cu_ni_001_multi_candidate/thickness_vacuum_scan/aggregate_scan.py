#!/usr/bin/env python
"""Aggregate the thickness/vacuum scan groups into one machine-readable CSV.

One row per (group, candidate) across every group directory under
F:/tmp/d1_thickness, carrying the group name, the knobs that were
swept, layer counts, the three gammas and the interface area. Groups
share the baseline (thickness 14.4 A, vacuum 15.0 A), so the baseline
row is emitted once under its own group name.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

EV_A2_TO_J_M2 = 16.02176634  # atominterface.physics.EV_A2_TO_J_M2

ROOT = Path("F:/tmp/d1_thickness")

GROUPS = [
    # (directory, group label, min_thickness, vacuum, min_thickness_substrate, min_thickness_film)
    ("thick_10.8_vac15", "thickness_scan", 10.8, 15.0, None, None),
    ("base_14.4_vac15", "thickness_scan", 14.4, 15.0, None, None),
    ("thick_18.0_vac15", "thickness_scan", 18.0, 15.0, None, None),
    ("thick_14.4_vac10", "vacuum_scan", 14.4, 10.0, None, None),
    ("base_14.4_vac15", "vacuum_scan", 14.4, 15.0, None, None),
    ("thick_14.4_vac20", "vacuum_scan", 14.4, 20.0, None, None),
    ("split_sub18_film10.8", "split_thickness", 14.4, 15.0, 18.0, 10.8),
]

FIELDS = [
    "group",
    "min_thickness_A",
    "min_thickness_substrate_A",
    "min_thickness_film_A",
    "vacuum_A",
    "rank",
    "interface_id",
    "area_A2",
    "n_layers_sub",
    "n_layers_film",
    "gamma_surf_sub_eV_A2",
    "gamma_surf_film_eV_A2",
    "gamma_int_eV_A2",
    "gamma_int_J_m2",
    "gamma_Cu_J_m2",
    "gamma_Ni_J_m2",
    "status",
]


def main() -> int:
    rows = []
    for directory, group, mt, vac, mt_sub, mt_film in GROUPS:
        csv_path = ROOT / directory / "candidates.csv"
        if not csv_path.exists():
            print(f"MISSING: {csv_path}", file=sys.stderr)
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                if record["status"] != "ok":
                    continue
                gcu = float(record["gamma_surf_sub_eV_A2"]) * EV_A2_TO_J_M2
                gni = float(record["gamma_surf_film_eV_A2"]) * EV_A2_TO_J_M2
                rows.append(
                    {
                        "group": group,
                        "min_thickness_A": f"{mt:.4f}",
                        "min_thickness_substrate_A": "" if mt_sub is None else f"{mt_sub:.4f}",
                        "min_thickness_film_A": "" if mt_film is None else f"{mt_film:.4f}",
                        "vacuum_A": f"{vac:.4f}",
                        "rank": record["rank"],
                        "interface_id": record["interface_id"],
                        "area_A2": record["area_A2"],
                        "n_layers_sub": record["n_layers_sub"],
                        "n_layers_film": record["n_layers_film"],
                        "gamma_surf_sub_eV_A2": record["gamma_surf_sub_eV_A2"],
                        "gamma_surf_film_eV_A2": record["gamma_surf_film_eV_A2"],
                        "gamma_int_eV_A2": record["gamma_eV_per_A2"],
                        "gamma_int_J_m2": record["gamma_J_per_m2"],
                        "gamma_Cu_J_m2": f"{gcu:.4f}",
                        "gamma_Ni_J_m2": f"{gni:.4f}",
                        "status": record["status"],
                    }
                )
    out_path = ROOT / "thickness_vacuum_scan.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
