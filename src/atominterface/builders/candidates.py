"""Candidate summary table (CSV / Markdown / list).

A dedicated export so users can eyeball the candidate set before
spending CPU on relaxation.  This is intentionally separate from
the full audit report: it is cheap and runs before relaxation.
"""

from __future__ import annotations

from typing import Any

_CHARGE_GROUPS = ("total", "substrate", "film")
_CHARGE_NUMERIC_FIELDS = (
    "n_atoms",
    "n_known",
    "net_charge_e",
    "dipole_normal_eA",
    "dipole_density_e_per_A",
)
_SUMMARY_FIELDS = (
    "interface_id",
    "substrate",
    "film",
    "sub_miller",
    "film_miller",
    "sub_termination",
    "film_termination",
    "strain_sub",
    "strain_film",
    "mean_strain",
    "area_A2",
    "n_substrate",
    "n_film",
    "n_total",
)


def charge_summary(analysis: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten the charge-analysis snapshot into stable, unit-labelled columns.

    Missing analysis on older candidates is distinct from an analysis whose
    charge model or slab geometry is unknown.  Numeric unknowns stay ``None``;
    sources and reasons are joined for CSV/Markdown rather than discarded.
    """
    data = analysis if isinstance(analysis, dict) else {}
    status = "unknown" if isinstance(analysis, dict) else "not_analyzed"
    thresholds = data.get("thresholds", {})
    row: dict[str, Any] = {
        "charge_snapshot": data.get("snapshot"),
        "charge_tolerance_e": thresholds.get("charge_tolerance_e"),
        "dipole_tolerance_e_per_A": thresholds.get("dipole_tolerance_e_per_A"),
        "tasker_type": data.get("tasker_type"),
        "polar": data.get("polar", False),
        "nominal_charge_e": data.get("nominal_charge_e"),
        "symmetric": data.get("symmetric", False),
    }
    groups = data.get("groups", {})
    for name in _CHARGE_GROUPS:
        group = groups.get(name, {})
        for field in _CHARGE_NUMERIC_FIELDS:
            row[f"{name}_{field}"] = group.get(field)
        row[f"{name}_neutrality"] = group.get("neutrality", status)
        row[f"{name}_polarity"] = group.get("polarity", status)
        row[f"{name}_sources"] = "; ".join(group.get("sources", []))
        row[f"{name}_reasons"] = "; ".join(group.get("reasons", []))
    return row


def summarize(candidates: list[Any]) -> list[dict[str, Any]]:
    """Flatten Candidates, retaining the original columns before charge fields."""
    rows: list[dict[str, Any]] = []
    for c in candidates:
        rows.append(
            {
                "interface_id": c.interface_id,
                "substrate": c.substrate_formula,
                "film": c.film_formula,
                "sub_miller": _fmt_miller(c.substrate_miller),
                "film_miller": _fmt_miller(c.film_miller),
                "sub_termination": c.substrate_termination,
                "film_termination": c.film_termination,
                "strain_sub": round(c.strain_substrate, 4),
                "strain_film": round(c.strain_film, 4),
                "mean_strain": round(c.mean_strain, 4),
                "area_A2": round(c.area, 2),
                "n_substrate": c.n_substrate,
                "n_film": c.n_film,
                "n_total": c.total_atoms,
                **charge_summary(c.metadata.get("charge_analysis")),
            }
        )
    return rows


def to_markdown(candidates: list[Any]) -> str:
    rows = summarize(candidates)
    if not rows:
        return "*No candidates passed the filters.*\n"

    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(_markdown_cell(r[h]) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def to_csv(candidates: list[Any], path: str) -> None:
    import csv

    rows = summarize(candidates)
    fieldnames = list(rows[0]) if rows else [*_SUMMARY_FIELDS, *charge_summary(None)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r\n", "<br>")
        .replace("\r", "<br>")
        .replace("\n", "<br>")
    )


def _fmt_miller(m: Any) -> str:
    return "(" + " ".join(str(int(x)) for x in m) + ")"
