"""Candidate summary table (CSV / Markdown / list).

A dedicated export so users can eyeball the candidate set before
spending CPU on relaxation.  This is intentionally separate from
the full audit report: it is cheap and runs before relaxation.
"""

from __future__ import annotations

from typing import Any


def summarize(candidates: list[Any]) -> list[dict[str, Any]]:
    """Flatten a list of Candidates into a row-oriented table."""
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
        lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def to_csv(candidates: list[Any], path: str) -> None:
    import csv

    rows = summarize(candidates)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt_miller(m: Any) -> str:
    return "(" + " ".join(str(int(x)) for x in m) + ")"
