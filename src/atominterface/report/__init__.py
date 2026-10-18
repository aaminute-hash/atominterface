"""Report rendering: Markdown + CSV + metadata summary."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def candidates_to_markdown(candidates: Sequence[dict]) -> str:  # noqa: D401
    """Render a candidate table as a Markdown string."""
    if not candidates:
        return "_No candidates passed the filters._\n"
    header = "| id | termination | von_mises_strain | area (A^2) | natoms |\n"
    header += "|----|-------------|------------------|------------|--------|\n"
    rows = []
    for c in candidates:
        rows.append(
            f"| {c.get('id', '')} "
            f"| {c.get('termination', '')} "
            f"| {c.get('von_mises_strain', ''):.4f} "
            f"| {c.get('area_A2', ''):.2f} "
            f"| {c.get('natoms', '')} |"
        )
    return header + "\n".join(rows) + "\n"


def candidates_to_csv(candidates: Sequence[dict], path: str | Path) -> Path:  # noqa: D401
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        f.write("id,termination,von_mises_strain,area_A2,natoms\n")
        for c in candidates:
            f.write(
                f"{c.get('id', '')},{c.get('termination', '')},"
                f"{c.get('von_mises_strain', '')},{c.get('area_A2', ''):.3f},"
                f"{c.get('natoms', '')}\n"
            )
    return p


def relax_report(result, *, out_dir: str | Path = ".") -> dict:  # noqa: D401
    """Write a relaxation audit report (Markdown) + metadata.yaml."""
    from atominterface.provenance import RunMeta, write_metadata

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    md = [
        "# AtomInterface Relaxation Audit\n",
        f"- steps: {result.n_steps}",
        f"- converged: {result.converged}",
        f"- ood_triggered: {result.ood_triggered}",
        f"- max_ood_score: {result.max_ood_score:.3f}",
        f"- quarantine: {result.quarantine_path or 'none'}",
        "",
        "## Reasons",
        "",
    ]
    md += [f"- {r}" for r in result.reasons] or ["- (none)"]
    (d / "relax_audit.md").write_text("\n".join(md), encoding="utf-8")

    meta = RunMeta(
        backend="guarded",
        seed=None,
        extra={"n_steps": result.n_steps, "ood_triggered": result.ood_triggered},
    )
    write_metadata(meta, d / "metadata.yaml")
    return {"audit_md": str(d / "relax_audit.md"), "metadata": str(d / "metadata.yaml")}
