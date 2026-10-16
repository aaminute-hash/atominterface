"""Export CZM parameters to FEM inputs (MOOSE / FEniCSx-ready text)."""

from __future__ import annotations

from pathlib import Path


def export_czm_moose(
    czm_params: dict,
    *,
    interface_id: str = "interface0",
    out_path: str | Path = "interface_czm.i",
) -> Path:
    """Emit a MOOSE input snippet with CohesiveZoneMasterMaterial params."""
    p = Path(out_path)
    sigma = float(czm_params.get("max_traction", 0.0))
    delta = float(czm_params.get("critical_separation", 0.0))
    Gc = float(czm_params.get("fracture_energy", 0.0))
    txt = f"""# AtomInterface-generated CZM parameters for {interface_id}
# sigma_max = {sigma:.6e}
# delta_c   = {delta:.6e}
# G_c       = {Gc:.6e}

[Materials]
  [czm]
    type = CohesiveZoneMasterMaterial
    interface_ids = {interface_id}
    max_traction = {sigma:.6e}
    critical_separation = {delta:.6e}
    fracture_energy = {Gc:.6e}
  []
[]
"""
    p.write_text(txt, encoding="utf-8")
    return p
