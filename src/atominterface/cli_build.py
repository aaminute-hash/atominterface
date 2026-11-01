"""``atominterface build`` -- generate interface candidates from bulk structures.

The command is a thin front-end over :class:`~atominterface.builders.base.InterfaceBuilder`:
it loads two bulk structures, picks the requested builder from the registry,
runs the enumeration and writes the surviving candidates plus CSV/JSON reports.

``pymatgen`` is only imported when a structure actually has to be read, so
``atominterface build --help`` keeps working on a slim installation.  The
same laziness applies to calculators: without ``--relax`` no backend is
imported beyond its (dependency-light) registration module, and no model
weights are ever loaded.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003  (runtime: typer resolves build() annotations via get_type_hints)

import typer

import atominterface.builders  # noqa: F401  (import side effect: registers built-in builders)
from atominterface._cli_build.params import (
    _CHARGE_OPTIONS,
    _CHARGE_POLICIES,
    _COHERENT_DEFAULTS,
    _CSV_FIELDS,
    _DEFAULT_OUTDIR,
    _DEVICES,
    _RELAX_FIELDS,
    _RUN_CONFIG_KEYS,
    _charge_policy,
    _check_charge_support,
    _finite_nonnegative,
    _finite_positive,
    _load_config_overrides,
    _load_structure,
    _make_config,
    _merged_registry,
    _optional_finite_positive,
    _parse_backend_list,
    _parse_oxidation,
    _registry,
    _resolve_backend,
    _resolve_builder,
    _resolve_run_params,
    _validate_device,
)
from atominterface._cli_build.relax import _ood_columns, _relax_candidates
from atominterface._cli_build.report import _print_candidates, _report_no_candidates, _write_outputs
from atominterface.run_config import save_run_config

app = typer.Typer(
    help="Build interface candidates from two bulk structures.",
    no_args_is_help=True,
)


@app.command("build")
def build(
    ctx: typer.Context,
    substrate: Path = typer.Argument(..., help="Substrate bulk structure (CIF/POSCAR)."),
    film: Path = typer.Argument(..., help="Film bulk structure (CIF/POSCAR)."),
    builder: str = typer.Option("coherent", "--builder", "-b", help="Registered builder name."),
    sub_miller: tuple[int, int, int] = typer.Option(
        (0, 0, 1), "--sub-miller", help="Substrate surface Miller indices."
    ),
    film_miller: tuple[int, int, int] = typer.Option(
        (0, 0, 1), "--film-miller", help="Film surface Miller indices."
    ),
    max_strain: float = typer.Option(0.05, "--max-strain", help="Absolute strain tolerance."),
    max_area: float = typer.Option(
        100.0,
        "--max-area",
        help="Maximum interface area in A^2. Cost grows fast with this value.",
    ),
    gap: float = typer.Option(2.0, "--gap", help="Separation gap between slabs in A."),
    min_thickness: float = typer.Option(
        _COHERENT_DEFAULTS.min_thickness,
        "--min-thickness",
        callback=_finite_nonnegative,
        help="Minimum slab thickness in A, applied to both sides unless overridden per side.",
    ),
    min_thickness_substrate: float | None = typer.Option(
        None,
        "--min-thickness-substrate",
        callback=_optional_finite_positive,
        help="Override --min-thickness for the substrate side (A, must be > 0).",
    ),
    min_thickness_film: float | None = typer.Option(
        None,
        "--min-thickness-film",
        callback=_optional_finite_positive,
        help="Override --min-thickness for the film side (A, must be > 0).",
    ),
    vacuum: float = typer.Option(
        _COHERENT_DEFAULTS.vacuum_thickness,
        "--vacuum",
        callback=_finite_nonnegative,
        help="Fixed vacuum padding normal to the interface in A; never scales with thickness.",
    ),
    max_candidates: int = typer.Option(10, "--max-candidates", help="Keep at most N candidates."),
    backend: str = typer.Option(
        "dummy",
        "--backend",
        help="Calculator backend for --relax: dummy, mace or chgnet.",
    ),
    committee_backend: list[str] = typer.Option(
        [],
        "--committee-backend",
        help=(
            "Extra calculator backend(s) mounted as the OOD committee with "
            "--relax; repeatable or comma-separated. Empty (the default) "
            "runs without a committee."
        ),
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        callback=_validate_device,
        help="Compute device for --relax: cpu or cuda.",
    ),
    relax: bool = typer.Option(
        False,
        "--relax",
        help="Guarded-relax each candidate with the selected backend.",
    ),
    fmax: float = typer.Option(
        0.05,
        "--fmax",
        callback=_finite_nonnegative,
        help="Relaxation force tolerance in eV/A (with --relax).",
    ),
    max_steps: int = typer.Option(
        200, "--max-steps", help="Maximum relaxation steps per candidate (with --relax)."
    ),
    ood_threshold: float = typer.Option(
        0.8,
        "--ood-threshold",
        callback=_finite_nonnegative,
        help="OOD score that stops a relaxation early (with --relax).",
    ),
    charge_policy: str = typer.Option(
        "report",
        "--charge-policy",
        callback=_charge_policy,
        help="Charge filter: report, require-neutral, or require-nonpolar; unknown fails strict filters.",
    ),
    substrate_oxidation: list[str] = typer.Option(
        [], "--substrate-oxidation", help="Repeat Element=value mappings, e.g. Na=1 or Cl=-1."
    ),
    film_oxidation: list[str] = typer.Option(
        [], "--film-oxidation", help="Repeat Element=value mappings, e.g. Na=1 or Cl=-1."
    ),
    registry: str = typer.Option(
        "auto",
        "--registry",
        callback=_registry,
        help=(
            "Lateral registry of the join: auto, on-top, hcp, an explicit u,v "
            "offset (e.g. 0.5,0.5), or none for the pre-0.5.0 behaviour (the join "
            'stays where the two slab cuts leave it). Names a site; use "scan" to '
            "enumerate every distinguishable site."
        ),
    ),
    max_registry_sites: int | None = typer.Option(
        None,
        "--max-registry-sites",
        help=(
            "With --registry scan, keep at most N sites (and thus at most N "
            "candidates per match/termination). Default: all distinguishable sites."
        ),
    ),
    charge_tolerance: float = typer.Option(
        1e-6,
        "--charge-tolerance",
        callback=_finite_nonnegative,
        help="Non-negative net-charge tolerance in e.",
    ),
    dipole_tolerance: float = typer.Option(
        1e-3,
        "--dipole-tolerance",
        callback=_finite_nonnegative,
        help="Non-negative normal dipole/area tolerance in e/A.",
    ),
    symmetric: bool | None = typer.Option(
        None,
        "--symmetric/--no-symmetric",
        # Own panel: the paired flag names are wider than the main option
        # column, and rich would otherwise shrink that column until names
        # like --substrate-oxidation truncate on an 80-column terminal.
        rich_help_panel="Symmetric slabs",
        help=(
            "Force mirror-symmetric slabs (same termination on both "
            "surfaces); default only warns about polar terminations."
        ),
    ),
    outdir: Path = typer.Option(_DEFAULT_OUTDIR, "--outdir", "-o", help="Output directory."),
    fmt: str = typer.Option("extxyz", "--format", help="Structure file format for candidates."),
    config_file: Path | None = typer.Option(
        None,
        "--config",
        help=(
            "YAML run configuration to replay (for example a run_config.yaml "
            "written by a previous run). Explicit command-line flags override it."
        ),
    ),
) -> None:
    """Enumerate interface candidates for a substrate/film pair.

    Every run writes its effective parameters to ``run_config.yaml`` in
    the output directory; passing that file back through ``--config``
    replays the run exactly, with explicit flags overriding single keys.
    """
    params = _resolve_run_params(
        ctx,
        _load_config_overrides(config_file),
        {
            "builder": builder,
            "sub_miller": sub_miller,
            "film_miller": film_miller,
            "max_strain": max_strain,
            "max_area": max_area,
            "gap": gap,
            "min_thickness": min_thickness,
            "min_thickness_substrate": min_thickness_substrate,
            "min_thickness_film": min_thickness_film,
            "vacuum": vacuum,
            "max_candidates": max_candidates,
            "backend": backend,
            "committee_backend": committee_backend,
            "device": device,
            "relax": relax,
            "fmax": fmax,
            "max_steps": max_steps,
            "ood_threshold": ood_threshold,
            "charge_policy": charge_policy,
            "substrate_oxidation": substrate_oxidation,
            "film_oxidation": film_oxidation,
            "charge_tolerance": charge_tolerance,
            "dipole_tolerance": dipole_tolerance,
            "symmetric": symmetric,
            "registry": registry,
            "max_registry_sites": max_registry_sites,
        },
    )

    builder_cls = _resolve_builder(params["builder"])
    _check_charge_support(ctx, params["builder"], params["symmetric"])
    _resolve_backend(params["backend"])
    for name in params["committee_backend"]:
        _resolve_backend(name)
    config = _make_config(
        params["builder"],
        substrate_miller=tuple(params["sub_miller"]),
        film_miller=tuple(params["film_miller"]),
        max_strain=params["max_strain"],
        max_area=params["max_area"],
        gap=params["gap"],
        min_thickness=params["min_thickness"],
        min_thickness_substrate=params["min_thickness_substrate"],
        min_thickness_film=params["min_thickness_film"],
        vacuum_thickness=params["vacuum"],
        max_candidates=params["max_candidates"],
        charge_policy=params["charge_policy"],
        substrate_oxidation=_parse_oxidation(
            params["substrate_oxidation"], "--substrate-oxidation"
        ),
        film_oxidation=_parse_oxidation(params["film_oxidation"], "--film-oxidation"),
        charge_tolerance=params["charge_tolerance"],
        dipole_tolerance=params["dipole_tolerance"],
        symmetric_slabs="auto" if params["symmetric"] is None else params["symmetric"],
        registry=params["registry"],
        max_registry_sites=params["max_registry_sites"],
    )

    sub = _load_structure(substrate)
    film_struct = _load_structure(film)

    typer.echo(f"builder   : {params['builder']}")
    typer.echo(f"substrate : {substrate} ({sub.composition.reduced_formula})")
    typer.echo(f"film      : {film} ({film_struct.composition.reduced_formula})")
    typer.echo(f"surfaces  : {tuple(params['sub_miller'])} | {tuple(params['film_miller'])}")

    instance = builder_cls(substrate=sub, film=film_struct, config=config)
    candidates = instance.build()
    report = getattr(instance, "last_build_report", None)
    relax_results = None
    if params["relax"]:
        relax_results = _relax_candidates(
            candidates,
            backend=params["backend"],
            committee_backends=params["committee_backend"],
            device=params["device"],
            fmax=params["fmax"],
            max_steps=params["max_steps"],
            ood_threshold=params["ood_threshold"],
            outdir=outdir,
        )
    csv_path = _write_outputs(candidates, outdir, fmt, report, relax_results)

    # Record the *effective* parameters (config file merged with explicit
    # flags) so the run can be replayed later with ``--config``.
    save_run_config(params, outdir / "run_config.yaml")

    if not candidates:
        _report_no_candidates(instance, params["max_strain"], params["max_area"])
    else:
        _print_candidates(candidates)
    if isinstance(report, dict) and report.get("stopped_at_limit"):
        typer.echo(
            "Enumeration stopped at --max-candidates; the report covers attempted candidates."
        )
    typer.echo(
        f"\nWrote {len(candidates)} structure file(s), {csv_path} and JSON reports to {outdir}/"
    )
    invalid = [rank for rank, result in (relax_results or {}).items() if not result.valid]
    if invalid:
        typer.echo(
            "relaxation failed validity checks for candidate(s): "
            + ", ".join(str(rank) for rank in invalid),
            err=True,
        )
        raise typer.Exit(code=2)


__all__ = [
    "app",
    "build",
    "_CHARGE_OPTIONS",
    "_CHARGE_POLICIES",
    "_COHERENT_DEFAULTS",
    "_CSV_FIELDS",
    "_DEFAULT_OUTDIR",
    "_DEVICES",
    "_RELAX_FIELDS",
    "_RUN_CONFIG_KEYS",
    "_charge_policy",
    "_check_charge_support",
    "_finite_nonnegative",
    "_finite_positive",
    "_load_config_overrides",
    "_load_structure",
    "_make_config",
    "_merged_registry",
    "_optional_finite_positive",
    "_parse_backend_list",
    "_parse_oxidation",
    "_registry",
    "_resolve_backend",
    "_resolve_builder",
    "_resolve_run_params",
    "_validate_device",
    "_ood_columns",
    "_relax_candidates",
    "_print_candidates",
    "_report_no_candidates",
    "_write_outputs",
]

if __name__ == "__main__":
    app()
