"""Builder class and candidate assembly for the coherent builder.

``CoherentInterfaceBuilder``, candidate conversion and the report helpers.
Depends on config, gates, assembly, match, cells, placement and the charge
facade; this is the root of the ``_coherent`` split.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from atominterface._coherent._util import _reduced_formula, _to_ase
from atominterface._coherent.assembly import (
    _assemble_interface,
    _make_slabs,
    _prepare_charge_structure,
    _prepare_stack,
)
from atominterface._coherent.cells import (
    _format_matrix,
    _lattice_area,
    _multiplicity,
    interface_areas,
)
from atominterface._coherent.config import CoherentInterfaceConfig
from atominterface._coherent.gates import GeometryGateError
from atominterface._coherent.match import _match_lattices, _MatchInfo
from atominterface._coherent.placement import (
    _frac_or_none,
    _registry_as_json,
    _registry_lateral_ang,
    _warn_unscanned_orientation,
    enumerate_registry_sites,
)
from atominterface.builders.base import BuilderConfig, Candidate, InterfaceBuilder
from atominterface.builders.charge import (
    SLAB_ID,
    analyze_charge_polarity,
    charge_rejection_reasons,
    tasker_classification,
)
from atominterface.builders.registry import register_builder
from atominterface.physics import EV_A2_TO_J_M2

if TYPE_CHECKING:
    from collections.abc import Sequence


def _warn_charge_diagnostics(analysis, config, sub_term_idx, film_term_idx):
    """Surface polar or charge-imbalanced terminations without touching them.

    ``symmetric_slabs="auto"`` must never transform a structure silently;
    the warning names the termination pair and the explicit opt-in instead.
    """
    groups = analysis["groups"]
    pair = f"{config.substrate_miller}:{sub_term_idx} / {config.film_miller}:{film_term_idx}"
    if any(group["polarity"] == "polar" for group in groups.values()):
        warnings.warn(
            f"termination pair {pair} is polar (Tasker Type 3) under the supplied charge "
            "model: an asymmetric board's normal dipole grows with thickness, so its surface "
            "energy diverges. Re-run with symmetric_slabs=True (CLI: --symmetric) for a "
            "mirror-symmetric board, or pair charge-compensating terminations.",
            stacklevel=2,
        )
        return
    charged = {
        name: group["net_charge_e"]
        for name, group in groups.items()
        if group["neutrality"] == "charged"
    }
    if charged:
        detail = ", ".join(f"{name} {charge:+.3f} e" for name, charge in charged.items())
        warnings.warn(
            f"termination pair {pair} carries net slab charges under the supplied charge "
            f"model ({detail}); pair charge-compensating terminations or relax the charge "
            "policy.",
            stacklevel=2,
        )


def _to_candidate(
    interface,
    substrate_slab,
    film_slab,
    match: _MatchInfo,
    sub_term_idx: int,
    film_term_idx: int,
    config: CoherentInterfaceConfig,
) -> Candidate:
    """Attach interface-specific metadata to a Candidate."""
    lattice = np.asarray(interface.lattice.matrix, dtype=np.float64)
    area = _lattice_area(lattice)

    properties = getattr(interface, "properties", None) or {}

    # Bilateral area.  ``area`` above is the stack's own cell, which is
    # what the rest of the pipeline keeps normalising by; this resolves
    # the two sides separately and records whether they agree.  An
    # unstrained film (legacy stacking) is a legitimate geometry, so a
    # mismatch is recorded, never raised.
    side_cells = properties.get("side_in_plane_cells")
    if isinstance(side_cells, dict) and "substrate" in side_cells and "film" in side_cells:
        substrate_lattice = np.asarray(side_cells["substrate"], dtype=np.float64)
        film_lattice = np.asarray(side_cells["film"], dtype=np.float64)
    else:
        substrate_lattice = film_lattice = lattice[:2, :]
    areas = interface_areas(substrate_lattice, film_lattice, policy=config.area_policy)

    # Energy placeholders; real values are filled by the relax layer.
    energy = float("nan")
    if hasattr(interface, "energy") and interface.energy is not None:
        energy = float(interface.energy)

    energy_per_area = energy / area if area > 0 and not math.isnan(energy) else 0.0

    # Unit conversion demonstration: eV/Å² -> J/m² is a recurring
    # footgun in interface thermodynamics.
    energy_per_area_jm2 = energy_per_area * EV_A2_TO_J_M2

    metadata = {
        "energy_per_area_ev_a2": energy_per_area,
        "energy_per_area_J_m2": energy_per_area_jm2,
        "builder": "coherent",
        "substrate_miller": config.substrate_miller,
        "film_miller": config.film_miller,
        "substrate_termination": sub_term_idx,
        "film_termination": film_term_idx,
        "substrate_supercell": match.substrate_supercell,
        "film_supercell": match.film_supercell,
        "substrate_supercell_multiplicity": _multiplicity(match.substrate_supercell),
        "film_supercell_multiplicity": _multiplicity(match.film_supercell),
        "match_area_ang2": match.match_area,
        "match_substrate_area_ang2": match.substrate_area,
        "match_film_area_ang2": match.film_area,
        "area_substrate": areas.area_substrate,
        "area_film": areas.area_film,
        "area_used": areas.area_used,
        "area_consistent": areas.consistent,
        "area_policy": areas.policy,
        "area_note": areas.note,
        "mismatch_stretch": match.strain,
        "mismatch_shear": match.shear_strain,
        "mismatch": max(match.strain, match.shear_strain),
        "stretch_strain": match.strain,
        "shear_strain": match.shear_strain,
        "geometry": "coherent-supercell",
        "gap_ang": config.gap,
        "vacuum_ang": config.vacuum_thickness,
        # Measured once during assembly (see ``_closest_cross_pair``) and
        # carried through so a downstream reader can tell two candidates
        # apart by the site they start on, not only by their energy.
        # Downstream consumers read this key; the four beside it add the
        # request and its result without touching it.
        "registry_lateral_offset_ang": properties.get("registry_lateral_offset"),
        # What the user asked for, what the request resolved to, and
        # where the join was actually built. Three separate entries
        # because they are three separate claims: the run that asked for
        # the trough and the run that asked for the ridge can share a
        # relaxation protocol and differ only here, and a candidate that
        # relaxed off its as-built site has to stay readable as one.
        "registry_requested": _registry_as_json(config.registry),
        "registry_target_frac": _frac_or_none(properties.get("registry_target_frac")),
        "registry_as_built_frac": _frac_or_none(
            (properties.get("registry_as_built") or {}).get("frac")
        ),
        "registry_as_built_lateral_A": _registry_lateral_ang(properties.get("registry_as_built")),
        **match.metadata,
        "rotation_matrix": None if match.rotation is None else match.rotation.tolist(),
    }

    # The substrate cell is the reference; the film is strained onto it
    # (see ``_assemble_interface``), so all of the mismatch lands on the
    # film and none on the substrate.  Reporting ``shear`` as the film
    # strain -- as an earlier version did -- made a 2.6 % isotropic
    # mismatch show up as "0.0258 / 0.0000" with no relation to which
    # side actually carries it.
    atoms = _to_ase(interface)
    if SLAB_ID not in atoms.arrays:
        raise ValueError("assembled interface is missing per-atom slab identities")
    tags = atoms.arrays[SLAB_ID]
    n_substrate = int(np.count_nonzero(tags == 0))
    n_film = int(np.count_nonzero(tags == 1))
    if not n_substrate or not n_film or n_substrate + n_film != len(atoms):
        raise ValueError("assembled interface must contain both slabs with valid identities")
    analysis = analyze_charge_polarity(
        atoms,
        charge_tolerance=config.charge_tolerance,
        dipole_tolerance=config.dipole_tolerance,
    )
    analysis["snapshot"] = "built_geometry"
    analysis["side_overridden_sources"] = interface.properties.get("charge_overridden_sources", {})
    metadata["charge_analysis"] = analysis
    metadata["charge_geometry"] = dict(atoms.info.get("charge_geometry", {}))

    # Tasker bookkeeping over the supplied charge model: 3 when any group is
    # polar or charged, 1 when every group is neutral and nonpolar, else
    # unknown.  The "auto" policy only warns; it never transforms.
    tasker_type = tasker_classification(analysis)
    metadata["tasker_type"] = tasker_type
    metadata["polar"] = any(g["polarity"] == "polar" for g in analysis["groups"].values())
    metadata["nominal_charge"] = analysis["groups"]["total"]["net_charge_e"]
    metadata["symmetric"] = config.symmetric_slabs is True
    analysis["tasker_type"] = tasker_type
    analysis["polar"] = metadata["polar"]
    analysis["nominal_charge_e"] = metadata["nominal_charge"]
    analysis["symmetric"] = metadata["symmetric"]
    if config.symmetric_slabs == "auto":
        _warn_charge_diagnostics(analysis, config, sub_term_idx, film_term_idx)

    sub_mat_str = _format_matrix(match.substrate_supercell)
    film_mat_str = _format_matrix(match.film_supercell)

    return Candidate(
        atoms=atoms,
        interface_id=(
            f"coherent:{config.substrate_miller}-{config.film_miller}"
            f":{sub_term_idx}-{film_term_idx}:sub{sub_mat_str}-film{film_mat_str}"
        ),
        builder_name="coherent",
        substrate_formula=_reduced_formula(substrate_slab),
        film_formula=_reduced_formula(film_slab),
        substrate_miller=config.substrate_miller,
        film_miller=config.film_miller,
        substrate_termination=str(sub_term_idx),
        film_termination=str(film_term_idx),
        strain_substrate=0.0,
        strain_film=match.strain,
        area=area,
        n_substrate=n_substrate,
        n_film=n_film,
        metadata=metadata,
        area_substrate=areas.area_substrate,
        area_film=areas.area_film,
        area_consistent=areas.consistent,
        area_policy=areas.policy,
    )


#: every lattice match the same way, and each match states the fractions in
#: its own cell -- so the messages differ while the cause does not.
_ASSEMBLY_DETAIL_LIMIT = 3
_ASSEMBLY_DETAIL_WIDTH = 240


def _note_assembly_reason(report: dict, exc: BaseException) -> None:
    """Keep one short line saying why an assembled candidate was dropped.

    "No interface could be assembled" is only half an answer: a registry
    request this surface cannot name -- ``hcp`` on a two-sublattice
    surface, say -- fails *every* candidate the same way, and without the
    reason the failure reads like a lattice match that went nowhere. The
    text is kept once per distinct message rather than once per candidate,
    because a hundred candidates that die identically have one cause.
    """
    reasons = report.setdefault("assembly_reasons", {})
    key = type(exc).__name__
    reasons[key] = reasons.get(key, 0) + 1
    detail = report.setdefault("assembly_detail", [])
    lines = str(exc).strip().splitlines()
    text = lines[0] if lines else key
    if text in detail:
        return
    if len(detail) >= _ASSEMBLY_DETAIL_LIMIT:
        # Appended once and only once: every later message takes this same
        # branch, and a summary per message would be as long as the list it
        # was meant to replace.
        if len(detail) == _ASSEMBLY_DETAIL_LIMIT:
            detail.append(
                "... and further attempt(s) that failed the same way; "
                "assembly_reasons holds the count."
            )
        return
    detail.append(
        text if len(text) <= _ASSEMBLY_DETAIL_WIDTH else text[: _ASSEMBLY_DETAIL_WIDTH - 1] + "…"
    )


@register_builder("coherent")
class CoherentInterfaceBuilder(InterfaceBuilder):
    """Coherent solid-solid interface builder.

    Examples
    --------
    >>> from atominterface.builders import CoherentInterfaceBuilder
    >>> from atominterface.builders.coherent import CoherentInterfaceConfig
    >>> builder = CoherentInterfaceBuilder(
    ...     substrate=substrate_structure,
    ...     film=film_structure,
    ...     config=CoherentInterfaceConfig(
    ...         substrate_miller=(0, 0, 1),
    ...         film_miller=(0, 0, 1),
    ...         max_strain=0.05,
    ...     ),
    ... )
    >>> candidates = builder.build_all()
    """

    name: str = "coherent"

    def __init__(self, substrate, film, config: CoherentInterfaceConfig | None = None):
        if config is None:
            config = CoherentInterfaceConfig()
        self.config = config
        self.substrate = substrate
        self.film = film
        self.last_build_report: dict = {}

    def validate(self) -> list[str]:
        """Validate the input pair before building."""
        errors = self.config.validate()

        if self.substrate is None or self.film is None:
            errors.append("substrate and film structures must be provided")
            return errors

        sub_lat = np.asarray(self.substrate.lattice.matrix, dtype=np.float64)
        film_lat = np.asarray(self.film.lattice.matrix, dtype=np.float64)

        if np.linalg.det(sub_lat) <= 0:
            errors.append("substrate lattice has non-positive volume")
        if np.linalg.det(film_lat) <= 0:
            errors.append("film lattice has non-positive volume")

        if not math.isfinite(self.config.max_area) or self.config.max_area <= 0:
            errors.append("max_area must be finite and positive")

        if any(
            miller == (0, 0, 0)
            for miller in (self.config.substrate_miller, self.config.film_miller)
        ):
            errors.append("Miller indices must be non-zero")

        return errors

    def _prepared_slabs(self):
        substrate = _prepare_charge_structure(
            self.substrate, self.config.substrate_charges, self.config.substrate_oxidation
        )
        film = _prepare_charge_structure(
            self.film, self.config.film_charges, self.config.film_oxidation
        )
        return _make_slabs(substrate, film, self.config)

    def list_terminations(self) -> dict[str, list[tuple[int, str]]]:
        """Enumerate terminations for both sides."""
        sub_terms, film_terms = self._prepared_slabs()

        def _describe(idx_structure_pairs):
            out = []
            for idx, slab in idx_structure_pairs:
                out.append((idx, _reduced_formula(slab)))
            return out

        return {
            "substrate": _describe(sub_terms),
            "film": _describe(film_terms),
        }

    def build(self, config: BuilderConfig | None = None) -> list[Candidate]:
        """Generate all strain-acceptable interface candidates."""
        return self.build_all()

    def _match(self, strain_limit=None, slabs=None) -> list[_MatchInfo]:
        """Match the two slabs' in-plane cells.

        Matching must happen on the **slab** cells, not on the bulk
        cells: ``SlabGenerator`` re-orients the cell so the surface
        normal is the third axis, and for any surface other than (001)
        the bulk's first two vectors are not even in the interface
        plane.  All terminations of one slab share the same lateral
        cell, so a single match covers every termination pair.
        """
        sub_terms, film_terms = self._prepared_slabs() if slabs is None else slabs
        # The match helper cannot see the Miller indices; the builder can,
        # so it labels the pair for the ZSL-fallback warning.
        label = (
            f"substrate {tuple(int(i) for i in self.config.substrate_miller)} / "
            f"film {tuple(int(i) for i in self.config.film_miller)}"
        )
        return _match_lattices(
            sub_terms[0][1].lattice.matrix,
            film_terms[0][1].lattice.matrix,
            self.config,
            strain_limit=strain_limit,
            label=label,
        )

    def preview_matches(self, limit: int = 5) -> list[_MatchInfo]:
        """Best lattice matches **before** strain filtering.

        When a tolerance rejects every candidate the useful question is
        "how close can this pair actually get?".  This method answers it
        without generating a single structure, so a CLI can suggest a
        concrete ``--max-strain`` instead of a dead end.
        """
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))

        return self._match(strain_limit=math.inf)[: max(limit, 1)]

    def _reset_build_report(self) -> None:
        self.last_build_report = {
            "charge_policy": self.config.charge_policy,
            "thresholds": {
                "charge_tolerance_e": self.config.charge_tolerance,
                "dipole_tolerance_e_per_A": self.config.dipole_tolerance,
            },
            "attempted": 0,
            "assembled": 0,
            "assembly_failed": 0,
            "analyzed": 0,
            "kept": 0,
            "rejected": 0,
            "rejection_reasons": {},
            "stopped_at_limit": False,
            "empty_stage": None,
        }

    def _accept_charge(self, candidate: Candidate) -> bool:
        report = self.last_build_report
        report["analyzed"] += 1
        analysis = candidate.metadata["charge_analysis"]
        reasons = charge_rejection_reasons(analysis, self.config.charge_policy)
        analysis["filter"] = {
            "policy": self.config.charge_policy,
            "accepted": not reasons,
            "reasons": reasons,
        }
        if not reasons:
            report["kept"] += 1
            return True
        report["rejected"] += 1
        for reason in reasons:
            counts = report["rejection_reasons"]
            counts[reason] = counts.get(reason, 0) + 1
        return False

    def build_all(self) -> list[Candidate]:
        """Analyze geometric candidates and retain those passing the requested policy."""
        self._reset_build_report()
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))

        slabs = self._prepared_slabs()
        matches = self._match(slabs=slabs)
        if not matches:
            self.last_build_report["empty_stage"] = "no_matches"
            return []

        candidates = self._collect_candidates(matches, slabs)

        if not candidates:
            self.last_build_report["empty_stage"] = (
                "charge_filter" if self.last_build_report["rejected"] else "assembly"
            )
        # Once per build, and only after assembly: the message quotes the
        # registry that was actually built, which does not exist yet when
        # the config is read.  ``None`` keeps the advisory useful even when
        # nothing assembled -- the orientation is still unscanned.
        _warn_unscanned_orientation(
            self.config,
            self._built_lateral_offset(candidates),
            getattr(self.config, "registry", None),
        )
        return candidates

    def _collect_candidates(self, matches: list[_MatchInfo], slabs) -> list[Candidate]:
        """Assemble every accepted candidate up to the configured limit.

        Separate from ``build_all`` so that hitting ``max_candidates``
        cannot return before the per-build advisories are emitted: the
        orientation warning carries a number measured during assembly, and
        an early ``return`` here would silently drop it on exactly the
        runs that build the most candidates.
        """
        candidates: list[Candidate] = []
        for match in matches:
            for sub_idx, sub_slab in slabs[0]:
                for film_idx, film_slab in slabs[1]:
                    if self.config.registry == "scan":
                        sub_replicated, _, sub_2d, _, _ = _prepare_stack(
                            sub_slab, film_slab, match, self.config.max_strain
                        )
                        sub_coords = np.asarray(sub_replicated.cart_coords, dtype=np.float64).copy()
                        sites = enumerate_registry_sites(sub_coords, sub_2d)
                        if self.config.max_registry_sites is not None:
                            sites = sites[: self.config.max_registry_sites]
                        for site in sites:
                            candidate = self._build_candidate(
                                match, sub_idx, sub_slab, film_idx, film_slab, registry=site
                            )
                            if candidate is not None and self._accept_charge(candidate):
                                candidates.append(candidate)
                            if len(candidates) >= self.config.max_candidates:
                                self.last_build_report["stopped_at_limit"] = True
                                return candidates
                    else:
                        candidate = self._build_candidate(
                            match, sub_idx, sub_slab, film_idx, film_slab
                        )
                        if candidate is not None and self._accept_charge(candidate):
                            candidates.append(candidate)
                        if len(candidates) >= self.config.max_candidates:
                            self.last_build_report["stopped_at_limit"] = True
                            return candidates
        return candidates

    @staticmethod
    def _built_lateral_offset(candidates: Sequence[Candidate]) -> float | None:
        """The as-built registry of the first assembled candidate, in A.

        Handing the warning a number instead of a description is the whole
        point: two non-(001) runs are only comparable if they started on
        the same site, and that is a distance the caller cannot see from
        anything else in the output.
        """
        for candidate in candidates:
            lateral = candidate.metadata.get("registry_lateral_offset_ang")
            if isinstance(lateral, (int, float)):
                return float(lateral)
        return None

    def _build_candidate(self, match, sub_idx, sub_slab, film_idx, film_slab, registry=None):
        report = self.last_build_report
        report["attempted"] += 1
        # Validate charge provenance outside the geometry error handler.
        for slab in (sub_slab, film_slab):
            _prepare_charge_structure(slab)
        try:
            interface = _assemble_interface(
                sub_slab, film_slab, match, self.config, registry=registry
            )
        except GeometryGateError:
            # A gate that is swallowed is not a gate.  Any other assembly
            # problem is specific to one candidate and is worth dropping
            # quietly; a geometry violation is a property of the *boards*,
            # so every remaining candidate would fail the same way and the
            # honest answer is to say so rather than to return nothing.
            report["assembly_failed"] += 1
            raise
        except Exception as exc:  # noqa: BLE001
            report["assembly_failed"] += 1
            _note_assembly_reason(report, exc)
            return None
        report["assembled"] += 1
        return _to_candidate(interface, sub_slab, film_slab, match, sub_idx, film_idx, self.config)

    def export_candidates(
        self,
        candidates: Sequence[Candidate],
        out_dir: str = "interface_candidates",
        formats: Sequence[str] = ("json", "csv", "md"),
    ) -> list[Path]:
        """Persist a candidate list for downstream relaxation."""
        from atominterface.builders.candidates import (  # noqa: WPS433
            to_csv,
            to_markdown,
        )

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        rows = [
            {
                "interface_id": c.interface_id,
                "substrate": c.substrate_formula,
                "film": c.film_formula,
                "sub_miller": c.substrate_miller,
                "film_miller": c.film_miller,
                "sub_termination": c.substrate_termination,
                "film_termination": c.film_termination,
                "strain_sub": c.strain_substrate,
                "strain_film": c.strain_film,
                "area_A2": c.area,
                "n_total": c.total_atoms,
                **c.metadata,
            }
            for c in candidates
        ]

        written: list[Path] = []
        if "json" in formats:
            import json

            path = out / "candidates.json"
            with open(path, "w") as f:
                json.dump(rows, f, indent=2, default=str)
            written.append(path)

        if "csv" in formats and rows:
            path = out / "candidates.csv"
            to_csv(list(candidates), str(path))
            written.append(path)

        if "md" in formats:
            path = out / "candidates.md"
            md = to_markdown(list(candidates))
            path.write_text(md)
            written.append(path)

        return written
