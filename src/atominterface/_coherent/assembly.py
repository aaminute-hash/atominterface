"""Slab cutting and interface assembly for the coherent builder.

Cutting, termination enumeration, stacking and the reference-slab helper.
Depends on config, gates, match, cells, placement and the charge facade.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from atominterface._coherent._util import _to_ase
from atominterface._coherent.cells import _scale_in_plane, _strain_film_onto
from atominterface._coherent.gates import _check_area_conservation, _check_inplane_period
from atominterface._coherent.placement import (
    _closest_cross_pair,
    _place_join,
    _record_registry_properties,
    _warn_interface_proximity,
)
from atominterface.builders.charge import (
    ANALYSIS_CHARGE,
    CHARGE_KNOWN,
    CHARGE_SOURCE,
    SLAB_ID,
    _orient_slab,
    _site_dependent_charge,
    make_symmetric_slab,
    resolve_charge_data,
    termination_cuts,
)

if TYPE_CHECKING:
    from typing import Any

    from atominterface._coherent.config import CoherentInterfaceConfig
    from atominterface._coherent.match import _MatchInfo


def _prepare_charge_structure(structure, charges=None, oxidation_states=None):
    """Resolve on a bulk copy, before cutting or replicating any sites."""
    if not structure.is_ordered:
        raise ValueError("charge analysis requires ordered structures")
    properties = structure.site_properties
    stored = None
    if ANALYSIS_CHARGE in properties:
        required = (ANALYSIS_CHARGE, CHARGE_KNOWN, CHARGE_SOURCE)
        if not all(key in properties for key in required):
            raise ValueError("incomplete stored analysis charge properties")
        stored = dict(
            zip(("values", "known", "sources"), (properties[k] for k in required), strict=True)
        )
    oxidation = [getattr(site.specie, "oxi_state", None) for site in structure]
    data = resolve_charge_data(
        [site.specie.symbol for site in structure],
        charges=charges,
        oxidation_states=oxidation_states,
        stored=stored,
        initial_charges=properties.get("charge"),
        oxi_states=oxidation if any(value is not None for value in oxidation) else None,
    )
    prepared = structure.copy()
    prepared.properties.update(structure.properties)
    for key, name in (
        (ANALYSIS_CHARGE, "values"),
        (CHARGE_KNOWN, "known"),
        (CHARGE_SOURCE, "sources"),
    ):
        prepared.add_site_property(key, data[name].tolist())
    prepared.properties["charge_overridden_sources"] = data["overridden_sources"]
    return prepared


def _get_terminations(structure, miller, min_thickness, vacuum_thickness):
    """Return numbered slabs for one surface.

    The bulk is never handed back as a slab: a surface whose termination
    cuts all fail raises :class:`ValueError` naming the Miller indices, so
    a failed cut can never masquerade as a finite interface slab.
    """
    from pymatgen.core.surface import SlabGenerator

    preserve_sites = _site_dependent_charge(structure)
    try:
        generator = SlabGenerator(
            structure,
            miller_index=miller,
            min_slab_size=min_thickness,
            min_vacuum_size=vacuum_thickness,
            in_unit_planes=False,
            center_slab=True,
            primitive=not preserve_sites,
        )
        slabs = termination_cuts(generator) if preserve_sites else generator.get_slabs()
    except Exception as exc:
        raise ValueError(f"slab generation failed for Miller indices {miller}: {exc}") from exc

    out = []
    for idx, generated in enumerate(slabs):
        if generated is None:
            continue
        slab = _orient_slab(generated)
        slab.properties["ai_slab_generated"] = True
        slab.properties["charge_overridden_sources"] = structure.properties.get(
            "charge_overridden_sources", []
        )
        out.append((idx, slab))
    if not out:
        raise ValueError(f"no termination slab could be generated for Miller indices {miller}")
    return out


def _flatten_terminations(raw_terms):
    """Normalize ``(index, structure)`` pairs from the termination enumeration."""
    terms = []
    for item in raw_terms:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            idx, slab = item
            if isinstance(slab, (list, tuple)):
                slab = slab[0]  # unpack nested (slab, metadata)
            terms.append((idx, slab))
        else:
            terms.append((0, item))
    return terms


def _make_slabs(substrate, film, config: CoherentInterfaceConfig):
    """Build substrate / film slabs for every termination.

    Returns two lists of ``(termination_index, structure)``.  When
    ``symmetric_slabs`` is forced, every termination becomes a
    mirror-symmetric board (same termination on both surfaces) whose
    finished thickness respects ``min_thickness``.

    Each side is cut at its own resolved thickness
    (:meth:`CoherentInterfaceConfig.slab_thicknesses`), so a thicker
    substrate no longer forces an equally thick film.  The vacuum padding
    is a fixed absolute amount shared by both sides and never scales
    with either thickness.
    """
    symmetric = config.symmetric_slabs
    sub_thickness, film_thickness = config.slab_thicknesses()
    sides = (
        (substrate, config.substrate_miller, sub_thickness),
        (film, config.film_miller, film_thickness),
    )
    if symmetric is True:
        terms = [
            _symmetric_terminations(bulk, miller, config, thickness)
            for bulk, miller, thickness in sides
        ]
    else:
        terms = [
            _flatten_terminations(
                _get_terminations(bulk, miller, thickness, config.vacuum_thickness)
            )
            for bulk, miller, thickness in sides
        ]
    if not config.relax_terminations:
        terms = [side[:1] for side in terms]
    return terms[0], terms[1]


def _symmetric_terminations(bulk, miller, config: CoherentInterfaceConfig, min_thickness: float):
    """Mirror-symmetric boards for every termination of one surface.

    Routed through :func:`make_symmetric_slab` rather than mirroring the
    enumerated half-cut boards directly: pymatgen counts
    ``min_slab_size`` in whole atomic planes, so a half-cut board mirrored
    about its top plane lands roughly ``2 * d_hkl`` short of the requested
    thickness (at the default ``min_thickness`` a Cu(001) board would come
    out 3.6 A instead of 6.0 A).  The bounded thickening loop inside
    :func:`make_symmetric_slab` is what makes the finished board respect
    ``min_thickness`` -- the per-side resolved value passed in by
    :func:`_make_slabs`, never the shared field directly.
    """
    n_cuts = len(
        _flatten_terminations(
            _get_terminations(bulk, miller, min_thickness / 2.0, config.vacuum_thickness)
        )
    )
    return [
        (
            idx,
            make_symmetric_slab(
                bulk,
                miller,
                termination=idx,
                min_thickness=min_thickness,
                vacuum=config.vacuum_thickness,
            ).structure,
        )
        for idx in range(n_cuts)
    ]


def _prepare_stack(substrate_slab, film_slab, match, max_strain):
    """Replicate both boards, gate them, and strain the film onto the substrate.

    Steps 1 and 2 of :func:`_assemble_interface`, plus the geometry gates
    that have to bracket them: criterion B belongs right after the
    replication (which must conserve area) and criterion C right after
    the strain (which must not).  Returns ``(substrate_slab, film_slab,
    sub_2d, sub_height, d_refs)`` -- the boards to stack, the substrate's
    in-plane block and vacuum column, and the contact scale each side
    measured.
    """
    # Criterion B runs on the *replicated* board, before the coherent
    # strain: replication must conserve area and straining must not, so
    # folding the two into one comparison would fire on every match.
    sub_before = np.asarray(substrate_slab.lattice.matrix, dtype=np.float64)
    film_before = np.asarray(film_slab.lattice.matrix, dtype=np.float64)
    if match is not None:
        substrate_slab = _scale_in_plane(substrate_slab, match.substrate_supercell)
        film_slab = _scale_in_plane(film_slab, match.film_supercell)
    _check_area_conservation(
        "substrate", sub_before, substrate_slab, match.substrate_supercell if match else None
    )
    _check_area_conservation(
        "film", film_before, film_slab, match.film_supercell if match else None
    )

    sub_mat = np.asarray(substrate_slab.lattice.matrix, dtype=np.float64)
    # In-plane block: exact, since ``_orient_slab`` guarantees a and b carry
    # no z component and the vacuum column no in-plane one.
    sub_2d = np.asarray(sub_mat[:2, :2], dtype=np.float64)
    film_2d = np.asarray(film_slab.lattice.matrix, dtype=np.float64)[:2, :2]

    # Strain the film onto the substrate cell.  The shared helper keeps
    # the fractional coordinates -- i.e. the stacking sequence -- intact
    # and is the very same call ``reference_slabs`` performs, so a
    # reference board and the interface side it references are strained
    # by literally the same code.
    coherent = (
        match is not None
        and abs(float(np.linalg.det(sub_2d))) > 1e-9
        and abs(float(np.linalg.det(film_2d))) > 1e-9
    )
    if coherent:
        film_slab = _strain_film_onto(film_slab, sub_2d)
    # Criterion C runs on the cells the stack will actually repeat, so
    # after the film has been strained onto the substrate.
    d_refs = (
        _check_inplane_period("substrate", substrate_slab, max_strain),
        _check_inplane_period("film", film_slab, max_strain),
    )
    return substrate_slab, film_slab, sub_2d, float(sub_mat[2, 2]), d_refs


def _assemble_interface(substrate_slab, film_slab, match=None, config=None, **kwargs):
    """Stack substrate and film slabs into one coherent interface.

    pymatgen's ``CoherentInterfaceBuilder`` triggers a full ZSL
    enumeration in its ``__init__`` (extremely slow, ~100 s for a simple
    Cu cell), so the stacking is done here instead.  The steps are:

    1. replicate each slab by its integer in-plane supercell,
    2. strain the film in-plane so its cell matches the substrate cell
       exactly -- this is what makes the interface coherent, and it is
       the geometry described by ``match.strain`` / ``match.shear_strain``,
    3. drop the film on top with ``config.gap`` and pad ``config.vacuum_thickness``.

    Without ``match`` the slabs are stacked as given (legacy behaviour,
    used by hand-built candidates and by tests).
    """
    from pymatgen.core import Structure
    from pymatgen.core.lattice import Lattice as _Lattice

    # Defensive unpack: some callers pass tuple-wrapped slabs.
    if isinstance(substrate_slab, (tuple, list)):
        substrate_slab = substrate_slab[0]
    if isinstance(film_slab, (tuple, list)):
        film_slab = film_slab[0]

    if not hasattr(substrate_slab, "sites") or not hasattr(film_slab, "sites"):
        raise TypeError(
            "Expected Structure-like slabs with .sites, got "
            f"sub={type(substrate_slab).__name__}, film={type(film_slab).__name__}"
        )

    gap = 2.0
    vacuum = 10.0
    max_strain = 0.05
    if config is not None:
        gap = float(getattr(config, "gap", gap))
        vacuum = float(getattr(config, "vacuum_thickness", vacuum))
        max_strain = float(getattr(config, "max_strain", max_strain))

    if ANALYSIS_CHARGE not in substrate_slab.site_properties:
        substrate_slab = _prepare_charge_structure(substrate_slab)
    if ANALYSIS_CHARGE not in film_slab.site_properties:
        film_slab = _prepare_charge_structure(film_slab)

    substrate_slab, film_slab, sub_2d, sub_height, d_refs = _prepare_stack(
        substrate_slab, film_slab, match, max_strain
    )
    sub_coords = np.asarray(substrate_slab.cart_coords, dtype=np.float64).copy()

    # In-plane cell the film ends up with: the substrate cell once it has
    # been strained onto it, its own cell in the legacy no-match stacking.
    # The z column of the cell carries nothing else either, so reading the
    # normal direction off ``cart_coords[:, 2]`` below is legitimate: both
    # slabs were leveled by ``_orient_slab``.
    film_in_plane = np.asarray(film_slab.lattice.matrix, dtype=np.float64)[:2, :2]
    film_coords = np.asarray(film_slab.cart_coords, dtype=np.float64).copy()

    # Place the join in-plane, before anything touches z: every site is
    # read off the substrate's own planes, so none of this cares where
    # the film will end up along the normal.
    target_frac, as_built = _place_join(
        sub_coords,
        film_coords,
        sub_2d,
        getattr(config, "registry", None) if config is not None else None,
    )

    # Rebuild both cells with a clean z axis so the vacuum budget is exact.
    def _cell(in_plane, c):
        rows = np.zeros((3, 3), dtype=np.float64)
        rows[0, :2] = in_plane[0]
        rows[1, :2] = in_plane[1]
        rows[2, 2] = float(c)
        return rows

    sub_cell = _cell(sub_2d, sub_height if sub_height > 0 else 1.0)

    # Shift film so its bottom sits at substrate_top + gap.  z is the
    # surface normal for both boards, so these extents are real heights.
    sub_top = float(np.max(sub_coords[:, 2])) if len(sub_coords) else 0.0
    film_bottom = float(np.min(film_coords[:, 2])) if len(film_coords) else 0.0
    z_shift = sub_top + gap - film_bottom
    film_coords[:, 2] += z_shift

    all_species = list(substrate_slab.species) + list(film_slab.species)
    all_coords = np.vstack([sub_coords, film_coords])

    # Extend the shared cell to fit everything plus vacuum.
    top = float(np.max(all_coords[:, 2])) if len(all_coords) else 0.0
    sub_cell[2, 2] = top + vacuum

    site_properties = {
        key: list(substrate_slab.site_properties[key]) + list(film_slab.site_properties[key])
        for key in (ANALYSIS_CHARGE, CHARGE_KNOWN, CHARGE_SOURCE)
    }
    site_properties[SLAB_ID] = [0] * len(substrate_slab) + [1] * len(film_slab)
    interface = Structure(
        _Lattice(sub_cell),
        all_species,
        all_coords,
        coords_are_cartesian=True,
        to_unit_cell=False,
        site_properties=site_properties,
    )
    _attach_charge_geometry(interface, substrate_slab, film_slab, vacuum)
    # The registry is measured here, once, and handed to both consumers:
    # criterion D quotes it, and the orientation advisory names it so two
    # runs can be compared. Two independent routes to the same number
    # would be two chances to disagree about what the run shipped.
    registry = _closest_cross_pair(interface, len(substrate_slab))
    if registry is not None:
        interface.properties["registry_lateral_offset"] = registry.lateral
    _record_registry_properties(interface, target_frac, as_built, sub_2d)
    # Advisory, and deliberately bounded by ``gap`` as well as by the bond
    # scale: ``gap`` is a separation the caller asked for, so a pair at the
    # join sitting at ``gap`` is a request being honoured, not a collision.
    # Only a pair closer than the smaller of the two is worth reporting.
    _warn_interface_proximity(interface, registry, d_refs, max_strain, gap)
    # Which in-plane cell each side actually occupies in the stack.  The
    # substrate is the reference and is never deformed; the film either
    # inherits that cell or, when nothing was strained, keeps its own --
    # the case where the two sides stop agreeing and the area policy has
    # to say which one the energy was normalised by.
    interface.properties["side_in_plane_cells"] = {
        "substrate": sub_2d.tolist(),
        "film": film_in_plane.tolist(),
    }
    return interface


def _attach_charge_geometry(interface, substrate_slab, film_slab, vacuum):
    generated = all(
        slab.properties.get("ai_slab_generated", False) for slab in (substrate_slab, film_slab)
    )
    height = float(interface.lattice.matrix[2, 2])
    bottom = float(np.min(interface.cart_coords[:, 2]))
    top = float(np.max(interface.cart_coords[:, 2]))
    # The external interval connects the film top to the next substrate
    # bottom. The internal interface gap is never used as a vacuum cut.
    finite = bool(generated and vacuum > 0 and height > 0 and top - bottom < height)
    interface.properties["charge_geometry"] = {
        "normal_axis": 2,
        "finite_slab": finite,
        "vacuum_reference": [top / height, 1 + bottom / height] if finite else None,
    }
    interface.properties["charge_overridden_sources"] = {
        side: slab.properties.get("charge_overridden_sources", [])
        for side, slab in (("substrate", substrate_slab), ("film", film_slab))
    }


def _select_termination(terms, index: int, side: str):
    """Pick one termination by index from a :func:`_make_slabs` result."""
    for idx, slab in terms:
        if idx == index:
            return slab
    available = ", ".join(str(idx) for idx, _ in terms) or "none"
    raise ValueError(f"{side} termination {index} not found; available indices: {available}")


def reference_slabs(
    substrate,
    film,
    config: CoherentInterfaceConfig,
    match: _MatchInfo | None = None,
    *,
    substrate_termination: int = 0,
    film_termination: int = 0,
) -> tuple[Any, Any]:
    """Isolated reference slabs for the two sides of a coherent interface.

    An interface energy is only meaningful against the boards the
    interface was built from: same supercell, same thickness, same
    termination.  This returns ``(substrate_reference,
    film_reference)`` as ASE ``Atoms``, cut through the very same
    :func:`_make_slabs` / :func:`_get_terminations` path the builder
    uses, so each board differs from its interface side only by the
    rigid stacking transform:

    * the substrate board is replicated by ``match.substrate_supercell``
      and **never** strained -- its cell is the reference the film is
      deformed onto;
    * the film board is replicated by ``match.film_supercell`` and then
      strained by the same :func:`_strain_film_onto` call
      :func:`_assemble_interface` performs, so its in-plane cell equals
      the substrate cell exactly;
    * with ``match=None`` both boards are the plain 1x1 terminations
      (legacy stacking, no strain), mirroring ``_assemble_interface``'s
      "Without match the slabs are stacked as given".

    ``substrate_termination`` / ``film_termination`` select the cut by
    index; an unavailable index raises :class:`ValueError` naming the
    available ones.  ``symmetric_slabs=True`` routes through
    :func:`_symmetric_terminations` exactly like :func:`_make_slabs`,
    so a forced mirror board references itself.  ``SlabGenerator``
    hands back centred boards and neither replicating nor straining
    moves atoms along z, so both references stay centred in their
    vacuum.
    """
    sub_bulk = _prepare_charge_structure(
        substrate, config.substrate_charges, config.substrate_oxidation
    )
    film_bulk = _prepare_charge_structure(film, config.film_charges, config.film_oxidation)
    sub_terms, film_terms = _make_slabs(sub_bulk, film_bulk, config)

    sub_slab = _select_termination(sub_terms, substrate_termination, "substrate")
    film_slab = _select_termination(film_terms, film_termination, "film")

    if match is not None:
        sub_slab = _scale_in_plane(sub_slab, match.substrate_supercell)
        film_slab = _scale_in_plane(film_slab, match.film_supercell)
        # Same exact in-plane block as ``_assemble_interface`` reads: a and
        # b are level and the vacuum column carries no in-plane part.
        sub_2d = np.asarray(sub_slab.lattice.matrix, dtype=np.float64)[:2, :2]
        film_2d = np.asarray(film_slab.lattice.matrix, dtype=np.float64)[:2, :2]
        if abs(float(np.linalg.det(sub_2d))) > 1e-9 and abs(float(np.linalg.det(film_2d))) > 1e-9:
            film_slab = _strain_film_onto(film_slab, sub_2d)

    return _to_ase(sub_slab), _to_ase(film_slab)
