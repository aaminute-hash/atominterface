"""Coherent interface builder.

Wraps ``pymatgen.analysis.interfaces.coherent_interfaces`` and
:mod:`pymatgen.analysis.interfaces.zsl` behind a backend-agnostic
protocol, adding:

* lattice-match / strain filtering **before** structure generation
* substrate / film termination enumeration with bookkeeping
* gap and vacuum handling
* rotation-matrix preservation for downstream stress transforms
"""

from __future__ import annotations

import inspect
import math
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from atominterface.builders.base import (
    BuilderConfig,
    Candidate,
    InterfaceBuilder,
)
from atominterface.builders.registry import register_builder
from atominterface.builders.strain import (
    max_shear_strain,
    max_stretch_strain,
)
from atominterface.physics import EV_A2_TO_J_M2

if TYPE_CHECKING:
    from collections.abc import Sequence

# Pymatgen is an optional dependency.  These are bound here at module
# level (rather than inside the functions that use them) so that tests
# and downstream code can patch them predictably.  Accessing them
# before pymatgen is installed raises ImportError with a clear message.
try:
    from pymatgen.analysis.interfaces.coherent_interfaces import (  # noqa: WPS433
        CoherentInterfaceBuilder as _PYMTG_COHERENT_BUILDER,
    )
    from pymatgen.core import Structure as _PYMTG_STRUCTURE  # noqa: WPS433
except ImportError:  # pragma: no cover - optional dependency
    _PYMTG_COHERENT_BUILDER = None  # type: ignore
    _PYMTG_STRUCTURE = None  # type: ignore

__all__ = [
    "CoherentInterfaceConfig",
    "CoherentInterfaceBuilder",
]

# Strains within 1e-4 (0.01 %) of each other are treated as equally good
# when ranking lattice matches; see ``_match_lattices``.
_STRAIN_TIE_DIGITS = 4


@dataclass(frozen=True)
class CoherentInterfaceConfig(BuilderConfig):
    """Configuration for coherent interface generation.

    Parameters
    ----------
    substrate_miller : tuple[int, int, int]
        Miller indices of the substrate surface.
    film_miller : tuple[int, int, int]
        Miller indices of the film surface.
    max_strain : float, optional
        Maximum allowed lattice mismatch (stretch or shear) in
        fractional units, e.g. ``0.05`` means 5 %. Candidates
        exceeding this are rejected **before** structure
        construction.
    max_area : float, optional
        Maximum allowed coherent interface area in Å². The ZSL
        enumeration enumerates every integer supercell below this
        area, and its cost grows faster than linearly: on a typical
        2D cell, 100 Å² takes under a second while 500 Å² takes
        minutes. 100 Å² (pymatgen's own default) is enough for the
        usual few-percent mismatch; raise it deliberately.
    gap : float, optional
        Separation gap inserted between the two slabs in Å.
    zsl_max_area_ratio : float, optional
        Hard bound on the ratio between the two matched supercell
        areas. Values near 1 are physically expected, so this only
        rejects pathological matches.
    zsl_max_angle : float, optional
        Maximum allowed angle deviation (degrees) in ZSL search.
    relax_terminations : bool, optional
        If True, enumerate all substrate/film terminations. If
        False, only the first termination of each slab is used.
    vacuum_thickness : float, optional
        Thickness of vacuum padding normal to the interface (Å).
    """

    substrate_miller: tuple[int, int, int] = (0, 0, 1)
    film_miller: tuple[int, int, int] = (0, 0, 1)
    max_strain: float = 0.05
    max_area: float = 100.0
    gap: float = 2.0
    zsl_max_area_ratio: float = 4.0
    zsl_max_angle: float = 2.0
    relax_terminations: bool = True
    vacuum_thickness: float = 15.0


@dataclass
class _MatchInfo:
    """Lightweight container for a lattice match.

    ``substrate_supercell`` / ``film_supercell`` carry the integer 2x2
    transformation matrix of the matched supercell (rows = in-plane
    vectors).  A plain ``(m, n)`` tuple is still accepted so hand-built
    matches stay valid.
    """

    match_area: float
    substrate_supercell: tuple[tuple[int, int], tuple[int, int]] | tuple[int, int]
    film_supercell: tuple[tuple[int, int], tuple[int, int]] | tuple[int, int]
    strain: float
    shear_strain: float
    rotation: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def _lattice_area(lattice: np.ndarray) -> float:
    """Area of the interface plane (Å²)."""
    a = lattice[0]
    b = lattice[1]
    return float(np.linalg.norm(np.cross(a, b)))


def _as_pymatgen_lattice(obj):
    """Normalize a lattice argument to a pymatgen Lattice."""
    from pymatgen.core.lattice import Lattice  # noqa: WPS433

    if isinstance(obj, Lattice):
        return obj
    arr = np.asarray(obj, dtype=np.float64)
    if arr.ndim == 2 and arr.shape == (2, 2):
        # In-plane 2x2 slice -> pad a unit vacuum vector along z.
        mat = np.eye(3, dtype=np.float64)
        mat[:2, :2] = arr
        return Lattice(mat)
    if arr.ndim == 2 and arr.shape == (2, 3):
        # Two in-plane vectors already carrying a z component.
        mat = np.eye(3, dtype=np.float64)
        mat[:2, :] = arr
        return Lattice(mat)
    return Lattice(arr)


def _get_terminations(structure, miller):
    """Return all terminations for a given surface.

    Returns a list of (termination_index, Structure) tuples.
    Uses pymatgen SlabGenerator to produce actual slab structures.
    Falls back to bulk structure if slab generation fails.
    """
    from pymatgen.core.surface import SlabGenerator

    out = []
    try:
        sg = SlabGenerator(
            structure,
            miller_index=miller,
            min_slab_size=1.0,
            min_vacuum_size=1.0,
            in_unit_planes=True,
            center_slab=True,
        )
        slabs = sg.get_slabs()
        if slabs:
            for idx, slab in enumerate(slabs):
                out.append((idx, slab))
    except Exception:
        pass

    if not out:
        out.append((0, structure))

    return out


def _make_slabs(substrate, film, config: CoherentInterfaceConfig):
    """Build substrate / film slabs for every termination.

    Returns two lists of ``(termination_index, structure)``.
    """
    sub_terms_raw = _get_terminations(substrate, config.substrate_miller)
    film_terms_raw = _get_terminations(film, config.film_miller)

    # Flatten: _get_terminations may return (idx, (slab, ...)) or (idx, slab)
    sub_terms = []
    for item in sub_terms_raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            idx, slab = item
            if isinstance(slab, (list, tuple)):
                slab = slab[0]  # unpack nested (slab, metadata)
            sub_terms.append((idx, slab))
        else:
            sub_terms.append((0, item))

    film_terms = []
    for item in film_terms_raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            idx, slab = item
            if isinstance(slab, (list, tuple)):
                slab = slab[0]
            film_terms.append((idx, slab))
        else:
            film_terms.append((0, item))

    if not config.relax_terminations:
        sub_terms = sub_terms[:1]
        film_terms = film_terms[:1]

    return sub_terms, film_terms


def _supercell_transform(matrix: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Round a ZSL 2x2 transformation matrix to its integer form.

    ZSL returns matrices whose entries are integers up to floating-point
    noise.  Returning the rounded matrix (instead of, say, its diagonal
    entries) keeps rotated supercells honest: a 45-degree cell has
    off-diagonal entries that carry the whole meaning.  Use
    :func:`_supercell_multiplicity` for the number of primitive cells it
    contains.
    """
    rounded = np.rint(np.asarray(matrix, dtype=np.float64)).astype(int)
    return (
        (int(rounded[0, 0]), int(rounded[0, 1])),
        (int(rounded[1, 0]), int(rounded[1, 1])),
    )


def _multiplicity(transform) -> int:
    """Number of primitive cells in a supercell description.

    Accepts either the integer 2x2 matrix produced by
    :func:`_supercell_transform` or a legacy flat ``(m, n)`` pair.
    """
    rows = tuple(transform)
    if len(rows) == 2 and all(np.isscalar(row) for row in rows):
        return max(abs(int(rows[0]) * int(rows[1])), 1)
    (a, b), (c, d) = rows
    return max(abs(int(a) * int(d) - int(b) * int(c)), 1)


def _match_lattices(
    sub_lattice, film_lattice, config: CoherentInterfaceConfig, strain_limit=None
) -> list[_MatchInfo]:
    """Generate coherent matches with strain filtering.

    Uses :class:`pymatgen.analysis.interfaces.zsl.ZSLGenerator` when
    available, and falls back to a minimal commensurate-search when
    pymatgen's generator is unavailable or rejects every candidate.

    Parameters
    ----------
    strain_limit
        Override for ``config.max_strain``.  Pass ``math.inf`` to skip
        strain filtering entirely -- that is how
        :meth:`CoherentInterfaceBuilder.preview_matches` reports the best
        geometry the pair can achieve, which is what a user needs to see
        when a tolerance rejects everything.
    """
    from pymatgen.analysis.interfaces.zsl import ZSLGenerator  # noqa: WPS433

    limit = config.max_strain if strain_limit is None else float(strain_limit)

    sub_lat = _as_pymatgen_lattice(sub_lattice)
    film_lat = _as_pymatgen_lattice(film_lattice)

    # In-plane vectors normal to the surface normal.  The coherent
    # interface builder always cuts along the supplied Miller
    # direction, so we project the first two reciprocal-lattice
    # directions onto the surface plane.
    sub_vecs = np.array(sub_lat.matrix[:2], dtype=np.float64)
    film_vecs = np.array(film_lat.matrix[:2], dtype=np.float64)

    # ``max_area_ratio_tol`` is a *relative* tolerance on the area ratio
    # between the two matched supercells, so it must be derived from the
    # configured area window -- not from ``zsl_max_area_ratio``.  The old
    # ``zsl_max_area_ratio - 1.0`` mapping opened the search to +-300 %
    # and cost ~5x the runtime while returning exactly the same matches,
    # because anything that far off is rejected in the loop below anyway.
    ratio_lo, ratio_hi = config.max_area_ratio
    area_ratio_tol = max(ratio_hi - 1.0, 1.0 - ratio_lo)

    generator = ZSLGenerator(
        max_area_ratio_tol=area_ratio_tol,
        max_area=int(config.max_area),
        max_angle_tol=math.radians(config.zsl_max_angle),
    )

    sub_cell_area = abs(float(np.linalg.det(sub_vecs[:, :2])))
    film_cell_area = abs(float(np.linalg.det(film_vecs[:, :2])))

    matches: list[_MatchInfo] = []
    try:
        for match in generator(film_vecs, sub_vecs):
            area = float(match.match_area)
            if area <= 0 or area > config.max_area:
                continue

            film_t = np.asarray(match.film_transformation, dtype=np.float64)
            sub_t = np.asarray(match.substrate_transformation, dtype=np.float64)

            # ``zsl_max_area_ratio`` finally means what its docstring
            # says: a hard bound on the supercell area ratio.
            if not _area_ratio_ok(film_t, sub_t, film_cell_area, sub_cell_area, config):
                continue

            strain, shear = _strain_of(film_t @ film_vecs, sub_t @ sub_vecs)
            if max(strain, shear) > limit:
                continue

            rotation = None
            with suppress(Exception):
                rotation = np.asarray(match.match_transformation, dtype=np.float64)

            matches.append(
                _MatchInfo(
                    match_area=area,
                    substrate_supercell=_supercell_transform(sub_t),
                    film_supercell=_supercell_transform(film_t),
                    strain=strain,
                    shear_strain=shear,
                    rotation=rotation,
                    metadata={"zsl": True},
                )
            )
    except Exception:  # noqa: BLE001
        # ZSL is best-effort; fall through to the fallback below.
        pass

    if not matches:
        matches = _fallback_matches(sub_lattice, film_lattice, config, strain_limit=limit)

    # ``max(strain, shear)`` is the quantity the filter actually applies,
    # so it is also the honest ranking key.  The strain is quantised
    # first: for a pair with identical cell shapes every integer
    # supercell has *exactly* the same mismatch, so the raw floats differ
    # only in the last bits and the tie-break collapsed to whichever
    # giant supercell happened to round down.  Quantising restores the
    # intended behaviour -- among equally good matches, prefer the
    # smallest cell.
    return sorted(
        matches,
        key=lambda m: (round(max(m.strain, m.shear_strain), _STRAIN_TIE_DIGITS), m.match_area),
    )


def _area_ratio_ok(
    film_t, sub_t, film_cell_area: float, sub_cell_area: float, config: CoherentInterfaceConfig
) -> bool:
    """Whether the two matched supercell areas are within the ratio bound."""
    film_area = abs(float(np.linalg.det(film_t))) * film_cell_area
    sub_area = abs(float(np.linalg.det(sub_t))) * sub_cell_area
    if film_area <= 0 or sub_area <= 0:
        return False
    ratio = max(film_area / sub_area, sub_area / film_area)
    return bool(ratio <= config.zsl_max_area_ratio)


def _strain_of(film_cell, sub_cell) -> tuple[float, float]:
    """In-plane (stretch, shear) strain needed to fit the film onto the substrate.

    ``film_cell`` comes first because it is the cell that gets deformed.

    ``film_cell`` / ``sub_cell`` are plain arrays here: wrapping them in
    :class:`pymatgen.core.Lattice` objects (as an earlier version did)
    costs two full lattice constructions per candidate for no benefit,
    and ZSL already returns hundreds of candidates.
    """
    strain = float(max_stretch_strain(film_cell, sub_cell))
    shear = float(max_shear_strain(film_cell, sub_cell))
    return strain, shear


def _fallback_matches(
    sub_lattice, film_lattice, config: CoherentInterfaceConfig, strain_limit=None
):
    """Brute-force commensurate search for small supercells.

    This is intentionally simple: it scales the *film* cell by integer
    factors ``(m, n)`` up to a small cutoff and keeps the cells that land
    close to the substrate cell, reporting the film supercell in the
    metadata.  It is **not** a replacement for ZSLGenerator, merely a
    guarantee that the builder never returns an empty list when ZSL is
    unavailable.
    """
    limit = config.max_strain if strain_limit is None else float(strain_limit)

    matches: list[_MatchInfo] = []
    sub_lat = _as_pymatgen_lattice(sub_lattice)
    film_lat = _as_pymatgen_lattice(film_lattice)
    sub_vecs = np.array(sub_lat.matrix[:2], dtype=np.float64)
    film_vecs = np.array(film_lat.matrix[:2], dtype=np.float64)

    film_area = float(abs(np.linalg.det(film_vecs[:, :2])))
    if film_area <= 0:
        return []

    cutoff = max(2, int(math.sqrt(config.max_area / film_area)) + 1)
    for m in range(1, cutoff + 1):
        for n in range(1, cutoff + 1):
            if m * n > cutoff:
                continue
            area = film_area * m * n
            if area > config.max_area:
                continue
            scaled = np.diag([float(m), float(n)]) @ film_vecs
            strain, shear = _strain_of(scaled, sub_vecs)
            if max(strain, shear) > limit:
                continue
            matches.append(
                _MatchInfo(
                    match_area=area,
                    substrate_supercell=((1, 0), (0, 1)),
                    film_supercell=((m, 0), (0, n)),
                    strain=strain,
                    shear_strain=shear,
                    metadata={"zsl": False, "fallback": True},
                )
            )
    return matches


def _dense_transform(transform) -> np.ndarray:
    """2x2 float form of a supercell description."""
    rows = tuple(transform) if transform is not None else ()
    if len(rows) == 2 and all(np.isscalar(row) for row in rows):
        return np.diag([float(rows[0]), float(rows[1])])
    return np.asarray(rows, dtype=np.float64)


def _scale_in_plane(structure, transform):
    """Replicate a slab by its integer in-plane supercell.

    Two guards matter here:

    * the third row of the scaling matrix is fixed to 1 so the slab
      thickness (and hence the vacuum budget) is untouched;
    * the slab is copied before scaling.  Newer pymatgen versions default
      ``make_supercell`` to ``in_place=True``, which silently turned the
      candidate loop into a compounding supercell (6.5 -> 85 -> 1104 A^2
      within four candidates) because every candidate re-scaled the same
      slab object.
    """
    mat = np.rint(_dense_transform(transform)).astype(int)
    scaling = np.array(
        [
            [mat[0, 0], mat[0, 1], 0],
            [mat[1, 0], mat[1, 1], 0],
            [0, 0, 1],
        ],
        dtype=int,
    )
    slab = structure.copy()
    if "in_place" in inspect.signature(slab.make_supercell).parameters:
        return slab.make_supercell(scaling, in_place=False)
    return slab.make_supercell(scaling)


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
    if config is not None:
        gap = float(getattr(config, "gap", gap))
        vacuum = float(getattr(config, "vacuum_thickness", vacuum))

    if match is not None:
        substrate_slab = _scale_in_plane(substrate_slab, match.substrate_supercell)
        film_slab = _scale_in_plane(film_slab, match.film_supercell)

    sub_mat = np.asarray(substrate_slab.lattice.matrix, dtype=np.float64)
    film_mat = np.asarray(film_slab.lattice.matrix, dtype=np.float64)
    sub_2d = np.asarray(sub_mat[:2, :2], dtype=np.float64)
    film_2d = np.asarray(film_mat[:2, :2], dtype=np.float64)

    sub_coords = np.asarray(substrate_slab.cart_coords, dtype=np.float64).copy()
    film_coords = np.asarray(film_slab.cart_coords, dtype=np.float64).copy()

    # Strain the film onto the substrate cell.  Applying the same 2x2
    # map to the cartesian coordinates and to the lattice keeps the
    # fractional coordinates -- i.e. the stacking sequence -- intact.
    coherent = (
        match is not None
        and abs(float(np.linalg.det(sub_2d))) > 1e-9
        and abs(float(np.linalg.det(film_2d))) > 1e-9
    )
    if coherent:
        deformation = sub_2d @ np.linalg.inv(film_2d)
        film_coords[:, :2] = film_coords[:, :2] @ deformation.T
        film_2d = sub_2d.copy()

    # Rebuild both cells with a clean z axis so the vacuum budget is exact.
    def _cell(in_plane, c):
        rows = np.zeros((3, 3), dtype=np.float64)
        rows[0, :2] = in_plane[0]
        rows[1, :2] = in_plane[1]
        rows[2, 2] = float(c)
        return rows

    sub_cell = _cell(sub_2d, sub_mat[2, 2] if sub_mat[2, 2] > 0 else 1.0)

    # Shift film so its bottom sits at substrate_top + gap.
    sub_top = float(np.max(sub_coords[:, 2])) if len(sub_coords) else 0.0
    film_bottom = float(np.min(film_coords[:, 2])) if len(film_coords) else 0.0
    z_shift = sub_top + gap - film_bottom
    film_coords[:, 2] += z_shift

    all_species = list(substrate_slab.species) + list(film_slab.species)
    all_coords = np.vstack([sub_coords, film_coords])

    # Extend the shared cell to fit everything plus vacuum.
    top = float(np.max(all_coords[:, 2])) if len(all_coords) else 0.0
    sub_cell[2, 2] = top + vacuum

    return Structure(
        _Lattice(sub_cell),
        all_species,
        all_coords,
        coords_are_cartesian=True,
        to_unit_cell=False,
    )


def _reduced_formula(structure) -> str:
    """Best-effort reduced formula without requiring composition."""
    try:
        return str(structure.composition.reduced_formula)
    except Exception:  # noqa: BLE001
        try:
            from collections import Counter

            c = Counter(str(s.species) for s in structure.sites)
            return "".join(f"{k}{v}" for k, v in sorted(c.items()))
        except Exception:  # noqa: BLE001
            return "unknown"


def _to_ase(obj):
    """Return an ASE ``Atoms`` view of a pymatgen ``Structure``.

    ``Candidate.atoms`` is documented as the ASE view, and ASE is what the
    relaxers and every exporter consume, so the conversion happens here
    rather than at each call site.
    """
    from ase.atoms import Atoms

    if isinstance(obj, Atoms):
        return obj

    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_atoms(obj)


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
        "energy_per_area_jm2": energy_per_area_jm2,
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
        "mismatch_stretch": match.strain,
        "mismatch_shear": match.shear_strain,
        "mismatch": max(match.strain, match.shear_strain),
        "stretch_strain": match.strain,
        "shear_strain": match.shear_strain,
        "geometry": "coherent-supercell",
        "gap_ang": config.gap,
        "vacuum_ang": config.vacuum_thickness,
        **match.metadata,
        "rotation_matrix": None if match.rotation is None else match.rotation.tolist(),
    }

    # The substrate cell is the reference; the film is strained onto it
    # (see ``_assemble_interface``), so all of the mismatch lands on the
    # film and none on the substrate.  Reporting ``shear`` as the film
    # strain -- as an earlier version did -- made a 2.6 % isotropic
    # mismatch show up as "0.0258 / 0.0000" with no relation to which
    # side actually carries it.
    sub_sc = _multiplicity(match.substrate_supercell)
    film_sc = _multiplicity(match.film_supercell)

    return Candidate(
        atoms=_to_ase(interface),
        interface_id=(
            f"coherent:{config.substrate_miller}-{config.film_miller}"
            f":{sub_term_idx}-{film_term_idx}:{sub_sc}x{film_sc}"
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
        n_substrate=len(substrate_slab),
        n_film=len(film_slab),
        metadata=metadata,
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

    def validate(self) -> list[str]:
        """Validate the input pair before building."""
        errors: list[str] = []

        if self.substrate is None or self.film is None:
            errors.append("substrate and film structures must be provided")
            return errors

        sub_lat = np.asarray(self.substrate.lattice.matrix, dtype=np.float64)
        film_lat = np.asarray(self.film.lattice.matrix, dtype=np.float64)

        if np.linalg.det(sub_lat) <= 0:
            errors.append("substrate lattice has non-positive volume")
        if np.linalg.det(film_lat) <= 0:
            errors.append("film lattice has non-positive volume")

        if not (0.0 <= self.config.max_strain <= 1.0):
            errors.append("max_strain must be in [0, 1]")
        if self.config.max_area <= 0:
            errors.append("max_area must be positive")

        if any(
            miller == (0, 0, 0)
            for miller in (self.config.substrate_miller, self.config.film_miller)
        ):
            errors.append("Miller indices must be non-zero")

        return errors

    def list_terminations(self) -> dict[str, list[tuple[int, str]]]:
        """Enumerate terminations for both sides."""
        sub_terms = _get_terminations(self.substrate, self.config.substrate_miller)
        film_terms = _get_terminations(self.film, self.config.film_miller)

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

    def _match(self, strain_limit=None) -> list[_MatchInfo]:
        """Match the two slabs' in-plane cells.

        Matching must happen on the **slab** cells, not on the bulk
        cells: ``SlabGenerator`` re-orients the cell so the surface
        normal is the third axis, and for any surface other than (001)
        the bulk's first two vectors are not even in the interface
        plane.  All terminations of one slab share the same lateral
        cell, so a single match covers every termination pair.
        """
        sub_terms, film_terms = _make_slabs(self.substrate, self.film, self.config)
        return _match_lattices(
            sub_terms[0][1].lattice.matrix,
            film_terms[0][1].lattice.matrix,
            self.config,
            strain_limit=strain_limit,
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

    def build_all(self) -> list[Candidate]:
        """Generate all strain-acceptable interface candidates."""
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))

        matches = self._match()
        if not matches:
            return []

        sub_terms, film_terms = _make_slabs(self.substrate, self.film, self.config)

        candidates: list[Candidate] = []
        for match in matches:
            for sub_idx, sub_slab in sub_terms:
                for film_idx, film_slab in film_terms:
                    try:
                        interface = _assemble_interface(sub_slab, film_slab, match, self.config)
                    except Exception:  # noqa: BLE001
                        # A single bad termination must not abort the
                        # whole enumeration.
                        continue

                    candidates.append(
                        _to_candidate(
                            interface=interface,
                            substrate_slab=sub_slab,
                            film_slab=film_slab,
                            match=match,
                            sub_term_idx=sub_idx,
                            film_term_idx=film_idx,
                            config=self.config,
                        )
                    )
                    # Stop as soon as the cap is reached: building a
                    # supercell is the expensive part, and matches are
                    # already ranked best-first.
                    if len(candidates) >= self.config.max_candidates:
                        return candidates

        return candidates

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


# Register the coherent builder eagerly so `list-builders` works
# without importing the class explicitly.
try:
    from atominterface.builders.registry import builder_registry as _registry

    _registry.register(CoherentInterfaceBuilder.name, CoherentInterfaceBuilder)
except Exception:  # noqa: BLE001
    pass
