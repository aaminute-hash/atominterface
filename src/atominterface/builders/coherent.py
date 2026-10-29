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
import warnings
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
from atominterface.builders.charge import (
    ANALYSIS_CHARGE,
    CHARGE_KNOWN,
    CHARGE_SOURCE,
    SLAB_ID,
    _site_dependent_charge,
    analyze_charge_polarity,
    charge_rejection_reasons,
    make_symmetric_slab,
    resolve_charge_data,
    tasker_classification,
    termination_cuts,
    validate_charge_options,
)
from atominterface.builders.registry import register_builder
from atominterface.builders.strain import (
    max_shear_strain,
    max_stretch_strain,
)
from atominterface.physics import EV_A2_TO_J_M2

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

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
    "reference_slabs",
]

# Strains within 1e-4 (0.01 %) of each other are treated as equally good
# when ranking lattice matches; see ``_match_lattices``.
_STRAIN_TIE_DIGITS = 4

# Absolute tolerance (Å) below which a cell component counts as nothing
# when deciding whether a slab already satisfies the "+z normal" contract
# enforced by :func:`_orient_slab`.  Cell lengths are a few Å, so this is
# ~1e-10 relative -- far above float64 round-off, far below any physical
# or numerical quantity downstream.
_ORIENT_TOL = 1e-9


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
    symmetric_slabs : bool or str, optional
        ``"auto"`` (default) never transforms a structure: a termination
        that the supplied charge model marks polar only raises a warning
        naming the Miller indices and the ``--symmetric`` option.  ``True``
        forces mirror-symmetric boards (same termination on both surfaces,
        cut at half the configured thickness).  ``False`` disables both.
    min_thickness_substrate : float, optional
        Per-side override of ``min_thickness`` for the substrate slab (Å).
        ``None`` (default) falls back to the shared ``min_thickness``, so
        existing configurations behave exactly as before.
    min_thickness_film : float, optional
        Per-side override of ``min_thickness`` for the film slab (Å), with
        the same ``None`` fallback as ``min_thickness_substrate``.
    area_policy : str, optional
        Which side's area an interface energy is normalised by when the
        two sides do **not** agree: ``"strict"`` (default) reports no
        usable area at all, ``"substrate"`` or ``"film"`` commit to that
        side.  A geometric mean of the two is never used -- it would make
        a mismatch look handled instead of reported.
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
    charge_policy: str = "report"
    substrate_oxidation: dict[str, float] | None = None
    film_oxidation: dict[str, float] | None = None
    substrate_charges: tuple[float, ...] | None = None
    film_charges: tuple[float, ...] | None = None
    charge_tolerance: float = 1e-6
    dipole_tolerance: float = 1e-3
    symmetric_slabs: bool | str = "auto"
    min_thickness_substrate: float | None = None
    min_thickness_film: float | None = None
    area_policy: str = "strict"

    def validate(self) -> list[str]:
        errors = super().validate()
        try:
            validate_charge_options(
                self.charge_policy, self.charge_tolerance, self.dipole_tolerance
            )
        except ValueError as exc:
            errors.append(str(exc))
        if not (
            self.symmetric_slabs is True
            or self.symmetric_slabs is False
            or self.symmetric_slabs == "auto"
        ):
            errors.append("symmetric_slabs must be True, False or 'auto'")
        if self.vacuum_dir != 2:
            errors.append("coherent construction currently requires vacuum_dir=2")
        if not math.isfinite(self.vacuum_thickness) or self.vacuum_thickness < 0:
            errors.append("vacuum_thickness must be finite and non-negative")
        if not math.isfinite(self.gap) or self.gap < 0:
            errors.append("gap must be finite and non-negative")
        for name, value in (
            ("min_thickness_substrate", self.min_thickness_substrate),
            ("min_thickness_film", self.min_thickness_film),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                errors.append(f"{name} must be finite and positive when provided")
        if self.area_policy not in AREA_POLICIES:
            errors.append(
                f"area_policy must be one of {list(AREA_POLICIES)}, got {self.area_policy!r}"
            )
        return errors

    def slab_thicknesses(self) -> tuple[float, float]:
        """Resolved ``(substrate, film)`` minimum slab thicknesses in Å.

        Each side falls back to the shared ``min_thickness`` when its
        per-side override is ``None``; the fallback is an explicit
        ``is None`` check (not truthiness) so a nonsensical ``0.0``
        override survives to be reported by :meth:`validate` instead of
        being silently replaced.
        """
        substrate = self.min_thickness
        if self.min_thickness_substrate is not None:
            substrate = self.min_thickness_substrate
        film = self.min_thickness
        if self.min_thickness_film is not None:
            film = self.min_thickness_film
        return substrate, film


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
    # Matched in-plane supercell areas (Å²), *before* the film is strained
    # onto the substrate.  They are what ``_area_ratio_ok`` already
    # computes to bound the ratio; keeping them spares a second pass and
    # records the pre-strain mismatch the strain then removes.
    substrate_area: float = 0.0
    film_area: float = 0.0


def _lattice_area(lattice: np.ndarray) -> float:
    """Area of the interface plane (Å²)."""
    a = lattice[0]
    b = lattice[1]
    return float(np.linalg.norm(np.cross(a, b)))


#: Accepted values of :attr:`CoherentInterfaceConfig.area_policy`.
AREA_POLICIES: tuple[str, ...] = ("strict", "substrate", "film")


@dataclass(frozen=True)
class InterfaceAreas:
    """The two sides' interface areas plus the one to normalise by.

    ``area_substrate`` / ``area_film`` are the in-plane areas (Å²) each
    side actually occupies -- the pair a "bilateral" energy ledger needs
    in order to say which side a normalisation came from.  A coherent
    stack shares one cell, so the pair is normally equal; it diverges as
    soon as one side is left unstrained.

    ``area_used`` is the single number a downstream energy division
    should take, resolved from ``policy``:

    ``"strict"``
        Refuse to choose when the sides disagree: ``area_used`` is NaN
        and ``note`` reports both numbers.
    ``"substrate"`` / ``"film"``
        Always take that side, disagreement or not.

    A geometric mean ``sqrt(a_sub * a_film)`` is deliberately **not**
    offered.  It would turn a real mismatch into a plausible-looking
    number, i.e. hide exactly the thing this record exists to expose.
    """

    area_substrate: float
    area_film: float
    area_used: float
    consistent: bool
    policy: str
    note: str


def _in_plane_vectors(lattice) -> np.ndarray:
    """First two lattice rows as 2x3 cartesian vectors.

    Accepts a 3x3 or 2x3 matrix, a 2x2 in-plane slice (padded with a
    zero z component) or anything exposing a pymatgen-style ``matrix``.
    """
    matrix = getattr(lattice, "matrix", lattice)
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape == (2, 2):
        padded = np.zeros((2, 3), dtype=np.float64)
        padded[:, :2] = mat
        return padded
    return mat[:2, :]


def interface_areas(
    substrate_lattice,
    film_lattice,
    *,
    tolerance: float = 1e-6,
    policy: str = "strict",
) -> InterfaceAreas:
    """Resolve the interface area of a two-sided stack.

    ``tolerance`` is a *relative* window on ``abs(a_sub - a_film)``
    against the larger of the two, so the verdict is scale-free.
    ``policy`` must be one of :data:`AREA_POLICIES`.

    Nothing here raises on a mismatch: an inconsistent pair is a fact to
    record, and deciding what to do with such a candidate belongs to the
    caller.  Only an unknown ``policy`` raises, because it is a
    programming error rather than a property of the structure.
    """
    if policy not in AREA_POLICIES:
        raise ValueError(f"area_policy must be one of {list(AREA_POLICIES)}, got {policy!r}")

    a_sub = _lattice_area(_in_plane_vectors(substrate_lattice))
    a_film = _lattice_area(_in_plane_vectors(film_lattice))
    scale = max(a_sub, a_film)
    consistent = bool(abs(a_sub - a_film) <= tolerance * scale) if scale > 0 else True

    if policy == "film":
        area_used = a_film
    elif policy == "substrate" or consistent:
        area_used = a_sub
    else:
        area_used = float("nan")

    if consistent:
        note = f"both sides {a_sub:.6g} Å²; area taken from the {policy} policy"
    elif policy == "strict":
        note = (
            f"area mismatch: substrate {a_sub:.6g} Å² vs film {a_film:.6g} Å²; "
            "no single area is valid and no average is substituted"
        )
    else:
        note = (
            f"area mismatch: substrate {a_sub:.6g} Å² vs film {a_film:.6g} Å²; "
            f"area taken from the {policy} side by policy"
        )

    return InterfaceAreas(
        area_substrate=a_sub,
        area_film=a_film,
        area_used=area_used,
        consistent=consistent,
        policy=policy,
        note=note,
    )


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


def _orient_slab(slab):
    """Return a slab whose surface normal is exactly ``+z``, ``a``/``b`` in ``xy``.

    ``SlabGenerator(primitive=True)`` returns cells whose in-plane vectors
    carry a z component for every surface but (001): fcc (111) comes back
    as ``a = (1.81, 0, -1.81)``, ``b = (0, 1.81, -1.81)``.  Every
    downstream ``[:2]`` slice then projects those vectors onto the xy
    plane, which silently shrinks the interface area by ``sqrt(3)``
    (3.2671 instead of 5.6587 Å²) and lets ``_assemble_interface`` rebuild
    a hexagonal cell as a square one.  Rotating once here turns that
    assumption into a fact for every consumer further down the chain
    instead of patching six separate slices.

    The rotation is applied to the orthogonal-c form so that ``c`` really
    is parallel to ``cross(a, b)`` -- clearing the in-plane part of a skew
    ``c`` would otherwise discard ~40 Å of real cell vector.

    A slab that already satisfies the contract is handed back unchanged
    and **bit-identical**, which is what keeps the published (001)
    reference numbers pinned.
    """
    from pymatgen.core import Structure
    from pymatgen.core.lattice import Lattice as _Lattice

    matrix = np.asarray(slab.lattice.matrix, dtype=np.float64)
    already_aligned = max(
        abs(float(matrix[0, 2])),
        abs(float(matrix[1, 2])),
        abs(float(matrix[2, 0])),
        abs(float(matrix[2, 1])),
    )
    if already_aligned <= _ORIENT_TOL:
        return slab

    orth = slab.get_orthogonal_c_slab()  # c perpendicular to a, b
    L = np.asarray(orth.lattice.matrix, dtype=np.float64)
    n = np.cross(L[0], L[1])
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    e1 = L[0] / np.linalg.norm(L[0])
    e2 = np.cross(n, e1)
    e2 = e2 / np.linalg.norm(e2)
    rot = np.vstack([e1, e2, n])  # rows are the new basis vectors, in old coordinates
    new_L = L @ rot.T
    new_L[2, :2] = 0.0  # defensive: drop any residual in-plane part of c
    coords = np.asarray(orth.cart_coords, dtype=np.float64) @ rot.T
    out = Structure(
        _Lattice(new_L),
        list(orth.species),
        coords,
        coords_are_cartesian=True,
        to_unit_cell=False,
        site_properties={key: list(values) for key, values in orth.site_properties.items()},
    )
    out.properties.update(getattr(orth, "properties", {}) or {})
    return out


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


def _format_matrix(matrix) -> str:
    """Format a 2x2 supercell matrix as an unambiguous, filename-safe string.

    The determinant alone cannot identify a supercell (det = 1, 2, 3 covers
    both ``1x1``/``2x2``/``3x3`` diagonal cells and rotated cells such as
    ``sqrt(2) x sqrt(2) R45deg``), which is how duplicate ``interface_id``
    values used to collide.  The full integer matrix is written with ``_``
    delimiters so no two distinct matrices can produce the same label
    (``((11, 0), (0, 1))`` and ``((1, 10), (0, 1))`` stay distinguishable).
    """
    dense = np.rint(_dense_transform(matrix)).astype(int)
    return "_".join(str(int(value)) for value in dense.ravel())


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

    # True parallelogram area |a x b|.  Using the norm of the cross product
    # rather than the 2x2 determinant keeps the result independent of how
    # the cell is rotated: the determinant implicitly projects a, b onto xy
    # and under-reports any cell whose in-plane vectors are not level.
    sub_cell_area = float(np.linalg.norm(np.cross(sub_vecs[0], sub_vecs[1])))
    film_cell_area = float(np.linalg.norm(np.cross(film_vecs[0], film_vecs[1])))

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
                    substrate_area=abs(float(np.linalg.det(sub_t))) * sub_cell_area,
                    film_area=abs(float(np.linalg.det(film_t))) * film_cell_area,
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

    # True area |a x b|, not the 2x2 determinant -- see ``_match_lattices``.
    film_area = float(np.linalg.norm(np.cross(film_vecs[0], film_vecs[1])))
    sub_cell_area = float(np.linalg.norm(np.cross(sub_vecs[0], sub_vecs[1])))
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
                    substrate_area=sub_cell_area,
                    film_area=area,
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
        slab = slab.make_supercell(scaling, in_place=False)
    else:
        slab = slab.make_supercell(scaling)
    # Slab.copy() in some pymatgen versions preserves site properties but
    # drops Structure.properties, including the finite-slab provenance.
    slab.properties.update(structure.properties)
    return slab


def _strain_film_onto(film_slab, substrate_slab_2d):
    """Strain ``film_slab`` in-plane onto the substrate cell.

    Returns a new ``Structure`` whose in-plane lattice *is*
    ``substrate_slab_2d`` and whose cartesian coordinates carry the
    matching deformation ``sub_2d @ inv(film_2d)`` applied to the
    in-plane components (``coords @ deformation.T``).  Because the same
    map acts on the coordinates and on the cell, the fractional
    coordinates -- the stacking sequence -- are preserved.

    Shared by :func:`_assemble_interface` and :func:`reference_slabs`
    so an interface and the reference boards it is referenced against
    are strained by one implementation, not two approximations.  The
    slab's vacuum column along z is carried over untouched and the
    input is never modified.
    """
    from pymatgen.core import Structure
    from pymatgen.core.lattice import Lattice as _Lattice

    film_mat = np.asarray(film_slab.lattice.matrix, dtype=np.float64)
    # ``[:2, :2]`` is the true in-plane block because the z components of a
    # and b are zero -- guaranteed by ``_orient_slab`` for every slab that
    # comes out of :func:`_get_terminations`.
    film_2d = np.asarray(film_mat[:2, :2], dtype=np.float64)
    sub_2d = np.asarray(substrate_slab_2d, dtype=np.float64)

    deformation = sub_2d @ np.linalg.inv(film_2d)
    coords = np.asarray(film_slab.cart_coords, dtype=np.float64).copy()
    coords[:, :2] = coords[:, :2] @ deformation.T

    # Clean z axis, mirroring the cell ``_assemble_interface`` builds:
    # the in-plane rows carry no z component and the film keeps its own
    # vacuum column, so a strained reference board stays centred.
    rows = np.zeros((3, 3), dtype=np.float64)
    rows[0, :2] = sub_2d[0]
    rows[1, :2] = sub_2d[1]
    rows[2, 2] = film_mat[2, 2] if film_mat[2, 2] > 0 else 1.0

    site_properties = {key: list(values) for key, values in film_slab.site_properties.items()}
    strained = Structure(
        _Lattice(rows),
        list(film_slab.species),
        coords,
        coords_are_cartesian=True,
        to_unit_cell=False,
        site_properties=site_properties or None,
    )
    strained.properties.update(film_slab.properties)
    return strained


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

    if ANALYSIS_CHARGE not in substrate_slab.site_properties:
        substrate_slab = _prepare_charge_structure(substrate_slab)
    if ANALYSIS_CHARGE not in film_slab.site_properties:
        film_slab = _prepare_charge_structure(film_slab)
    if match is not None:
        substrate_slab = _scale_in_plane(substrate_slab, match.substrate_supercell)
        film_slab = _scale_in_plane(film_slab, match.film_supercell)

    sub_mat = np.asarray(substrate_slab.lattice.matrix, dtype=np.float64)
    film_mat = np.asarray(film_slab.lattice.matrix, dtype=np.float64)
    # In-plane block: exact, since ``_orient_slab`` guarantees a and b carry
    # no z component and the vacuum column no in-plane one.
    sub_2d = np.asarray(sub_mat[:2, :2], dtype=np.float64)
    film_2d = np.asarray(film_mat[:2, :2], dtype=np.float64)

    sub_coords = np.asarray(substrate_slab.cart_coords, dtype=np.float64).copy()

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
    # In-plane cell the film ends up with: the substrate cell once it has
    # been strained onto it, its own cell in the legacy no-match stacking.
    # The z column of the cell carries nothing else either, so reading the
    # normal direction off ``cart_coords[:, 2]`` below is legitimate: both
    # slabs were leveled by ``_orient_slab``.
    film_in_plane = np.asarray(film_slab.lattice.matrix, dtype=np.float64)[:2, :2]
    film_coords = np.asarray(film_slab.cart_coords, dtype=np.float64).copy()

    # Rebuild both cells with a clean z axis so the vacuum budget is exact.
    def _cell(in_plane, c):
        rows = np.zeros((3, 3), dtype=np.float64)
        rows[0, :2] = in_plane[0]
        rows[1, :2] = in_plane[1]
        rows[2, 2] = float(c)
        return rows

    sub_cell = _cell(sub_2d, sub_mat[2, 2] if sub_mat[2, 2] > 0 else 1.0)

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

    # Bilateral area.  ``area`` above is the stack's own cell, which is
    # what the rest of the pipeline keeps normalising by; this resolves
    # the two sides separately and records whether they agree.  An
    # unstrained film (legacy stacking) is a legitimate geometry, so a
    # mismatch is recorded, never raised.
    side_cells = (getattr(interface, "properties", None) or {}).get("side_in_plane_cells")
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

        candidates: list[Candidate] = []
        for match in matches:
            for sub_idx, sub_slab in slabs[0]:
                for film_idx, film_slab in slabs[1]:
                    candidate = self._build_candidate(match, sub_idx, sub_slab, film_idx, film_slab)
                    if candidate is not None and self._accept_charge(candidate):
                        candidates.append(candidate)
                    if len(candidates) >= self.config.max_candidates:
                        self.last_build_report["stopped_at_limit"] = True
                        return candidates

        if not candidates:
            self.last_build_report["empty_stage"] = (
                "charge_filter" if self.last_build_report["rejected"] else "assembly"
            )
        return candidates

    def _build_candidate(self, match, sub_idx, sub_slab, film_idx, film_slab):
        report = self.last_build_report
        report["attempted"] += 1
        # Validate charge provenance outside the geometry error handler.
        for slab in (sub_slab, film_slab):
            _prepare_charge_structure(slab)
        try:
            interface = _assemble_interface(sub_slab, film_slab, match, self.config)
        except Exception:  # noqa: BLE001
            report["assembly_failed"] += 1
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


# Register the coherent builder eagerly so `list-builders` works
# without importing the class explicitly.
try:
    from atominterface.builders.registry import builder_registry as _registry

    _registry.register(CoherentInterfaceBuilder.name, CoherentInterfaceBuilder)
except Exception:  # noqa: BLE001
    pass
