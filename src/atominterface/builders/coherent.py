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
from typing import TYPE_CHECKING, NamedTuple

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
    _orient_slab,
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
    "GeometryGateError",
    "measure_registry",
    "reference_slabs",
]


class GeometryGateError(ValueError):
    """An assembled interface violates a geometry invariant.

    Raised by :func:`_assemble_interface` when a board's in-plane cell
    would be misread downstream.  Every slice the builder chain takes
    (``lattice[:2, :2]``, ``cart_coords[:, 2]``) reads the cell as if
    ``a`` and ``b`` were level and ``c`` carried the vacuum -- which
    :func:`_orient_slab` guarantees for generated boards and nothing
    guarantees for anything else.  A board that skips that step still
    looks fine to the slicer: it just gets a *smaller* cell, and every
    number derived from it (area, layer spacing, gamma) is quietly wrong.

    There is deliberately no way to switch this off.  A geometry that
    cannot be measured correctly cannot be reported correctly either, so
    the only honest answer is to refuse the structure.
    """


#: Relative slack on criterion B.  Cell areas are tens of A^2, so 1e-6 is
#: ~1e-5 A^2 -- four orders below the last digit anyone reports, and six
#: orders above the ~1e-3 relative loss a (111) projection produces.
AREA_CONSERVATION_RTOL = 1e-6

#: Half-width of the ``(m, n)`` search for the shortest in-plane vector.
#: 4 covers every combination a coherent match up to ``max_area`` can
#: produce while still costing 80 norm evaluations.
INPLANE_PERIOD_RANGE = 4

#: Relative slack on criterion C, so the gate trips on a real compression
#: and not on the last ulp of a period measured equal to its bound.
INPLANE_PERIOD_RTOL = 1e-9

#: A cross-interface pair is called on-top when its lateral offset is
#: below this (A).  Deliberately a tight absolute bound and not a
#: fraction of the bond scale: the claim the message makes is that the
#: in-plane gradient *vanishes by symmetry*, and that is true only at the
#: symmetric point itself.  A pair 0.1 A off it has a small gradient, not
#: a vanishing one -- a slow climb out of a shallow basin, which is a
#: step-budget concern and not a geometry violation.  Saying "symmetric"
#: about both would make the warning assert something false, which in a
#: package that sells "no trustworthy-looking output from an untrustworthy
#: calculation" is worse than missing the second case.  Coherent lateral
#: offsets are rational fractions of a lattice vector anyway, so they are
#: either exactly zero or of order a fraction of an angstrom.
ON_TOP_LATERAL_TOL = 1e-6

#: Tolerance on the fractional in-plane distance between the join site a
#: registry requested and the one measured after the film was moved.  A
#: placement that misses by more than this is refused outright: the join
#: is a rational fraction of a lattice vector, so a real miss is of order
#: a tenth of a cell and round-off is of order 1e-16 -- the two are ten
#: orders apart and there is nothing in between that a tolerance could
#: legitimately wave through.
REGISTRY_PLACEMENT_TOL = 1e-6

#: Same tolerance applied to a stacking step when deciding whether some
#: multiple of it is a lattice vector, i.e. how many sublattices the
#: stacking sequence resolves.
_STACKING_STEP_TOL = 1e-6

#: The one orientation whose gamma_int can be quoted without scanning the
#: lateral registry first -- and only because the published protocol
#: enumerated ten candidates itself.  Every other orientation is exposed
#: by ``--sub-miller``/``--film-miller`` as a single as-built registry.
PUBLISHED_MILLER: tuple[int, int, int] = (0, 0, 1)

# Strains within 1e-4 (0.01 %) of each other are treated as equally good
# when ranking lattice matches; see ``_match_lattices``.
_STRAIN_TIE_DIGITS = 4


#: z separations below this are the scatter inside one atomic plane rather
#: than a step to the next one.  A built board is unrelaxed, so its planes
#: are flat to ~1e-13; the tightest interplanar spacing the builder can
#: produce is over an angstrom, three orders above this floor.
_PLANE_Z_FLOOR = 1e-3

#: A shift shorter than this is not applied at all.  ``+= 0.0`` is a no-op
#: in exact arithmetic, but it is still a write -- and a write to every
#: film coordinate is a chance to disagree with a pinned baseline in the
#: last bit.  Skipping the operation entirely is what makes "this changed
#: nothing" a structural guarantee instead of a floating-point one.
_OFF_ON_TOP_MIN_SHIFT = 1e-12

#: A measured join offset shorter than this (Å) is the float scatter of an
#: exactly opposed pair, not a site.  Real sites are rational fractions of
#: a lattice vector -- either exactly zero or of order a tenth of an
#: angstrom -- so there is nothing between the two that this could be
#: suppressing. It is snapped to exact zeros before a placement is solved
#: for, so the arithmetic that follows is exact rather than carrying 1e-16
#: into every film coordinate.
_JOIN_ROUND_OFF = 1e-9


def _describe_registry(registry: str | Sequence[float] | None) -> str:
    """Name a requested registry the way an advisory message quotes it.

    A mode and an explicit ``(u, v)`` are not two kinds of thing to a
    reader who has to act on the message, so both are spelled out: the
    name alone would leave "where is the join?" unanswered, and a bare
    pair would leave "was this asked for or inherited?" unanswered.
    """
    if registry is None:
        return "registry=None (the site the two slab cuts happen to carry)"
    if isinstance(registry, str):
        return f"registry={registry!r}"
    values = tuple(float(item) for item in registry)
    if len(values) != 2:
        return f"registry={registry!r}"
    return f"registry=(u={values[0]:.6f}, v={values[1]:.6f})"


def _warn_unscanned_orientation(
    config, lateral: float | None = None, registry: str | Sequence[float] | None = None
) -> None:
    """Say out loud that a non-(001) run ships one unscanned registry.

    Three things the caller cannot see from the output otherwise.

    The registry is now *requested* -- :class:`CoherentInterfaceConfig`
    has a ``registry`` field and the join is placed on the site it names
    -- but asking for one site is not scanning the surface. The other
    sites this orientation resolves are never visited, so a gamma_int
    read off this run is a number for the site that was asked for and
    says nothing about the one a scan would have found lower. What the
    cut would have left is no longer what you get either: a film that
    would land exactly on top of a substrate atom is moved off that site
    by the request, so the offset no longer collapses to on-top as the
    thickness changes: what is left is a single site, decided by the
    request rather than by the physics.

    The geometry gate cannot catch any of this and must not be stretched
    to: a non-(001) board can be level, conserve its area and keep a
    healthy in-plane period while sitting on a high-symmetry registry,
    where the in-plane gradient vanishes by symmetry and the relaxation
    stops where it started -- converged, fmax met, OOD green, valid=True,
    and wrong. The (110) ridge site reads 0.5953 J/m^2 against a scanned
    0.088, 6.8x high, with a shortest contact of 2.27 A that no
    proximity check would flag.

    Relaxing longer does not help and is worth saying so: 900 steps left
    the (110) ridge site bit-identical. The fix is to scan the registry
    and quote that minimum, not to relax harder.

    ``lateral`` is the as-built registry in A, measured once per
    assembly. It is quoted so two runs can be checked for being the same
    run before their gamma_int is compared at all.  ``registry`` is what
    the user asked for; it is quoted because "the join sits where you
    asked" and "the join sits where the cut left it" are the two halves
    of this warning, and a reader can only tell them apart if the
    request is named.
    """
    off_001 = [
        f"{side} {tuple(int(index) for index in miller)}"
        for side, miller in (("substrate", config.substrate_miller), ("film", config.film_miller))
        if tuple(int(index) for index in miller) != PUBLISHED_MILLER
    ]
    if not off_001:
        return

    requested = _describe_registry(registry)
    as_built = (
        f"as-built lateral offset: {lateral:.6f} A (in-plane part of the closest "
        "substrate/film pair; two runs are comparable only if this matches). "
        if lateral is not None
        else ""
    )
    warnings.warn(
        f"[orientation] {' and '.join(off_001)}: this orientation is not (001), so the "
        "lateral registry the builder hands over has not been scanned: you asked for "
        f"{requested} and the join has been built on that site, but one requested "
        "site is not a scan -- every other site of this surface has not been scanned, "
        "and what the cut would have left is no longer what you get: a film that would "
        "sit exactly on top of a substrate atom is moved off that site by the request, "
        "so the registry no longer collapses to on-top as the thickness changes: it is "
        "one site, decided by the request rather than by the physics, and two runs "
        f"still have to be checked for being on it before their gamma_int can be "
        f"compared at all. {as_built}"
        "The site itself is not a geometry violation: it is level, conserves its area "
        "and keeps a healthy in-plane period, but on a high-symmetry registry the "
        "in-plane gradient vanishes by symmetry, so a relaxation stops where it "
        "started -- converged, fmax met, OOD green, valid=True, and wrong. (110) shows "
        "what a site is worth: its ridge site reads 0.5953 J/m^2 against a scanned "
        "0.088, 6.8x high, and registry='auto' (the default) asks for the trough "
        "instead -- naming a site is a choice, and this warning cannot make it for "
        "you. A longer relaxation does not fix it: 900 steps left that site "
        "bit-identical, and only a thickness step that accumulated in-plane shear "
        "moved it. Asking for a registry does not lift this warning either: the other "
        "sites are still unvisited, and only an explicit registry scan (registry="
        "'scan', not implemented in this release) would visit them. Run that scan and "
        "quote its minimum before citing gamma_int for a non-(001) orientation. (001) "
        "is the exception: the published protocol enumerated ten candidates itself, "
        "which is why its value can be quoted directly.",
        stacklevel=2,
    )


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
    registry : str or tuple[float, float] or None, optional
        Which lateral site the join is built on, as fractional
        in-plane coordinates of the substrate's in-plane cell:

        ``None``
            The v0.4.1-and-earlier behaviour, bit for bit: stack the
            boards as the slab cuts left them and move the film only
            when that put it exactly on top of a substrate atom.
        ``"auto"`` (default)
            The site that continues the substrate's stacking sequence,
            i.e. its stacking step away from on-top. On fcc (001) that
            is the site the cuts already give, so every (001) number is
            unchanged; on (110) it is the trough instead of the ridge.
        ``"on-top"``
            ``(0, 0)``: the film's join atom directly above a substrate
            top-layer atom.
        ``"hcp"``
            The *other* hollow of a close-packed surface. Only
            resolvable when the stacking sequence distinguishes three
            sublattices (fcc (111) and its relatives); anywhere else it
            raises, because there the two hollows are the same site and
            the name would be asserting a distinction that does not
            exist.
        ``(u, v)``
            An explicit fractional in-plane offset. Always accepted, and
            the only way to ask for a site this vocabulary has no name
            for.

        Placement is *targeted*: the site is named first and the shift
        that reaches it is computed second, rather than adding a
        stacking step to whatever the cut produced -- on (110) those two
        disagree by a factor of 2.55 in gamma_int. Whatever is asked
        for is measured after the move and a miss is an error, never a
        silent fallback. Asking for one site is still not a registry
        scan; ``"scan"`` is reserved for that and is rejected here.
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
    registry: str | tuple[float, float] | None = "auto"

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
        errors.extend(_registry_errors(self.registry))
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

#: Accepted string values of :attr:`CoherentInterfaceConfig.registry`.
#:
#: Deliberately short, and deliberately missing the site names a surface
#: scientist would use (``trough``, ``ridge``, ``bridge``,
#: ``fcc-hollow``). Those names are orientation- and lattice-specific --
#: fcc (111) resolves three sites at (0,0)/(1/3,1/3)/(2/3,2/3) while
#: fcc (110) resolves four at (0,0)/(1/2,0)/(0,1/2)/(1/2,1/2), and which
#: of the (111) hollows is "fcc" depends on the layer below it -- so no
#: general name parser can exist. What is offered instead is the small
#: set that can always be *computed*, plus an explicit ``(u, v)`` for
#: everything else, and every rejected name is answered with the list
#: rather than guessed at.
REGISTRY_MODES: tuple[str, ...] = ("auto", "on-top", "hcp")


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


def _shortest_inplane_period(in_plane: np.ndarray) -> tuple[float, int, int]:
    """Shortest ``|m*a + n*b|`` over ``m, n`` in ``[-4, 4]``, as ``(length, m, n)``.

    Brute force rather than ``min(|a|, |b|)``: a basis can be reduced in
    area and still carry a *combination* far shorter than either vector --
    ``a = (2.556, 0)``, ``b = (3 * 2.556 + 0.05, 0.3)`` reads
    ``min(|a|, |b|) = 2.556`` but repeats every ``|b - 3 a| = 0.304 A``.
    Reading ``|a|`` alone would call that cell healthy.
    """
    best = (float("inf"), 0, 0)
    for m in range(-INPLANE_PERIOD_RANGE, INPLANE_PERIOD_RANGE + 1):
        for n in range(-INPLANE_PERIOD_RANGE, INPLANE_PERIOD_RANGE + 1):
            if m == 0 and n == 0:
                continue
            length = float(np.linalg.norm(m * in_plane[0] + n * in_plane[1]))
            if length < best[0]:
                best = (length, m, n)
    return best


def _pair_label(slab, i: int, j: int) -> str:
    """``#3 Cu (u=..., v=..., w=...) / #7 Ni (...)`` for two sites.

    The fractional coordinates are the point.  A pair that is too close
    is ambiguous on distance alone: two atoms facing each other across
    the join (same ``u, v``, ``w`` straddling the interface) are the
    intended small gap, while two atoms inside one plane are a cell that
    was compressed.  Only ``w`` separates those, so it is always shown.
    """
    # Wrapped into the cell: the builders keep ``to_unit_cell=False``, so
    # raw fractional coordinates arrive as -6.33 rather than 0.67 and say
    # nothing about where in the stack the pair actually sits.
    coords = np.mod(np.asarray(slab.frac_coords, dtype=np.float64), 1.0)
    species = [str(entry) for entry in slab.species]

    def _site(index: int) -> str:
        u, v, w = coords[index]
        return f"#{index} {species[index]} (u={u:.6f}, v={v:.6f}, w={w:.6f})"

    return f"{_site(i)} / {_site(j)}"


def _check_area_conservation(side: str, before: np.ndarray, after, transform) -> None:
    """Criterion B: the in-plane block the stack uses spans the board's area.

    Replication by an integer supercell multiplies the area by exactly
    ``|det(transform)|``, and a level board's ``[:2, :2]`` block *is* its
    interface plane -- so the area the assembly writes into the interface
    cell has to equal the unscaled board's area times the multiplicity.
    Any gap between the two says ``a`` or ``b`` carries a z component and
    ``[:2, :2]`` is a projection onto xy rather than the plane itself.

    ``before`` is the pre-scaling lattice matrix and ``transform`` the
    supercell description (``None`` when nothing was scaled).
    """
    multiplicity = 1 if transform is None else _multiplicity(transform)
    expected = _lattice_area(np.asarray(before, dtype=np.float64)) * multiplicity
    if not math.isfinite(expected) or expected <= 0.0:
        return

    matrix = np.asarray(after.lattice.matrix, dtype=np.float64)
    # The two areas are only the same number for a level cell; reported
    # separately so the message says which step lost the area.
    true_area = _lattice_area(matrix)
    used_area = float(abs(np.linalg.det(matrix[:2, :2])))
    deviation = max(abs(true_area - expected), abs(used_area - expected)) / expected
    if deviation <= AREA_CONSERVATION_RTOL:
        return

    raise GeometryGateError(
        f"[geometry gate] criterion B (area conservation) failed on the {side} side. "
        f"measured in-plane block = {used_area:.6f} A^2, board's true cell area = "
        f"{true_area:.6f} A^2, reference = {multiplicity} x {expected / multiplicity:.6f} A^2 "
        f"= {expected:.6f} A^2; relative deviation {deviation:.3e} exceeds "
        f"{AREA_CONSERVATION_RTOL:.1e}. "
        f"a = ({matrix[0][0]:.6f}, {matrix[0][1]:.6f}, {matrix[0][2]:.6f}), "
        f"b = ({matrix[1][0]:.6f}, {matrix[1][1]:.6f}, {matrix[1][2]:.6f}): "
        "a or b carries a z component, so lattice[:2, :2] is the xy projection of the "
        "interface plane instead of the plane itself, and every area, spacing and gamma "
        "derived from this stack is scaled by that projection. Level the board first."
    )


def _check_inplane_period(side: str, slab, max_strain: float) -> float:
    """Criterion C: the in-plane cell must still repeat at a physical period.

    The board's tightest pair sets the scale: nothing in a relaxed board
    sits closer than a nearest-neighbour shell, and the coherent match is
    allowed to compress that by at most ``max_strain``.  A period below
    ``d_ref * (1 - max_strain)`` means the cell the stack will repeat was
    squeezed past what the chemistry allows -- which is what a projected
    in-plane block does, since a projection can only shorten vectors.

    Returns the ``d_ref`` it measured so the caller can reuse the scale
    (the proximity warning needs it); NaN when the board is too small to
    establish one.
    """
    from atominterface.relax.proximity import pairwise_min_distance

    if len(slab) < 2:
        return float("nan")
    d_ref, i, j = pairwise_min_distance(_to_ase(slab))
    if not math.isfinite(d_ref):
        return float("nan")

    matrix = np.asarray(slab.lattice.matrix, dtype=np.float64)
    period, m, n = _shortest_inplane_period(matrix[:2, :2])
    threshold = d_ref * (1.0 - float(max_strain))
    if period >= threshold * (1.0 - INPLANE_PERIOD_RTOL):
        return d_ref

    raise GeometryGateError(
        f"[geometry gate] criterion C (in-plane period) failed on the {side} side. "
        f"measured shortest in-plane period = {period:.6f} A "
        f"({m}*a + {n}*b, over m, n in [{-INPLANE_PERIOD_RANGE}, {INPLANE_PERIOD_RANGE}]), "
        f"threshold = {threshold:.6f} A = d_ref {d_ref:.6f} A x (1 - {float(max_strain):.6f}). "
        f"d_ref is the tightest pair in the board: {_pair_label(slab, i, j)}. "
        "A short period can be a compressed cell or an intended small gap -- read the "
        "fractional coordinates: a pair with the same u, v whose w straddles the join is "
        "the gap you asked for, a pair inside one plane is a cell that was squeezed."
    )


class _RegistryPair(NamedTuple):
    """The closest substrate/film pair across the join, and how it is offset.

    ``lateral`` -- the in-plane part of their minimum-image separation --
    is the registry.  It is what a registry scan varies and what two runs
    have to match before their gamma_int can be compared, so it is
    measured once per assembly and reused by both consumers rather than
    derived twice by two different routes that could disagree.
    """

    distance: float
    lateral: float
    substrate_index: int
    film_index: int
    du: float
    dv: float


def _closest_cross_pair(interface, n_substrate: int) -> _RegistryPair | None:
    """Closest substrate/film pair in ``interface``, with its lateral offset.

    ``None`` when one side is empty, so there is no join to measure.
    """
    from atominterface.relax.proximity import nearest_image_vectors, pairwise_distance_matrix

    if n_substrate <= 0 or n_substrate >= len(interface):
        return None
    atoms = _to_ase(interface)
    matrix = pairwise_distance_matrix(atoms)
    cross = matrix[:n_substrate, n_substrate:]
    flat = int(np.argmin(cross))
    i, j = divmod(flat, cross.shape[1])
    distance = float(cross[i, j])

    # Lateral part of the minimum-image separation: the pair's offset
    # within the interface plane, with the normal component removed.
    vectors, _ = nearest_image_vectors(atoms, atoms.positions[i])
    cell = np.asarray(interface.lattice.matrix, dtype=np.float64)
    normal = np.cross(cell[0], cell[1])
    normal = normal / np.linalg.norm(normal)
    separation = vectors[n_substrate + j]
    lateral_vector = separation - float(separation @ normal) * normal
    lateral = float(np.linalg.norm(lateral_vector))
    # Rounded to the precision the message prints, then ``+ 0.0`` to turn
    # the -0.0 a tiny negative component leaves behind into +0.0: an
    # exactly opposed pair has to read "du=0.000000", because "du=-0.000000"
    # invites the reader to go looking for an offset that is not there.
    fit = np.linalg.lstsq(cell[:2].T, lateral_vector, rcond=None)[0]
    du = float(np.round(fit[0], 6)) + 0.0
    dv = float(np.round(fit[1], 6)) + 0.0
    return _RegistryPair(distance, lateral, i, n_substrate + j, du, dv)


def _warn_interface_proximity(
    interface, pair: _RegistryPair | None, d_refs, max_strain: float, gap: float
) -> None:
    """Criterion D, advisory only: is the closest cross-interface pair a bond?

    Deliberately a warning and not a gate.  ``config.gap`` (2.0 A by
    default) is *shorter* than a Cu/Ni bond on purpose -- it hands the
    relaxer a slightly compressed join to resolve -- so "closer than a
    bond" is the normal state of a fresh interface and raising here would
    reject every legitimate one.  It is still worth saying out loud,
    because the same signature means two different things: the join the
    caller asked for, or two boards pushed into each other.  The distance
    cannot separate those; the fractional coordinates can, so they are
    what the message leads with.

    Nor is a small distance the worst thing this can be.  A pair that is
    both close *and* laterally undisplaced is on top of its counterpart,
    and an exactly opposed stack sits on a mirror-symmetric stationary
    point where the in-plane gradient vanishes by symmetry -- the
    relaxation converges there, every numerical check stays green, and
    gamma_int comes out 5.9x high on Cu/Ni(111) and 6.3x high on (110).
    That is why the lateral offset is measured and named rather than
    left for the reader to work out.

    The offset is compared against an absolute tolerance, not against a
    fraction of the bond scale: the gradient vanishes only at the
    symmetric point, and a pair a fraction of an angstrom off it is a
    shallow basin rather than a stationary point.  Calling both
    "symmetric" would make the message assert something false.
    """
    measured = [d for d in d_refs if math.isfinite(d)]
    if not measured or pair is None:
        return
    threshold = min(measured) * (1.0 - float(max_strain))
    if pair.distance >= threshold:
        return
    on_top = pair.lateral <= ON_TOP_LATERAL_TOL

    warnings.warn(
        f"[geometry gate] criterion D (interface proximity) on the assembled interface: "
        f"closest substrate/film pair is {pair.distance:.6f} A, below the {threshold:.6f} A "
        f"contact scale (d_ref {min(measured):.6f} A x (1 - {float(max_strain):.6f})); "
        f"the requested gap was {float(gap):.6f} A. "
        f"pair: {_pair_label(interface, pair.substrate_index, pair.film_index)}. "
        f"lateral offset = {pair.lateral:.6f} A (du={pair.du:.6f}, dv={pair.dv:.6f}). "
        "Advisory, and ambiguous on distance alone: a pair sharing u, v whose w straddles "
        "the join is the gap you asked for and will relax apart; a pair inside one board "
        "means the two sides were pushed into each other. Read w before acting."
        + (
            " This pair is exactly opposed -- zero lateral offset, i.e. on-top registry. "
            "Such a stack sits on a mirror-symmetric stationary point whose in-plane "
            "gradient vanishes by symmetry, so a relaxation can converge there and stay: "
            "it relaxes, passes OOD and looks numerically healthy while gamma_int reads "
            "5.9x high on Cu/Ni(111) (at 14.4 A: 7 Cu against 8 Ni layers leaves the join "
            "on-top and the optimizer stops after a single step) and 6.3x high on (110). "
            "Break the symmetry with any one of: shifting the initial registry; changing "
            "the thickness step so both sides get the same layer count (the 7/8 asymmetry "
            "above is what made the on-top site a stationary point); or running an "
            "explicit registry scan."
            if on_top
            else ""
        ),
        stacklevel=2,
    )


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


def _atomic_planes(z: np.ndarray) -> list[np.ndarray]:
    """Indices of each atomic plane in ``z``, ordered from the bottom up.

    Grouped by a tolerance read off the data itself (half the smallest
    real z step) rather than a fixed constant: a plane is flat, so every
    separation seen below :data:`_PLANE_Z_FLOOR` is scatter inside one
    plane and the rest are steps between them.
    """
    order = np.argsort(z, kind="stable")
    steps = np.diff(z[order])
    real = steps[steps > _PLANE_Z_FLOOR]
    tol = 0.5 * float(np.min(real)) if real.size else 0.0

    planes: list[list[int]] = []
    current = [int(order[0])]
    reference = float(z[current[0]])
    for index in order[1:]:
        if float(z[index]) - reference > tol:
            planes.append(current)
            current = [int(index)]
            reference = float(z[index])
        else:
            current.append(int(index))
    planes.append(current)
    return [np.asarray(plane, dtype=int) for plane in planes]


def _off_on_top_shift(
    sub_coords: np.ndarray, film_coords: np.ndarray, sub_2d: np.ndarray
) -> np.ndarray:
    """In-plane shift that lifts the film off an exactly opposed join (Å).

    Stacking the boards as they come puts the film's join layer on
    whatever sublattice pymatgen's cut left it on and the substrate's on
    the one its own layer count implies -- on fcc (111) that is sublattice
    ``(n_sub - 1) mod 3``, so every third ``--min-thickness`` step lands
    the two on top of each other.  An exactly opposed stack sits on a
    mirror-symmetric stationary point whose in-plane gradient vanishes by
    symmetry, so a relaxation converges there and stays.  Measured on
    Cu/Ni(111): 7 Cu layers at 14.4 A gives gamma_int 0.898 J/m^2 against
    0.081 one thickness step either side.

    The nudge is the substrate's own stacking step -- the in-plane
    offset from its second plane to its top one -- so the film lands on
    the site that continues the substrate's stacking sequence rather
    than on an arbitrary fraction of a cell.  That is the one choice that
    is defined by the geometry at hand and by nothing else.

    Returns zeros when the join is already off the symmetric point, or
    when either board has too few planes to say what "the next one up"
    means.  The caller skips the shift instead of applying a zero one.
    """
    empty = np.zeros(2, dtype=np.float64)
    if not len(sub_coords) or not len(film_coords):
        return empty
    sub_planes = _atomic_planes(sub_coords[:, 2])
    film_planes = _atomic_planes(film_coords[:, 2])
    if len(sub_planes) < 2 or not film_planes:
        return empty

    inv_2d = np.linalg.inv(np.asarray(sub_2d, dtype=np.float64))
    sub_top = sub_coords[sub_planes[-1], :2] @ inv_2d
    sub_below = sub_coords[sub_planes[-2], :2] @ inv_2d
    film_bottom = film_coords[film_planes[0], :2] @ inv_2d

    def _wrapped(delta: np.ndarray) -> np.ndarray:
        return delta - np.round(delta)

    # As-built registry of the join: the closest pair of in-plane offsets
    # between the two facing planes, minimum image, in Å.  This is the
    # same quantity ``_closest_cross_pair`` reports for the stack, just
    # read before the stack exists.
    lateral = min(
        float(np.min(np.linalg.norm(_wrapped(sub_top - point) @ sub_2d, axis=1)))
        for point in film_bottom
    )
    if lateral >= ON_TOP_LATERAL_TOL:
        return empty

    # Forward stacking step: from the nearest atom one plane down to an
    # atom in the top plane.  Wrapped to the shortest representative so
    # the film moves by one step and not by a lattice vector plus one.
    offsets = np.asarray(_wrapped(sub_top[0] - sub_below) @ sub_2d, dtype=np.float64)
    nearest = int(np.argmin(np.linalg.norm(offsets, axis=1)))
    return np.asarray(offsets[nearest], dtype=np.float64)


def _wrap_frac(delta: np.ndarray) -> np.ndarray:
    """Shortest representative of a fractional in-plane offset.

    One lattice vector is as good as no displacement, so every offset is
    reduced to a representative before it is measured, compared or
    applied. Two measured reasons this is load-bearing and not cosmetic:

    * wrapping a *cartesian* difference is meaningless -- the cell is
      generally not cubic, so one lattice vector is not a round number
      of Å -- and it does not raise. It was measured putting a (110)
      film 0.31 of a cell away from the site it was asked for, with
      every downstream check still green and the film sliding during
      relaxation to an energy that had nothing to do with the request.
    * the (001) registry is half a lattice vector, which exists as both
      ``+1/2`` and ``-1/2``. Unwrapped, "move the join onto the site the
      stacking sequence continues to" would move a (001) film by a whole
      lattice vector: a physically identical structure carrying
      different numbers, on the one orientation this package has a DFT
      cross-check for.

    So the order is always fractional -> wrapped -> cartesian, and a
    distance is only ever taken after the third step.
    """
    return delta - np.round(delta)


def _in_plane_frac(vector: np.ndarray, sub_2d: np.ndarray) -> np.ndarray:
    """Cartesian in-plane vector -> fractional coordinates of ``sub_2d``."""
    matrix = np.asarray(sub_2d, dtype=np.float64)
    return np.asarray(np.asarray(vector, dtype=np.float64) @ np.linalg.inv(matrix), float)


def _join_lateral(
    sub_coords: np.ndarray, film_coords: np.ndarray, sub_2d: np.ndarray
) -> np.ndarray | None:
    """As-built in-plane offset of the join: substrate minus film (Å).

    The minimum-image vector from a film atom in the plane facing the
    substrate to the nearest substrate atom in its own facing plane,
    minimised over that film plane; ``None`` when either board is empty
    or has no atomic plane to measure.

    Convert first, wrap second, multiply third: see :func:`_wrap_frac`.
    Reading this number on the two boards before they are stacked is
    what makes targeted placement possible -- it is the same quantity
    :func:`_closest_cross_pair` reports for the finished stack, so the
    request and the measurement cannot drift apart into two different
    definitions of "where the join is".
    """
    if not len(sub_coords) or not len(film_coords):
        return None
    sub_planes = _atomic_planes(sub_coords[:, 2])
    film_planes = _atomic_planes(film_coords[:, 2])
    if not sub_planes or not film_planes:
        return None

    inv_2d = np.linalg.inv(np.asarray(sub_2d, dtype=np.float64))
    sub_top = sub_coords[sub_planes[-1], :2] @ inv_2d
    film_bottom = film_coords[film_planes[0], :2] @ inv_2d
    best: np.ndarray | None = None
    best_norm = math.inf
    for point in film_bottom:
        offsets = np.asarray(_wrap_frac(sub_top - point) @ sub_2d, dtype=np.float64)
        norms = np.linalg.norm(offsets, axis=1)
        nearest = int(np.argmin(norms))
        if float(norms[nearest]) < best_norm:
            best_norm = float(norms[nearest])
            best = np.asarray(offsets[nearest], dtype=np.float64)
    return best


def _stacking_step(sub_coords: np.ndarray, sub_2d: np.ndarray) -> np.ndarray | None:
    """In-plane offset from the substrate's top plane back down one plane.

    Fractional coordinates, wrapped to the shortest representative.
    ``None`` when the board has fewer than two planes, i.e. when "which
    way the sequence is going" is not a question with an answer.

    **Sign.** This is the step *backwards* along the stacking sequence,
    the negative of the forward one :func:`_off_on_top_shift` moves the
    film by, and it is the sign the placement target needs. The two are
    not interchangeable. Written as offsets from an atom in the top
    plane, the site that continues the sequence is at ``+s_up`` (one
    more step up), which is ``-s_down`` subtracted back; the registry
    offset is measured the other way round (substrate minus film), so
    the target that names that site is ``s_down``. On fcc (111) the two
    signs are the difference between the fcc and the hcp hollow, and on
    fcc (110) they coincide only because there every site is half a
    lattice vector -- which is exactly the coincidence that makes a
    wrong sign invisible on the orientation where it was tested.
    """
    if not len(sub_coords):
        return None
    sub_planes = _atomic_planes(sub_coords[:, 2])
    if len(sub_planes) < 2:
        return None
    inv_2d = np.linalg.inv(np.asarray(sub_2d, dtype=np.float64))
    sub_top = sub_coords[sub_planes[-1], :2] @ inv_2d
    sub_below = sub_coords[sub_planes[-2], :2] @ inv_2d
    offsets = _wrap_frac(sub_below - sub_top[0])
    nearest = int(np.argmin(np.linalg.norm(np.asarray(offsets @ sub_2d, float), axis=1)))
    return np.asarray(offsets[nearest], dtype=np.float64)


def _stacking_period(step: np.ndarray) -> int:
    """Layers after which the stacking sequence repeats (0 if none found).

    The number of sublattices the sequence resolves: 2 for a surface
    that alternates, 3 for a close-packed one, 0 for a step that is not
    a rational fraction the search recognises. It decides which site
    names mean anything -- "the other hollow" needs three.
    """
    for count in (1, 2, 3, 4, 6):
        if float(np.max(np.abs(_wrap_frac(count * step)))) < _STACKING_STEP_TOL:
            return count
    return 0


def _sites_of_step(step: np.ndarray, count: int) -> str:
    """Text list of the ``k x step`` sites, in fractional in-plane coords."""
    return ", ".join(
        f"{index}x step -> ({site[0]:+.6f}, {site[1]:+.6f})"
        for index, site in enumerate(_wrap_frac(float(index) * step) for index in range(count))
    )


def _registry_target_frac(
    sub_coords: np.ndarray, sub_2d: np.ndarray, mode: str | Sequence[float]
) -> np.ndarray | None:
    """The site the join is to be placed on, in fractional in-plane coords.

    ``None`` means "this request cannot be resolved for this board",
    which the caller turns into an error: a site that cannot be named is
    not a licence to leave the join where the cut put it.

    The target is a site, not a displacement. That distinction is the
    whole point: the incremental reading ("move the film by the
    stacking step") and this one agree on fcc (111), where the cut
    leaves the film on-top and 0 + step is the fcc hollow, and they
    disagree on fcc (110), where the cut leaves the film on the ridge
    and ridge + step is the short bridge -- 1.5189 J/m^2 against a
    scanned 0.0881, 2.55x the correct target's value.
    """
    if isinstance(mode, str):
        if mode == "on-top":
            return np.zeros(2, dtype=np.float64)
        if mode == "auto":
            step = _stacking_step(sub_coords, sub_2d)
            return None if step is None else np.asarray(_wrap_frac(step), dtype=np.float64)
        if mode == "hcp":
            step = _stacking_step(sub_coords, sub_2d)
            if step is None:
                return None
            period = _stacking_period(step)
            if period != 3:
                raise ValueError(
                    "registry='hcp' names the other hollow of a close-packed surface, "
                    "which is only a different atom when the stacking sequence "
                    "resolves three sublattices. This one's stacking step is "
                    f"(u, v) = ({step[0]:+.6f}, {step[1]:+.6f}) and repeats after "
                    f"{period if period else 'more than 6'} layer(s), so the sites it "
                    f"resolves are: {_sites_of_step(step, max(period, 2))} -- 'hcp' and "
                    "'auto' would land on the same atom. Use registry='auto', "
                    "registry='on-top' or an explicit (u, v)."
                )
            return np.asarray(_wrap_frac(-step), dtype=np.float64)
        raise ValueError(
            f"registry must be one of {list(REGISTRY_MODES)}, or an explicit (u, v) "
            f"fractional in-plane offset; got {mode!r}"
        )
    values = np.asarray([float(item) for item in mode], dtype=np.float64)
    if values.shape != (2,):
        raise ValueError(
            f"registry (u, v) must have exactly 2 components, got {values.size}: {mode!r}"
        )
    return np.asarray(_wrap_frac(values), dtype=np.float64)


def _registry_shift(
    sub_coords: np.ndarray,
    film_coords: np.ndarray,
    sub_2d: np.ndarray,
    mode: str | Sequence[float],
) -> np.ndarray:
    """In-plane displacement that lands the join on the requested site (Å).

    Targeted, so the arithmetic is "where it is minus where it was asked
    to be": after ``film += shift`` the join reads ``d - shift``, and
    ``d - shift = target`` is the condition being solved. The result is
    wrapped to its shortest representative, which is what keeps a (001)
    request from moving the film by a whole lattice vector.

    Raises when either the target or the as-built offset cannot be
    measured. It never returns zeros as a way of saying "I could not
    tell": a join whose site is unknown is reported as unknown, not
    quietly built at the cut's position and shipped as if it were the
    answer.
    """
    target_frac = _registry_target_frac(sub_coords, sub_2d, mode)
    if target_frac is None:
        raise ValueError(
            f"{_describe_registry(mode)} was requested, but the site it names cannot "
            "be resolved for this board: the substrate does not have two atomic "
            "planes to read a stacking step from. Place the join with an explicit "
            "(u, v) offset, or drop the request -- this package does not build a "
            "structure at a site it cannot name."
        )
    as_built = _join_lateral(sub_coords, film_coords, sub_2d)
    if as_built is None:
        raise ValueError(
            f"{_describe_registry(mode)} was requested, but the join's as-built "
            "offset cannot be measured: one of the two boards has no atomic plane at "
            "the join. Nothing is being placed."
        )
    # An offset below the floor is not a site, it is the float scatter of an
    # exactly opposed pair: both boards were cut from the same fractional
    # coordinates and the difference is the 1e-16 of a cartesian round trip.
    # Snapping it to exact zeros before the subtraction is what makes "this
    # request moved the film onto the site the previous release produced" a
    # bit-exact statement instead of one that differs in the last bit.
    measured = np.zeros(2, dtype=np.float64)
    if float(np.linalg.norm(as_built)) >= _JOIN_ROUND_OFF:
        measured = _in_plane_frac(as_built, sub_2d)
    shift_frac = _wrap_frac(measured - target_frac)
    return np.asarray(shift_frac @ sub_2d, dtype=np.float64)


def _check_registry_placement(
    as_built: np.ndarray | None,
    target_frac: np.ndarray | None,
    request: str,
    sub_2d: np.ndarray,
) -> None:
    """Re-measure the join after the move; refuse to ship a miss.

    Placement is verified against a fresh measurement rather than
    trusted, because the failure mode this package exists for is a
    structure that is self-consistent and wrong. The miss it is looking
    for has been observed: a cartesian wrap put a (110) film 0.31 of a
    cell off its requested site, the relaxation converged, fmax was met,
    OOD stayed green and the reported energy belonged to no site anyone
    had asked for.

    The comparison is done in fractional coordinates and wrapped before
    it is subtracted, so half-a-cell and minus-half-a-cell -- the same
    atom -- never read as a whole cell of error.
    """
    if as_built is None or target_frac is None:
        raise ValueError(
            f"{request} was requested, but the join could not be re-measured after "
            "the film was moved, so there is no evidence it sits on the requested "
            "site. Refusing to report it."
        )
    placed_frac = _in_plane_frac(as_built, sub_2d)
    deviation = _wrap_frac(placed_frac - target_frac)
    worst = float(np.max(np.abs(deviation)))
    if worst <= REGISTRY_PLACEMENT_TOL:
        return
    cartesian = np.asarray(deviation @ sub_2d, dtype=np.float64)
    raise ValueError(
        f"{request} was requested, but the join landed "
        f"{float(np.linalg.norm(cartesian)):.6f} A from it and the structure is not "
        "being built: requested site (u, v) = "
        f"({target_frac[0]:+.6f}, {target_frac[1]:+.6f}), measured "
        f"({placed_frac[0]:+.6f}, {placed_frac[1]:+.6f}), deviation "
        f"({deviation[0]:+.3e}, {deviation[1]:+.3e}) fractional. A relaxation would "
        "converge on the wrong site and look healthy doing it."
    )


def _registry_errors(value: str | tuple[float, float] | list[float] | None) -> list[str]:
    """Validation errors for a requested :attr:`CoherentInterfaceConfig.registry`.

    Every rejection names what could have been asked for instead: a
    message that only says "bad registry" leaves the user to guess, and
    the plausible guesses (``trough``, ``ridge``, ``fcc-hollow``) are
    exactly the names this vocabulary deliberately refuses to parse.
    """
    if isinstance(value, str):
        errors: list[str] = []
        if value not in REGISTRY_MODES:
            errors.append(
                f"registry must be one of {list(REGISTRY_MODES)}, or a (u, v) explicit "
                f"fractional in-plane offset, or None; got {value!r}"
            )
        if value == "scan":
            errors.append(
                "registry='scan' is reserved for a later release: it will build one "
                "candidate per distinguishable site of the surface. It is not "
                "implemented yet, so it cannot be requested"
            )
        return errors
    if value is None:
        return []
    try:
        components = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return [
            f"registry must be one of {list(REGISTRY_MODES)}, or a (u, v) pair of "
            f"finite floats, or None; got {value!r}"
        ]
    if len(components) != 2:
        return [f"registry (u, v) must have exactly 2 components, got {len(components)}: {value!r}"]
    if not all(math.isfinite(item) for item in components):
        return [f"registry (u, v) must be finite floats, got {value!r}"]
    return []


def _registry_as_json(value: str | tuple[float, float] | None) -> str | list[float] | None:
    """A requested registry in a form a JSON report can carry.

    Tuples survive ``json.dumps`` as arrays, but a tuple in a metadata
    dict is a value that round-trips into a list and then compares
    unequal to the tuple the config holds; normalising here is what
    keeps a recorded run and its replay identical.
    """
    if value is None or isinstance(value, str):
        return value
    return [float(item) for item in value]


def _frac_or_none(value: Sequence[float] | None) -> list[float] | None:
    """A fractional in-plane pair from the interface properties, or ``None``."""
    if value is None:
        return None
    return [float(item) for item in value]


def _registry_lateral_ang(record: dict[str, Any] | None) -> float | None:
    """Lateral part of a recorded as-built registry, or ``None``."""
    if not record:
        return None
    return float(record["lateral"])


def _place_join(
    sub_coords: np.ndarray,
    film_coords: np.ndarray,
    sub_2d: np.ndarray,
    mode: str | Sequence[float] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Put the join on the site the configuration asks for (Å), then check.

    Mutates ``film_coords[:, :2]`` in place and returns ``(target_frac,
    as_built)`` -- the site that was resolved and the site the join
    measured at afterwards, either of which may be ``None`` when the
    request did not name a site.

    Two placements, and only one of them is a choice of site.
    ``registry=None`` is the v0.4.1 path, kept bit for bit: nudge the
    film off an exactly opposed join and otherwise leave the cut alone.
    Anything else names a site first and solves for the displacement
    that reaches it, which is the only reading that is correct on
    (110) -- there the incremental "add the stacking step" arrives at
    the short bridge, 1.5189 J/m^2 against the trough's 0.0881, while
    the as-built ridge reads 0.5952.

    The check runs on a measurement taken *after* the move, so it would
    catch a placement that solved the wrong problem, and it compares in
    fractional coordinates so that +1/2 and -1/2 -- one atom -- never
    read as a whole cell of error.
    """
    target_frac: np.ndarray | None = None
    if mode is None:
        shift = _off_on_top_shift(sub_coords, film_coords, sub_2d)
    else:
        target_frac = _registry_target_frac(sub_coords, sub_2d, mode)
        shift = _registry_shift(sub_coords, film_coords, sub_2d, mode)
    # A shift below the floor is skipped entirely rather than applied as
    # zero: a write to every film coordinate is a chance to disagree
    # with a pinned baseline in the last bit, and skipping it is what
    # makes "this changed nothing" structural instead of floating-point.
    if float(np.linalg.norm(shift)) >= _OFF_ON_TOP_MIN_SHIFT:
        film_coords[:, :2] += shift
    as_built = _join_lateral(sub_coords, film_coords, sub_2d)
    if target_frac is not None:
        _check_registry_placement(as_built, target_frac, _describe_registry(mode), sub_2d)
    return target_frac, as_built


def _record_registry_properties(
    interface, target_frac: np.ndarray | None, as_built: np.ndarray | None, sub_2d: np.ndarray
) -> None:
    """Write what was asked for and where the join actually is onto the stack.

    Three things a reader needs and only assembly can supply. Kept as
    separate entries because they are three separate claims: a request
    that names a site and a structure that sits on it are different
    facts, and collapsing them into the single number
    ``registry_lateral_offset`` carried before would make a request that
    was not honoured invisible.
    """
    if as_built is not None:
        as_built_frac = _in_plane_frac(as_built, sub_2d)
        interface.properties["registry_as_built"] = {
            "frac": [float(as_built_frac[0]), float(as_built_frac[1])],
            "lateral": float(np.linalg.norm(as_built)),
        }
    if target_frac is not None:
        interface.properties["registry_target_frac"] = [
            float(target_frac[0]),
            float(target_frac[1]),
        ]


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


def measure_registry(atoms: Any) -> dict[str, Any] | None:
    """Measure the lateral registry an interface currently sits on.

    The same quantity :func:`_assemble_interface` places and records,
    re-measured on whatever structure is handed in -- which is what
    makes it usable *after* a relaxation, and after a relaxation is
    where it matters: the archive holds a Cu/Ni(110) run at 21.6 A that
    slid from the ridge to the trough on its own, so an as-built site
    reported without a final one would let a reader take "where it was
    put" for "where it stopped".

    The two sides are told apart by the ``SLAB_ID`` array every
    assembled interface carries (``0`` substrate, ``1`` film), and the
    measurement is the plane-to-plane one used during assembly rather
    than a second route to the same number: two definitions of "the
    registry" would be two chances to disagree about what a run
    shipped.

    Parameters
    ----------
    atoms : ase.Atoms
        An assembled interface carrying a ``SLAB_ID`` array.

    Returns
    -------
    dict or None
        ``{"frac": [du, dv], "lateral": A}`` -- the join's in-plane
        offset in fractional coordinates of the interface cell and its
        length in Å -- or ``None`` when the registry cannot be
        measured: no ``SLAB_ID`` array, a side missing, or a board with
        no atomic plane at the join. "Not measurable" is a fact and is
        reported as one; it is never turned into a zero.
    """
    arrays = getattr(atoms, "arrays", None)
    if not arrays or SLAB_ID not in arrays:
        return None
    tags = np.asarray(arrays[SLAB_ID], dtype=np.int64)
    substrate = tags == 0
    film = tags == 1
    if not bool(np.any(substrate)) or not bool(np.any(film)):
        return None
    positions = np.asarray(atoms.positions, dtype=np.float64)
    cell = np.asarray(atoms.cell, dtype=np.float64)
    if cell.shape != (3, 3) or abs(float(np.linalg.det(cell[:2, :2]))) <= 1e-12:
        return None
    offset = _join_lateral(positions[substrate], positions[film], cell[:2, :2])
    if offset is None:
        return None
    frac = _in_plane_frac(offset, cell[:2, :2])
    return {
        "frac": [float(frac[0]), float(frac[1])],
        "lateral": float(np.linalg.norm(offset)),
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


#: How many distinct assembly-failure messages a build report keeps, and how
#: wide each one may be.  A registry request this surface cannot name fails
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
                    candidate = self._build_candidate(match, sub_idx, sub_slab, film_idx, film_slab)
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

    def _build_candidate(self, match, sub_idx, sub_slab, film_idx, film_slab):
        report = self.last_build_report
        report["attempted"] += 1
        # Validate charge provenance outside the geometry error handler.
        for slab in (sub_slab, film_slab):
            _prepare_charge_structure(slab)
        try:
            interface = _assemble_interface(sub_slab, film_slab, match, self.config)
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


# Register the coherent builder eagerly so `list-builders` works
# without importing the class explicitly.
try:
    from atominterface.builders.registry import builder_registry as _registry

    _registry.register(CoherentInterfaceBuilder.name, CoherentInterfaceBuilder)
except Exception:  # noqa: BLE001
    pass
