"""Tolerances, registry/area policy constants and pymatgen bindings."""

from __future__ import annotations

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
