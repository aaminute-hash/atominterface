"""Builder configuration for the coherent interface builder.

``CoherentInterfaceConfig`` lives here on its own so ``match`` and
``assembly`` can import it without pulling the builder (and its registry
decorator) into the dependency graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from atominterface._coherent._constants import AREA_POLICIES
from atominterface._coherent.placement import _registry_errors
from atominterface.builders.base import BuilderConfig
from atominterface.builders.charge import validate_charge_options


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
