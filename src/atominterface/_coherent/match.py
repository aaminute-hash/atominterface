"""Lattice matching for the coherent builder.

ZSL enumeration with a brute-force commensurate fallback.  Depends on
:mod:`atominterface._coherent.cells` (one way) and the strain helpers, and
keeps ``CoherentInterfaceConfig`` a ``TYPE_CHECKING``-only import so the
dependency graph stays acyclic while that class still lives in the facade.
"""

from __future__ import annotations

import math
import warnings
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from atominterface._coherent._constants import _STRAIN_TIE_DIGITS
from atominterface._coherent.cells import _as_pymatgen_lattice, _supercell_transform
from atominterface.builders.strain import max_shear_strain, max_stretch_strain

if TYPE_CHECKING:
    from atominterface.builders.coherent import CoherentInterfaceConfig


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


def _match_lattices(
    sub_lattice,
    film_lattice,
    config: CoherentInterfaceConfig,
    strain_limit=None,
    label: str = "",
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
    label
        Optional identity string of the pair being matched (e.g. the
        two Miller index sets, which this function cannot see).  It is
        only surfaced in the ZSL-fallback warning so the operator can
        tell *which* pair silently changed enumeration path.
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
    except Exception as exc:  # noqa: BLE001
        # ZSL is best-effort, but the fallback below is a *different*
        # enumeration: the matches it returns (and their strains) can
        # differ as a whole, which silently changes physical results.
        # Warn with an identifying label so the switch is never silent.
        where = f" for {label}" if label else ""
        warnings.warn(
            f"ZSL lattice matching failed{where} "
            f"({type(exc).__name__}: {exc}); falling back to brute-force "
            "commensurate search -- matches and strains may differ from "
            "the ZSL path",
            stacklevel=2,
        )

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
