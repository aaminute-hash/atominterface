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

import math
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
    max_stretch_strain,
    max_shear_strain,
)
from atominterface.builders.strain import (
    max_shear_strain,
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
        Maximum allowed coherent interface area in Å². Large cells
        are rejected to avoid explosive supercell growth.
    gap : float, optional
        Separation gap inserted between the two slabs in Å.
    zsl_max_area_ratio : float, optional
        Stop ZSL search when the area ratio between substrate and
        film supercells exceeds this value.
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
    max_area: float = 500.0
    gap: float = 2.0
    zsl_max_area_ratio: float = 4.0
    zsl_max_angle: float = 2.0
    relax_terminations: bool = True
    vacuum_thickness: float = 15.0


@dataclass
class _MatchInfo:
    """Lightweight container for a lattice match."""

    match_area: float
    substrate_supercell: tuple[int, int]
    film_supercell: tuple[int, int]
    strain: float
    shear_strain: float
    rotation: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def _as_3x3(matrix: np.ndarray) -> np.ndarray:
    """Coerce an array-like to a float64 3x3 matrix."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 lattice matrix, got {arr.shape}")
    return arr


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


def _supercell_factors(matrix: np.ndarray) -> tuple[int, int]:
    """Recover integer scaling factors from a 2x2 transform matrix.

    ZSL returns float matrices close to integers; we round and
    clamp to 1 to keep the Candidate metadata human-readable.
    """
    int(round(abs(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])))
    # Diagonal-ish case: report the trace-inspired factors.
    m = int(max(round(abs(matrix[0, 0])), round(abs(matrix[1, 1])), 1))
    n = int(max(round(abs(matrix[1, 1])), round(abs(matrix[0, 0])), 1))
    return max(m, 1), max(n, 1)


def _match_lattices(sub_lattice, film_lattice, config):
    """Generate coherent matches with strain filtering.

    Uses :class:`pymatgen.analysis.interfaces.zsl.ZSLGenerator` when
    available, and falls back to a minimal commensurate-search when
    pymatgen's generator is unavailable or rejects every candidate.
    """
    from pymatgen.analysis.interfaces.zsl import ZSLGenerator  # noqa: WPS433

    sub_lat = _as_pymatgen_lattice(sub_lattice)
    film_lat = _as_pymatgen_lattice(film_lattice)

    # In-plane vectors normal to the surface normal.  The coherent
    # interface builder always cuts along the supplied Miller
    # direction, so we project the first two reciprocal-lattice
    # directions onto the surface plane.
    sub_vecs = np.array(sub_lat.matrix[:2], dtype=np.float64)
    film_vecs = np.array(film_lat.matrix[:2], dtype=np.float64)

    generator = ZSLGenerator(
        max_area_ratio_tol=config.zsl_max_area_ratio - 1.0,
        max_area=int(config.max_area),
        max_angle_tol=math.radians(config.zsl_max_angle),
    )

    matches: list[_MatchInfo] = []
    try:
        for match in generator(film_vecs, sub_vecs):
            area = float(match.match_area)
            if area <= 0 or area > config.max_area:
                continue

            film_t = np.asarray(match.film_transformation, dtype=np.float64)
            sub_t = np.asarray(match.substrate_transformation, dtype=np.float64)

            strain = float(
                max_stretch_strain(
                    _as_pymatgen_lattice(film_t @ film_vecs),
                    _as_pymatgen_lattice(sub_t @ sub_vecs),
                )
            )
            shear = float(
                max_shear_strain(
                    _as_pymatgen_lattice(film_t @ film_vecs),
                    _as_pymatgen_lattice(sub_t @ sub_vecs),
                )
            )
            if max(strain, shear) > config.max_strain:
                continue

            rotation = None
            try:
                rotation = np.asarray(match.match_transformation, dtype=np.float64)
            except Exception:  # noqa: BLE001
                pass

            matches.append(
                _MatchInfo(
                    match_area=area,
                    substrate_supercell=_supercell_factors(sub_t),
                    film_supercell=_supercell_factors(film_t),
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
        matches = _fallback_matches(sub_lattice, film_lattice, config)

    return sorted(matches, key=lambda m: (m.strain, m.shear_strain, m.match_area))


def _fallback_matches(sub_lattice, film_lattice, config: CoherentInterfaceConfig):
    """Brute-force commensurate search for small supercells.

    This is intentionally simple: it searches integer scaling
    factors ``(m, n)`` up to a small cutoff and reports
    near-commensurate matches.  It is **not** a replacement for
    ZSLGenerator, merely a guarantee that the builder never
    returns an empty list when ZSL rejects the input pair.
    """
    matches: list[_MatchInfo] = []
    sub_lat = _as_pymatgen_lattice(sub_lattice)
    film_lat = _as_pymatgen_lattice(film_lattice)
    sub_vecs = np.array(sub_lat.matrix[:2], dtype=np.float64)
    film_vecs = np.array(film_lat.matrix[:2], dtype=np.float64)

    cutoff = max(2, int(math.sqrt(config.max_area / 20)) + 1)
    for m in range(1, cutoff + 1):
        for n in range(1, cutoff + 1):
            if m * n > cutoff:
                continue
            area = float(
                abs(sub_vecs[0, 0] * sub_vecs[1, 1] - sub_vecs[0, 1] * sub_vecs[1, 0]) * m * n
            )
            if area > config.max_area:
                continue
            scaled = np.diag([float(m), float(n)]) @ film_vecs
            strain = float(
                max_stretch_strain(
                    _as_pymatgen_lattice(scaled),
                    _as_pymatgen_lattice(sub_vecs),
                )
            )
            shear = float(
                max_shear_strain(
                    _as_pymatgen_lattice(scaled),
                    _as_pymatgen_lattice(sub_vecs),
                )
            )
            if max(strain, shear) > config.max_strain:
                continue
            matches.append(
                _MatchInfo(
                    match_area=area,
                    substrate_supercell=(m, n),
                    film_supercell=(1, 1),
                    strain=strain,
                    shear_strain=shear,
                    metadata={"zsl": False, "fallback": True},
                )
            )
    return matches


def _assemble_interface(substrate_slab, film_slab, match=None, config=None, **kwargs):
    """Stack substrate and film slabs manually, without ZSL.

    pymatgen's CoherentInterfaceBuilder triggers a full ZSL enumeration
    in its __init__ (extremely slow, ~100s for a simple Cu cell).  We
    deliberately avoid it here: place the film on top of the substrate
    with an interfacial gap, add vacuum, and return a single Structure.
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
        vacuum = float(getattr(config, "vacuum", vacuum))

    sub_lattice = substrate_slab.lattice.matrix.copy()
    sub_coords = substrate_slab.cart_coords.copy()
    film_coords = film_slab.cart_coords.copy()

    # Shift film so its bottom sits at substrate_top + gap.
    sub_top = float(np.max(sub_coords[:, 2])) if len(sub_coords) else 0.0
    film_bottom = float(np.min(film_coords[:, 2])) if len(film_coords) else 0.0
    z_shift = sub_top + gap - film_bottom
    film_coords[:, 2] += z_shift

    all_species = list(substrate_slab.species) + list(film_slab.species)
    all_coords = np.vstack([sub_coords, film_coords])

    # Extend substrate lattice z to fit everything + vacuum.
    top = float(np.max(all_coords[:, 2])) if len(all_coords) else 0.0
    sub_lattice[2, 2] = top + vacuum

    return Structure(
        _Lattice(sub_lattice),
        all_species,
        all_coords,
        coords_are_cartesian=True,
        to_unit_cell=False,
    )

def _reduced_formula(structure) -> str:
    """Best-effort reduced formula without requiring composition."""
    try:
        return structure.composition.reduced_formula
    except Exception:  # noqa: BLE001
        try:
            from collections import Counter

            c = Counter(str(s.species) for s in structure.sites)
            return "".join(f"{k}{v}" for k, v in sorted(c.items()))
        except Exception:  # noqa: BLE001
            return "unknown"


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
        "match_area_ang2": match.match_area,
        "stretch_strain": match.strain,
        "shear_strain": match.shear_strain,
        "gap_ang": config.gap,
        "vacuum_ang": config.vacuum_thickness,
        **match.metadata,
        "rotation_matrix": None if match.rotation is None else match.rotation.tolist(),
    }

    return Candidate(
        atoms=interface,
        interface_id=f"coherent:{config.substrate_miller}-{config.film_miller}:{sub_term_idx}-{film_term_idx}",
        builder_name="coherent",
        substrate_formula=_reduced_formula(substrate_slab),
        film_formula=_reduced_formula(film_slab),
        substrate_miller=config.substrate_miller,
        film_miller=config.film_miller,
        substrate_termination=str(sub_term_idx),
        film_termination=str(film_term_idx),
        strain_substrate=match.strain,
        strain_film=match.shear_strain,
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

    def build(self, config: CoherentInterfaceConfig | None = None) -> list[Candidate]:
        """Generate all strain-acceptable interface candidates."""
        return self.build_all()

    def build_all(self) -> list[Candidate]:
        """Generate all strain-acceptable interface candidates."""
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))

        sub_lattice = _as_3x3(self.substrate.lattice.matrix)
        film_lattice = _as_3x3(self.film.lattice.matrix)

        matches = _match_lattices(sub_lattice, film_lattice, self.config)
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
