"""Reproducible run configurations.

A *run configuration* is a plain YAML mapping of the parameters that
steer one run (today: ``atominterface build``).  Where
:mod:`atominterface.provenance` records a run *after the fact* into
``metadata.yaml``, this module is the other half of the loop: the
effective parameters of a run are written with :func:`save_run_config`
and can later be *replayed* by loading them back with
:func:`load_run_config` and feeding them to the CLI (``--config``).

The on-disk format is deliberately boring: one flat YAML mapping with
YAML-native scalars, lists and ``null``.  Non-YAML-native Python values
are normalised on save (``tuple`` -> ``list``, ``Path`` -> ``str``) so
``yaml.safe_dump`` / ``yaml.safe_load`` round-trip losslessly and no
custom tags ever appear in a recorded configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["load_run_config", "save_run_config"]


def _yaml_module():
    """Import pyyaml lazily: it is a dev dependency, not a runtime one."""
    try:
        import yaml  # type: ignore[import-untyped]  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "run configurations are YAML files and need pyyaml: pip install pyyaml"
        ) from exc
    return yaml


def _normalize(value: Any) -> Any:
    """Convert a value into YAML-safe containers without losing information.

    ``tuple`` becomes ``list`` (YAML has no tuple type; loading a dumped
    tuple through ``safe_load`` yields a list), and ``Path`` becomes its
    string form.  Everything else -- ``str``, ``int``, ``float``,
    ``bool``, ``None`` -- is already YAML-native and passed through.
    """
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def save_run_config(params: dict[str, Any], path: str | Path) -> Path:
    """Serialize run parameters to a YAML file and return the written path.

    ``params`` must be a mapping; its values are normalised with
    :func:`_normalize` before dumping, so tuples and paths are safe to
    pass.  A suffix other than ``.yaml`` / ``.yml`` is replaced with
    ``.yaml`` so a recorded configuration is always recognisable.  Parent
    directories are created as needed.
    """
    if not isinstance(params, dict):
        raise TypeError(f"run parameters must be a mapping, got {type(params).__name__}")
    yaml = _yaml_module()
    target = Path(path)
    if target.suffix not in (".yaml", ".yml"):
        target = target.with_suffix(".yaml")
    target.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(_normalize(params), sort_keys=False, default_flow_style=False)
    target.write_text(text, encoding="utf-8")
    return target


def load_run_config(path: str | Path) -> dict[str, Any]:
    """Read a run configuration back into a plain dict.

    The top level of the file must be a mapping; anything else (a list,
    a scalar) is a usage error and raises :class:`ValueError`.  An empty
    file loads as an empty mapping, i.e. "no overrides".  Tuples saved
    by :func:`save_run_config` come back as lists -- callers that need
    tuples (e.g. Miller indices) convert at the point of use.
    """
    yaml = _yaml_module()
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"run configuration not found: {source}")
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"run configuration must be a mapping, got {type(data).__name__}: {source}"
        )
    return data
