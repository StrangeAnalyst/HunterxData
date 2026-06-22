"""Config loading utilities for the Markov cluster-transition pipeline.

The YAML at ``config/markov_config.yaml`` is the single source of truth. No
business number is hardcoded in Python; everything is read through here so that
Risk / Finance / Model Governance can audit and override centrally.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "markov_config.yaml"
)

#: Environment variable that, when set, overrides the default config location.
CONFIG_PATH_ENV = "MARKOV_CONFIG_PATH"


def resolve_config_path(config_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the config path from an explicit arg, env var, or default.

    Args:
        config_path: Explicit path. If ``None``, falls back to the
            ``MARKOV_CONFIG_PATH`` environment variable, then the packaged
            default.

    Returns:
        The resolved, existing config path.

    Raises:
        FileNotFoundError: If no config file exists at the resolved location.
    """
    if config_path is not None:
        path = Path(config_path)
    elif os.environ.get(CONFIG_PATH_ENV):
        path = Path(os.environ[CONFIG_PATH_ENV])
    else:
        path = DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Markov config not found at: {path}")
    return path


@lru_cache(maxsize=8)
def _load_yaml(path_str: str) -> Mapping[str, Any]:
    with open(path_str, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the Markov pipeline config as a plain dict.

    Args:
        config_path: Optional explicit path; see :func:`resolve_config_path`.

    Returns:
        Parsed configuration mapping.
    """
    path = resolve_config_path(config_path)
    # Copy out of the lru_cache so callers cannot mutate the cached object.
    return dict(_load_yaml(str(path)))


def get_section(config: Mapping[str, Any], dotted_key: str) -> Any:
    """Fetch a nested config value via a dotted path (e.g. ``"mrp.gamma"``).

    Args:
        config: The loaded config mapping.
        dotted_key: Dot-delimited key path.

    Returns:
        The value at the given path.

    Raises:
        KeyError: If any segment of the path is missing.
    """
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(f"Missing config key segment '{part}' in '{dotted_key}'")
        node = node[part]
    return node


@dataclass(frozen=True)
class HorizonSpec:
    """Typed view of a single horizon entry from config."""

    name: str
    months_ahead: int
    window_months: int
    required_months: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HorizonSpec":
        persistence = raw["persistence"]
        return cls(
            name=str(raw["name"]),
            months_ahead=int(raw["months_ahead"]),
            window_months=int(persistence["window_months"]),
            required_months=int(persistence["required_months"]),
        )


def get_horizons(config: Mapping[str, Any]) -> list[HorizonSpec]:
    """Return the configured horizons as typed :class:`HorizonSpec` objects."""
    return [HorizonSpec.from_mapping(h) for h in config["horizons"]]


def get_horizon(config: Mapping[str, Any], name: str) -> HorizonSpec:
    """Return a single horizon spec by name (e.g. ``"6m"``)."""
    for horizon in get_horizons(config):
        if horizon.name == name:
            return horizon
    raise KeyError(f"Horizon '{name}' not found in config")
