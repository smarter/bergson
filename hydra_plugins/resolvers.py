"""Custom OmegaConf resolvers for DVC Hydra composition."""

import json
from typing import Any, Optional

from omegaconf import OmegaConf


def _if_else(condition: bool, if_true: str, if_false: str) -> str:
    """Conditional resolver: returns if_true when condition is true, else if_false."""
    return if_true if condition else if_false


def _if_not_empty(value: str, prefix: str = "", suffix: str = "") -> str:
    """Returns prefix + value + suffix if value is not empty, else empty string."""
    if value:
        return f"{prefix}{value}{suffix}"
    return ""


def _if_set(value: Optional[Any], flag: str) -> str:
    """Returns 'flag value' if value is set (not None/null and > 0), else empty string."""
    if value is not None and value:
        return f"{flag} {value}"
    return ""


def _json(value: Any) -> str:
    """Serialize value to JSON string."""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return json.dumps(value)


# Register resolvers
OmegaConf.register_new_resolver("if", _if_else, replace=True)
OmegaConf.register_new_resolver("if_not_empty", _if_not_empty, replace=True)
OmegaConf.register_new_resolver("if_set", _if_set, replace=True)
OmegaConf.register_new_resolver("json", _json, replace=True)
