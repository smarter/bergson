"""Custom OmegaConf resolvers for DVC Hydra composition."""

from omegaconf import OmegaConf


def _if_else(condition: bool, if_true: str, if_false: str) -> str:
    """Conditional resolver: returns if_true when condition is true, else if_false."""
    return if_true if condition else if_false


def _if_not_empty(value: str, prefix: str = "", suffix: str = "") -> str:
    """Returns prefix + value + suffix if value is not empty, else empty string."""
    if value:
        return f"{prefix}{value}{suffix}"
    return ""


# Register resolvers
OmegaConf.register_new_resolver("if", _if_else, replace=True)
OmegaConf.register_new_resolver("if_not_empty", _if_not_empty, replace=True)
