"""Nested-dict flattening utilities for policy observations/actions."""

from __future__ import annotations

from typing import Any


def recursive_flatten(obj: Any, prefix: str = "", sep: str = "-") -> dict[str, Any]:
    """Flatten nested dictionaries with ``sep``-joined paths.

    Example: ``{"left": {"joint_pos": x}}`` becomes
    ``{"left-joint_pos": x}``.
    """
    if not isinstance(obj, dict):
        return {prefix: obj} if prefix else {"": obj}

    flat: dict[str, Any] = {}
    for key, value in obj.items():
        next_key = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(recursive_flatten(value, next_key, sep=sep))
        else:
            flat[next_key] = value
    return flat


# Historical misspelling kept for the diffusion agent API.
def recusive_flatten(obj: Any, prefix: str = "", sep: str = "-") -> dict[str, Any]:
    return recursive_flatten(obj, prefix=prefix, sep=sep)


def reverse_flatten(flat: dict[str, Any], sep: str = "-") -> dict[str, Any]:
    """Invert :func:`recursive_flatten` for dictionary keys."""
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        parts = str(key).split(sep) if key else [""]
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return nested
