"""Shared helpers for the route handlers."""
from __future__ import annotations

from fastapi import HTTPException

from ..registry import ROLE_NAMES, registry


def require_known_engine(name: str) -> None:
    """400 unless ``name`` is an installed engine in some role. The selection and
    option routes gate on this before persisting, so an unknown engine name can't
    be saved into the selection/state."""
    if not any(registry.has(role, name) for role in ROLE_NAMES):
        raise HTTPException(status_code=400, detail=f"unknown engine: {name}")


def require_role_engine(role: str, name: str) -> None:
    """400 unless ``name`` is an installed engine IN THAT ROLE. Option routes gate
    on this so a role-scoped override can't be stored under a role the engine
    doesn't serve."""
    if role not in ROLE_NAMES or not registry.has(role, name):
        raise HTTPException(status_code=400, detail=f"unknown {role}: {name}")
