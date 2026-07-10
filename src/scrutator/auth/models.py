"""Tenant identity + authorization context (SRCH-0023)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TenantContext:
    """Resolved identity + authorization for one authenticated request.

    Never constructed from a raw, unauthenticated request field — always produced by
    verifying a bearer credential (JWKS / arc_api_* introspection) and resolving the
    principal's allowed-namespace set (Auth Arcana ReBAC, or the local FK-cache fallback).
    An empty allowed_namespace_ids is a valid, deny-everything context.
    """

    principal_id: str
    principal_type: Literal["service", "user"]
    allowed_namespace_ids: frozenset[int]
    allowed_namespace_names: frozenset[str]
