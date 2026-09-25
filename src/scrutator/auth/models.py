"""Tenant identity + authorization context (SRCH-0023, A2-308)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class VerifiedPrincipal:
    """What a successfully verified bearer credential proved.

    `scopes` is the set the credential actually carries, never a default: A2-308 found
    `verify_*` discarding the scope claim after one equality check, so a `kb:ltm.read`
    token authorized mutations. Returning it as a value makes the grant travel with the
    principal instead of being asserted once at the door and forgotten.
    """

    principal_id: str
    principal_type: Literal["service", "user"]
    scopes: frozenset[str]


@dataclass(frozen=True)
class TenantContext:
    """Resolved identity + authorization for one authenticated request.

    Never constructed from a raw, unauthenticated request field — always produced by
    verifying a bearer credential (JWKS / arc_api_* introspection) and resolving the
    principal's allowed-namespace set (Auth Arcana ReBAC, or the local FK-cache fallback).
    An empty allowed_namespace_ids is a valid, deny-everything context.

    `scopes` has NO default on purpose (A2-308): every construction site must say what the
    credential authorizes. A forgotten field would otherwise inherit whichever default the
    last author picked, and the failure mode of picking wrong is a silent privilege grant.
    """

    principal_id: str
    principal_type: Literal["service", "user"]
    allowed_namespace_ids: frozenset[int]
    allowed_namespace_names: frozenset[str]
    scopes: frozenset[str]
