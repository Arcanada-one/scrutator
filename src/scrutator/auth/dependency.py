"""FastAPI tenant-auth dependency + namespace-selector resolution (SRCH-0023 Step 1/2).

`require_tenant_context` is the single choke-point every non-health route depends on.
Fail-closed: a verification failure denies (401) once `SCRUTATOR_AUTH_ENFORCE=True`. During
the dual-auth grace window (default, `auth_enforce=False`) an unverified caller is NOT
rejected — but it is granted an EMPTY allowed-namespace set (never all-namespaces) and the
attempt is logged as a would-deny audit event, so the caller still can't read cross-tenant
data; it can only hit the ambiguous/forbidden paths that an authenticated zero-grant
principal would hit.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request

from scrutator.auth.models import TenantContext
from scrutator.auth.rebac_client import resolve_allowed_namespaces
from scrutator.auth.verifier import Unauthenticated, verify_bearer_token
from scrutator.config import settings
from scrutator.db.connection import get_pool

logger = logging.getLogger(__name__)

_EMPTY_CONTEXT_PRINCIPAL = "anonymous"


async def _namespace_ids_to_names(namespace_ids: frozenset[int]) -> frozenset[str]:
    if not namespace_ids:
        return frozenset()
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT name FROM namespaces WHERE id = ANY($1::int[])", list(namespace_ids))
    return frozenset(row["name"] for row in rows)


async def require_tenant_context(request: Request) -> TenantContext:
    """FastAPI dependency: verify the bearer credential, resolve the allowed-namespace set."""
    authorization = request.headers.get("authorization")
    try:
        principal_id, principal_type = await verify_bearer_token(authorization)
    except Unauthenticated as exc:
        if settings.auth_enforce:
            raise HTTPException(status_code=401, detail="unauthenticated") from exc
        logger.warning(
            "would-deny (SCRUTATOR_AUTH_ENFORCE=False): %s %s — %s",
            request.method,
            request.url.path,
            exc,
        )
        return TenantContext(
            principal_id=_EMPTY_CONTEXT_PRINCIPAL,
            principal_type="service",
            allowed_namespace_ids=frozenset(),
            allowed_namespace_names=frozenset(),
        )

    allowed_ids = await resolve_allowed_namespaces(principal_id)
    allowed_names = await _namespace_ids_to_names(allowed_ids)
    return TenantContext(
        principal_id=principal_id,
        principal_type=principal_type,
        allowed_namespace_ids=allowed_ids,
        allowed_namespace_names=allowed_names,
    )


async def resolve_namespace_selector(ctx: TenantContext, requested: str | None) -> int:
    """Resolve the effective namespace_id for a request against the caller's allowed-set.

    - omitted + single-tenant principal -> that tenant.
    - omitted + multi-tenant principal -> 400 (must disambiguate).
    - omitted + zero-tenant principal -> 403 (nothing authorized).
    - requested outside the allowed-set -> 403, never silently re-scoped.
    - requested inside the allowed-set -> resolved id.
    """
    if requested is None:
        if len(ctx.allowed_namespace_ids) == 1:
            return next(iter(ctx.allowed_namespace_ids))
        if not ctx.allowed_namespace_ids:
            raise HTTPException(status_code=403, detail="no namespace authorized for this principal")
        raise HTTPException(status_code=400, detail="namespace is ambiguous — multiple namespaces authorized")

    if requested not in ctx.allowed_namespace_names:
        raise HTTPException(status_code=403, detail=f"namespace '{requested}' not in caller's allowed set")

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id FROM namespaces WHERE name = $1", requested)
    if row is None or row["id"] not in ctx.allowed_namespace_ids:
        raise HTTPException(status_code=403, detail=f"namespace '{requested}' not in caller's allowed set")
    return row["id"]
