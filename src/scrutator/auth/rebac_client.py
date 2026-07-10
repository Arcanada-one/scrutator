"""Principal -> allowed-namespace resolution (SRCH-0023 Fork A / Step 0).

Primary path: Auth Arcana's OpenFGA HTTP API (`list-objects` over the
`user:<principal_id>#can_read@namespace:<name>` tuple shape, per the existing
`authz/model.fga` convention). Not confirmed reachable at `/dr-do` time (plan Step 0) —
falls back to the local FK-only cache table `principal_namespace_grants`, seeded
out-of-band by an operator-run grant script. This degrades ReBAC to an RBAC facade
(mandate §5) until the live OpenFGA path is confirmed — tracked as a follow-up backlog item.

Fail-closed: if both the live check and the FK-cache miss, the principal resolves to an
empty allowed-set — never "all namespaces".
"""

from __future__ import annotations

import logging

import httpx

from scrutator.config import settings
from scrutator.db.connection import get_pool

logger = logging.getLogger(__name__)


async def _namespace_names_to_ids(names: set[str]) -> frozenset[int]:
    """Resolve namespace name -> id via the namespaces table (names come from OpenFGA objects)."""
    if not names:
        return frozenset()
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT id FROM namespaces WHERE name = ANY($1::text[])", list(names))
    return frozenset(row["id"] for row in rows)


async def _resolve_via_openfga(principal_id: str) -> frozenset[int] | None:
    """Try the live Auth Arcana OpenFGA check. Returns None if unconfigured/unreachable
    (triggers the FK-cache fallback), otherwise the resolved (possibly empty) id set."""
    if not settings.auth_arcana_openfga_url or not settings.auth_arcana_openfga_store_id:
        return None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{settings.auth_arcana_openfga_url}/stores/{settings.auth_arcana_openfga_store_id}/list-objects",
                params={"user": f"user:{principal_id}", "relation": "can_read", "type": "namespace"},
            )
    except httpx.HTTPError:
        logger.warning("OpenFGA check unreachable for principal=%s; falling back to FK cache", principal_id)
        return None
    if resp.status_code != 200:
        logger.warning(
            "OpenFGA check returned status=%s for principal=%s; falling back to FK cache",
            resp.status_code,
            principal_id,
        )
        return None
    data = resp.json()
    names = {obj.split(":", 1)[1] for obj in data.get("objects", []) if ":" in obj}
    return await _namespace_names_to_ids(names)


async def _resolve_via_fk_cache(principal_id: str) -> frozenset[int]:
    """Local FK-only cache fallback: `principal_namespace_grants` table."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT namespace_id FROM principal_namespace_grants WHERE principal_id = $1",
            principal_id,
        )
    return frozenset(row["namespace_id"] for row in rows)


async def resolve_allowed_namespaces(principal_id: str) -> frozenset[int]:
    """Resolve principal -> allowed namespace id set.

    Fail-closed: an empty result (no live ReBAC grant, no FK-cache row) denies — the
    caller MUST treat an empty set as "no namespaces authorized", never as "all namespaces".
    """
    live = await _resolve_via_openfga(principal_id)
    if live is not None:
        return live
    return await _resolve_via_fk_cache(principal_id)
