"""Shared fixtures for the SRCH-0023 negative cross-tenant suite (tests/security/).

No live network call ever happens here: JWKS / OpenFGA / Auth Arcana introspection are
mocked at the `scrutator.auth.dependency` boundary — this suite proves the app-layer
scoping contract, not the transport to Auth Arcana (Step 0 interface, out of scope here).
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

# Fixture "namespaces table" backing the auth.dependency name<->id lookups.
NAMESPACE_TABLE = {1: "arcanada", 2: "secret-tenant"}


def make_namespace_pool_mock():
    """Mock asyncpg pool backing the namespaces table lookups inside auth.dependency."""

    async def fetch(sql, *params):
        if "id = ANY" in sql:
            ids = params[0]
            return [{"name": NAMESPACE_TABLE[i]} for i in ids if i in NAMESPACE_TABLE]
        return []

    async def fetchrow(sql, *params):
        if "name = $1" in sql:
            name = params[0]
            for nid, nname in NAMESPACE_TABLE.items():
                if nname == name:
                    return {"id": nid}
            return None
        return None

    mock_conn = AsyncMock()
    mock_conn.fetch.side_effect = fetch
    mock_conn.fetchrow.side_effect = fetchrow

    mock_pool = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire.return_value = ctx
    return mock_pool


@contextmanager
def mock_authenticated_principal(principal_id: str, allowed_namespace_ids: frozenset[int]):
    """Simulate a verified principal scoped to allowed_namespace_ids — no real JWKS/OpenFGA
    network call, no assumption about live Auth Arcana reachability (HARD-GATE compliant)."""
    mock_pool = make_namespace_pool_mock()
    with (
        patch(
            "scrutator.auth.dependency.verify_bearer_token",
            new_callable=AsyncMock,
            return_value=(principal_id, "service"),
        ),
        patch(
            "scrutator.auth.dependency.resolve_allowed_namespaces",
            new_callable=AsyncMock,
            return_value=allowed_namespace_ids,
        ),
        patch("scrutator.auth.dependency.get_pool", new_callable=AsyncMock, return_value=mock_pool),
    ):
        yield
