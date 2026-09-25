"""A2-308 — a bearer scope must authorize the VERB, not just the door.

Before A2-308 `require_tenant_context` checked one global scope on entry
(`verifier.py` `verify_ltm_m2m_token`) and then only namespace grants, so a token whose
scope literally reads `kb:ltm.read` could create and delete graph edges, write memories,
create namespaces and trigger a (billed, writing) reflect run.

These tests drive the REAL app through the REAL verifier with a REAL signed EdDSA token —
the only thing mocked is the JWKS key lookup, the namespace-grant resolution and the
storage call whose invocation is the thing under assertion. No live service is contacted
and no production token is used.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519
from fastapi.testclient import TestClient

from scrutator.config import settings
from scrutator.health import app
from tests.security.conftest import make_namespace_pool_mock

READ_SCOPE = "kb:ltm.read"
WRITE_SCOPE = "kb:ltm.write"
CLIENT_ID = "muneral-kb-sync"
NAMESPACE = "arcanada"
NAMESPACE_IDS = frozenset({1})

CHUNK_A = "12345678-1234-5678-1234-567812345678"
CHUNK_B = "87654321-4321-8765-4321-876543218765"


def _signed_token(scope: str) -> tuple[str, object]:
    """Mint a token that satisfies every OTHER clause of the LTM M2M profile."""
    now = int(time.time())
    claims = {
        "iss": settings.auth_ltm_issuer,
        "aud": settings.auth_ltm_audience,
        "sub": CLIENT_ID,
        "client_id": CLIENT_ID,
        "scope": scope,
        "iat": now,
        "nbf": now,
        "exp": now + settings.auth_ltm_max_token_lifetime_seconds,
    }
    private_key = ed25519.Ed25519PrivateKey.generate()
    return jwt.encode(claims, private_key, algorithm="EdDSA"), private_key.public_key()


@contextmanager
def token_client(scope: str):
    """A TestClient whose requests carry a genuinely signed token with exactly `scope`."""
    token, public_key = _signed_token(scope)
    signing_key = MagicMock()
    signing_key.key = public_key
    jwks_client = MagicMock()
    jwks_client.get_signing_key_from_jwt.return_value = signing_key
    with (
        patch("scrutator.auth.verifier._get_jwks_client", return_value=jwks_client),
        patch(
            "scrutator.auth.dependency.resolve_allowed_namespaces",
            new_callable=AsyncMock,
            return_value=NAMESPACE_IDS,
        ),
        patch("scrutator.auth.dependency.get_pool", new_callable=AsyncMock, return_value=make_namespace_pool_mock()),
        TestClient(app) as client,
    ):
        client.headers["Authorization"] = f"Bearer {token}"
        yield client


# ── the mutating surface ────────────────────────────────────────────
#
# (id, storage symbol whose invocation would be the mutation, call) — the storage patch is
# what makes "denied" mean "denied BEFORE the write", not "wrote and then errored".

MUTATIONS = {
    "edges.create": (
        "scrutator.health.insert_edges",
        lambda c: c.post(
            "/v1/edges",
            json=[{"source_chunk_id": CHUNK_A, "target_chunk_id": CHUNK_B, "edge_type": "related"}],
        ),
        1,
    ),
    "edges.create_by_path": (
        "scrutator.health.create_edges_by_path",
        lambda c: c.post(
            "/v1/edges/by-path",
            params={"namespace": NAMESPACE},
            json=[{"source_path": "a.md", "target_path": "b.md", "edge_type": "related"}],
        ),
        {"created": 0, "not_found": []},
    ),
    "edges.delete": (
        "scrutator.health.delete_edges_by_creator",
        lambda c: c.request("DELETE", "/v1/edges", params={"created_by": "dreamer", "namespace": NAMESPACE}),
        0,
    ),
    "namespaces.create": (
        "scrutator.health.upsert_namespace",
        lambda c: c.post("/v1/namespaces", json={"name": NAMESPACE}),
        1,
    ),
    "memories.create": (
        "scrutator.health.index_memory",
        lambda c: c.post("/v1/memories", json={"content": "x", "actor": "agent", "namespace": NAMESPACE}),
        {"memory_id": "m-1", "chunk_id": CHUNK_A, "namespace": NAMESPACE, "tokens": 1},
    ),
    "memories.bulk": (
        "scrutator.health.memory_bulk_index",
        lambda c: c.post(
            "/v1/memories/bulk",
            json={"memories": [{"content": "x", "actor": "agent", "namespace": NAMESPACE}]},
        ),
        {"indexed": 1, "failed": 0, "memory_ids": ["m-1"], "errors": []},
    ),
    "memories.delete": (
        "scrutator.db.repository.delete_memories_by_actor",
        lambda c: c.request("DELETE", "/v1/memories", params={"actor": "agent", "namespace": NAMESPACE}),
        0,
    ),
}


@pytest.mark.parametrize("route_id", sorted(MUTATIONS))
def test_read_scoped_token_cannot_mutate(route_id):
    """The headline defect: `.read` in the scope name must mean `.read` in the app."""
    symbol, call, _ = MUTATIONS[route_id]
    with token_client(READ_SCOPE) as client, patch(symbol, new_callable=AsyncMock) as storage:
        response = call(client)
    assert response.status_code == 403, (route_id, response.status_code, response.text)
    storage.assert_not_awaited()


@pytest.mark.parametrize("route_id", sorted(MUTATIONS))
def test_write_scoped_token_reaches_the_mutation(route_id):
    """The same token plus `kb:ltm.write` is admitted — the fix denies a verb, not a caller."""
    symbol, call, returns = MUTATIONS[route_id]
    with (
        token_client(f"{READ_SCOPE} {WRITE_SCOPE}") as client,
        patch(symbol, new_callable=AsyncMock, return_value=returns) as storage,
    ):
        response = call(client)
    assert response.status_code != 403, (route_id, response.status_code, response.text)
    storage.assert_awaited()


def test_read_scoped_token_cannot_trigger_a_reflect_run():
    """`/v1/ltm/reflect` writes meta-facts AND spends an LLM budget — also a mutation."""
    with (
        token_client(READ_SCOPE) as client,
        patch("scrutator.ltm.router.settings.ltm_reflect_enabled", True),
        patch("scrutator.ltm.router.ReflectJob") as job,
    ):
        response = client.post("/v1/ltm/reflect", json={"namespace": NAMESPACE})
    assert response.status_code == 403, response.text
    job.assert_not_called()


# ── the read surface is untouched ───────────────────────────────────

READS = {
    "edges.get": ("scrutator.health.get_edges_for_chunk", [], lambda c: c.get(f"/v1/edges/{CHUNK_A}")),
    "namespaces.list": ("scrutator.health.get_namespaces", [], lambda c: c.get("/v1/namespaces")),
    "memories.stats": (
        "scrutator.health.get_memory_stats",
        {"total_memories": 0, "by_actor": {}, "by_type": {}, "oldest": None, "newest": None},
        lambda c: c.get("/v1/memories/stats"),
    ),
}


@pytest.mark.parametrize("route_id", sorted(READS))
def test_read_scoped_token_still_reads(route_id):
    """Regression guard: the fix must not cost the readers their reads."""
    symbol, value, call = READS[route_id]
    with token_client(READ_SCOPE) as client, patch(symbol, new_callable=AsyncMock, return_value=value) as storage:
        response = call(client)
    assert response.status_code == 200, (route_id, response.text)
    storage.assert_awaited()


def test_read_scoped_token_still_searches():
    with (
        token_client(READ_SCOPE) as client,
        patch("scrutator.health.search", new_callable=AsyncMock) as search,
    ):
        search.return_value = MagicMock(results=[], total=0, query="q", search_time_ms=1.0)
        response = client.post("/v1/search", json={"query": "q", "namespace": NAMESPACE})
    assert response.status_code == 200, response.text
    search.assert_awaited()


# ── the credential must actually carry the scope ────────────────────


def test_grace_window_anonymous_context_cannot_mutate():
    """`SCRUTATOR_AUTH_ENFORCE=False` keeps unverified callers reading, never writing.

    Pre-A2-308 an unverified caller fell through to an empty-namespace context and still
    reached `insert_edges` (the namespace filter inside the repository was the only thing
    between it and the table). Now it is refused at the route.
    """
    with (
        patch("scrutator.auth.dependency.settings.auth_enforce", False),
        patch("scrutator.health.insert_edges", new_callable=AsyncMock) as insert,
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/edges",
            json=[{"source_chunk_id": CHUNK_A, "target_chunk_id": CHUNK_B, "edge_type": "related"}],
        )
    assert response.status_code == 403, response.text
    insert.assert_not_awaited()


def test_a_token_carrying_only_the_write_scope_is_not_a_valid_credential():
    """`kb:ltm.write` does not stand alone: the LTM profile still requires its read scope."""
    with token_client(WRITE_SCOPE) as client, patch("scrutator.health.insert_edges", new_callable=AsyncMock) as insert:
        response = client.post(
            "/v1/edges",
            json=[{"source_chunk_id": CHUNK_A, "target_chunk_id": CHUNK_B, "edge_type": "related"}],
        )
    # Unverifiable credential → grace-window empty context → still no write scope → 403.
    assert response.status_code == 403, response.text
    insert.assert_not_awaited()


def test_an_invented_scope_does_not_ride_along_with_a_valid_one():
    """Fail-closed parsing: an unrecognized scope denies rather than being ignored.

    Without this, a caller could present `kb:ltm.read kb:ltm.admin` and be treated exactly
    like a plain reader — the claim would widen silently in whatever Auth Arcana issues next.
    The credential fails verification entirely, so (grace window) it degrades to the
    zero-grant context: no write, and no namespace to read from either.
    """
    with (
        token_client(f"{READ_SCOPE} kb:ltm.admin") as client,
        patch("scrutator.health.get_edges_for_chunk", new_callable=AsyncMock, return_value=[]) as get_edges,
        patch("scrutator.health.insert_edges", new_callable=AsyncMock) as insert,
    ):
        read = client.get(f"/v1/edges/{CHUNK_A}")
        write = client.post(
            "/v1/edges",
            json=[{"source_chunk_id": CHUNK_A, "target_chunk_id": CHUNK_B, "edge_type": "related"}],
        )

    assert write.status_code == 403, write.text
    insert.assert_not_awaited()
    # The read is not granted the principal's namespaces — it is handed the empty set, the
    # same deny-everything context an unauthenticated caller gets.
    assert read.status_code == 200
    get_edges.assert_awaited_once_with(CHUNK_A, frozenset())
