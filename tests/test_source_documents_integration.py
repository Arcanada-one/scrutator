"""SRCH-0038 1b — live-PostgreSQL integration coverage for the isolated ``source_documents`` store.

This is the durable confirmation the mock suite CANNOT give. The 1a mechanism stamped the exact
pre-chunk bytes into ``chunks.metadata``, which carries ``idx_chunks_metadata … USING gin(metadata)``
(default ``jsonb_ops``). ``jsonb_ops`` indexes every scalar value as its own entry with a hard
~2704-byte (⅓-page) ceiling, so a real multi-KB skill body raised
``ERROR: index row size … exceeds maximum 2704`` at INSERT and failed the whole
``replace_source_chunks_atomic`` transaction. Real skills are 3–50 KB → essentially every real
skill was un-indexable. The 947-test mock suite (no live Postgres) is structurally blind to it.

These tests ingest a >2.7 KB skill AND a ~50 KB skill through the REAL ingest path and assert
(a) the ingest transaction SUCCEEDS and the ``source_documents`` row lands (the assertion that
would have caught the GIN failure), and (b) ``POST /v1/fetch`` returns byte-exact ``content`` with
``sha256(content) == content_hash``.

Skipped unless a disposable Scrutator-schema database is provided — mirroring the repo's other
live-PG integration suites:

    export SCRUTATOR_SRCH0038_TEST_DATABASE_URL=postgresql://user:pw@host:5432/scrutator_srch0038_test
    export SRCH0038_TEST_DB_GO=scrutator_srch0038_test      # must equal the DB name (belt-and-braces)
    .venv-linux/bin/python -m pytest tests/test_source_documents_integration.py -q
"""

from __future__ import annotations

import hashlib
import os
import uuid

import asyncpg
import pytest

from scrutator.config import settings
from scrutator.db.connection import apply_schema, close_pool, get_pool
from scrutator.db.models import FetchRequest
from scrutator.search.indexer import compute_doc_content_hash

_DENSE_DIMENSIONS = 1024


def _bare_hash(content_hash: str) -> str:
    return content_hash.split(":", 1)[1] if ":" in content_hash else content_hash


def _skill_doc(target_bytes: int) -> str:
    """A realistic markdown skill: YAML frontmatter + headings (so every chunk carries a
    ``section``, and thus a ``doc_id`` stamp, matching how fetch-by-doc_id resolves rows) + enough
    body to cross ``target_bytes``. Under 1a a body this size would exceed the 2704-byte jsonb_ops
    GIN entry ceiling when stamped into ``chunks.metadata``."""
    body = "Step: assemble, sign, and verify the release artifact deterministically. " * 30 + "\n\n"
    doc = "---\nname: big-release\nversion: 7\ntrust: skill\n---\n\n# Big Release Skill\n\n"
    section = 0
    while len(doc.encode("utf-8")) < target_bytes:
        section += 1
        doc += f"## Step {section}\n\n{body}"
    return doc


@pytest.fixture
async def skills_db(monkeypatch):
    dsn = os.environ.get("SCRUTATOR_SRCH0038_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("no disposable PostgreSQL in SCRUTATOR_SRCH0038_TEST_DATABASE_URL")

    # Safety: never touch the configured application database; require an explicitly approved
    # disposable *_test database (same contract as the LTM-0014 integration suite).
    probe = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2)
    database_name = await probe.fetchval("SELECT current_database()")
    approval = os.environ.get("SRCH0038_TEST_DB_GO")
    if not database_name.endswith("_test") or approval != database_name or dsn == settings.database_url:
        await probe.close()
        pytest.fail("SRCH-0038 integration DB must be a separately approved *_test database")
    await probe.close()

    # Repoint the application pool at the disposable DB and ensure the current schema (incl. the
    # new source_documents table) is applied — apply_schema() is idempotent (IF NOT EXISTS).
    monkeypatch.setattr(settings, "database_url", dsn)
    await close_pool()
    await apply_schema()

    # A per-run unique namespace treated as the skills namespace, so the exact-bytes path fires
    # and cleanup is isolated from any real "skills" data.
    test_ns = f"skills-srch0038-{uuid.uuid4()}"
    monkeypatch.setattr(settings, "skills_namespace", test_ns)

    # Deterministic, network-free embeddings so the real ingest path runs without the embed API.
    async def _fake_dense(texts):
        return [[0.001 * (i + 1)] * _DENSE_DIMENSIONS for i in range(len(texts))]

    async def _fake_sparse(texts):
        return [{"tok": 1.0} for _ in texts]

    monkeypatch.setattr("scrutator.search.indexer.embed_texts", _fake_dense)
    monkeypatch.setattr("scrutator.search.indexer.embed_sparse", _fake_sparse)

    pool = await get_pool()
    namespace_id = await pool.fetchval("INSERT INTO namespaces (name) VALUES ($1) RETURNING id", test_ns)
    try:
        yield pool, test_ns, namespace_id
    finally:
        await pool.execute("DELETE FROM source_documents WHERE namespace_id = $1", namespace_id)
        await pool.execute("DELETE FROM chunks WHERE namespace_id = $1", namespace_id)
        await pool.execute("DELETE FROM namespaces WHERE id = $1", namespace_id)
        await close_pool()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target_bytes",
    [
        pytest.param(4 * 1024, id="over-2.7KB-GIN-ceiling"),
        pytest.param(50 * 1024, id="~50KB-real-skill"),
    ],
)
async def test_multi_kb_skill_ingest_succeeds_and_fetch_is_byte_exact(skills_db, target_bytes):
    from scrutator.chunker.splitters import compute_doc_id
    from scrutator.search import fetcher, indexer

    pool, test_ns, namespace_id = skills_db
    source_path = f"skills/big-{target_bytes}.md"
    doc = _skill_doc(target_bytes)
    assert len(doc.encode("utf-8")) > 2704, "doc must exceed the 2704-byte jsonb_ops GIN entry ceiling"

    # (a) The REAL ingest transaction must SUCCEED. Under 1a this INSERT raised
    # `index row size ... exceeds maximum 2704` and aborted the whole transaction — the failure
    # this test exists to catch.
    resp = await indexer.index_document(doc, source_path, namespace=test_ns)
    assert resp.chunks_indexed > 1, "a multi-KB skill must span >1 chunk"

    # The exact-source row landed, byte-identical, with a hash the fetch path advertises.
    expected_hash = compute_doc_content_hash(doc)
    row = await pool.fetchrow(
        "SELECT raw_content, content_hash FROM source_documents WHERE namespace_id = $1 AND source_path = $2",
        namespace_id,
        source_path,
    )
    assert row is not None, "source_documents row must exist after ingest"
    assert row["raw_content"] == doc
    assert row["content_hash"] == expected_hash
    # By-construction: the stored bytes hash to the stored hash.
    assert hashlib.sha256(row["raw_content"].encode()).hexdigest() == _bare_hash(row["content_hash"])

    # (b) POST /v1/fetch returns byte-exact content with sha256(content) == content_hash.
    doc_id = compute_doc_id(test_ns, source_path)
    fetched = await fetcher.fetch(
        FetchRequest(by="source_id", id=doc_id, range="full"),
        frozenset({namespace_id}),
    )
    assert fetched.trust_class == "skill"
    assert fetched.content_exact is True
    assert fetched.content == doc
    assert fetched.content_hash == expected_hash
    assert hashlib.sha256(fetched.content.encode()).hexdigest() == _bare_hash(fetched.content_hash)
