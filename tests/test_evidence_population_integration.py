"""Actual disposable PostgreSQL raw-only effects and receiving controls."""

import asyncio

import pytest

from scrutator.chunker.splitters import compute_doc_id
from scrutator.config import settings
from scrutator.db.models import FetchRequest
from scrutator.db.repository import fetch_chunks_by_doc_id, fetch_evidence_raw_content
from scrutator.search import fetcher, indexer

pytest_plugins = ("tests.test_evidence_documents_integration",)


async def seed(evidence_db, monkeypatch):
    pool, namespace, namespace_id, _ = evidence_db
    body = "---\r\ntitle: Original\r\n---\r\n# Source\n\n  λ " + "detail " * 500 + "\n  \n"
    path = "exact/source.md"
    monkeypatch.setattr(settings, "evidence_exact_bytes", False)
    await indexer.index_document(body, path, namespace=namespace)
    monkeypatch.setattr(settings, "evidence_exact_bytes", True)
    monkeypatch.setattr(settings, "evidence_exact_namespaces", [namespace])
    return pool, namespace, namespace_id, body, path


@pytest.mark.asyncio
async def test_additive_population_keeps_chunk_generation_and_graph(evidence_db, monkeypatch):
    pool, namespace, nid, body, path = await seed(evidence_db, monkeypatch)
    before = await pool.fetch(
        "SELECT id, metadata, indexed_at FROM chunks WHERE namespace_id=$1 ORDER BY chunk_index", nid
    )
    assert len(before) > 1
    await pool.execute(
        "INSERT INTO graph_edges(source_chunk_id,target_chunk_id,edge_type) VALUES($1,$2,'fixture')",
        before[0]["id"],
        before[1]["id"],
    )
    result = await indexer.populate_exact_evidence(body, path, namespace)
    assert result.chunks_indexed == 0 and result.strategy_used == "evidence_exact_created"
    assert (
        await pool.fetch("SELECT id, metadata, indexed_at FROM chunks WHERE namespace_id=$1 ORDER BY chunk_index", nid)
        == before
    )
    assert await pool.fetchval("SELECT count(*) FROM graph_edges WHERE source_chunk_id=$1", before[0]["id"]) == 1
    response = await fetcher.fetch(
        FetchRequest(by="source_id", id=compute_doc_id(namespace, path), range="full"), frozenset({nid})
    )
    assert response.content_exact and response.content == body
    assert (await indexer.populate_exact_evidence(body, path, namespace)).strategy_used == "evidence_exact_present"


@pytest.mark.asyncio
async def test_wrong_generation_and_conflicting_raw_are_refused(evidence_db, monkeypatch):
    pool, namespace, nid, body, path = await seed(evidence_db, monkeypatch)
    with pytest.raises(ValueError, match="generation mismatch"):
        await indexer.populate_exact_evidence(body + "changed", path, namespace)
    assert await pool.fetchval("SELECT count(*) FROM evidence_documents WHERE namespace_id=$1", nid) == 0
    await indexer.populate_exact_evidence(body, path, namespace)
    await pool.execute("UPDATE evidence_documents SET raw_content='corrupt' WHERE namespace_id=$1", nid)
    with pytest.raises(ValueError, match="conflicting"):
        await indexer.populate_exact_evidence(body, path, namespace)
    response = await fetcher.fetch(
        FetchRequest(by="source_id", id=compute_doc_id(namespace, path), range="full"), frozenset({nid})
    )
    assert not response.content_exact and response.content != body


@pytest.mark.asyncio
async def test_actual_insert_failure_rolls_back_without_chunk_mutation(evidence_db, monkeypatch):
    pool, namespace, nid, body, path = await seed(evidence_db, monkeypatch)
    before = await pool.fetch(
        "SELECT id,metadata,indexed_at FROM chunks WHERE namespace_id=$1 ORDER BY chunk_index", nid
    )
    await pool.execute(
        "CREATE OR REPLACE FUNCTION fixture_exact_refuse() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RAISE EXCEPTION 'fixture rollback'; END $$"
    )
    await pool.execute(
        "CREATE TRIGGER fixture_exact_refuse AFTER INSERT ON evidence_documents "
        "FOR EACH ROW EXECUTE FUNCTION fixture_exact_refuse()"
    )
    try:
        with pytest.raises(Exception, match="fixture rollback"):
            await indexer.populate_exact_evidence(body, path, namespace)
        assert await pool.fetchval("SELECT count(*) FROM evidence_documents WHERE namespace_id=$1", nid) == 0
        assert (
            await pool.fetch(
                "SELECT id,metadata,indexed_at FROM chunks WHERE namespace_id=$1 ORDER BY chunk_index", nid
            )
            == before
        )
    finally:
        await pool.execute(
            "DROP TRIGGER fixture_exact_refuse ON evidence_documents; DROP FUNCTION fixture_exact_refuse()"
        )


@pytest.mark.asyncio
async def test_equal_body_new_generation_does_not_bind_old_chunk_snapshot(evidence_db, monkeypatch):
    pool, namespace, nid, body, path = await seed(evidence_db, monkeypatch)
    docid = compute_doc_id(namespace, path)
    old = await fetch_chunks_by_doc_id(docid, frozenset({nid}))
    await indexer.index_document(body, path, namespace=namespace)
    assert await fetch_evidence_raw_content(docid, frozenset({nid}), expected_rows=old) is None
    current = await fetch_chunks_by_doc_id(docid, frozenset({nid}))
    assert await fetch_evidence_raw_content(docid, frozenset({nid}), expected_rows=current) == (
        body,
        indexer.compute_doc_content_hash(body),
    )


@pytest.mark.asyncio
async def test_population_waits_for_source_lock_then_refuses_changed_generation(evidence_db, monkeypatch):
    pool, namespace, nid, body, path = await seed(evidence_db, monkeypatch)
    async with pool.acquire() as owner:
        async with owner.transaction():
            await owner.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1::int::text || ':' || $2, 0))", nid, path
            )
            pending = asyncio.create_task(indexer.populate_exact_evidence(body, path, namespace))
            try:
                for _ in range(100):
                    waiting = await pool.fetchval(
                        "SELECT count(*) FROM pg_locks l JOIN pg_stat_activity a ON a.pid=l.pid "
                        "WHERE l.locktype='advisory' AND NOT l.granted AND a.datname=current_database()"
                    )
                    if waiting:
                        break
                    await asyncio.sleep(0.01)
                assert waiting > 0, "actual population session must wait on the source lock"
                assert not pending.done()
                await owner.execute(
                    "UPDATE chunks SET metadata=jsonb_set(metadata,'{section,doc_content_hash}',to_jsonb($3::text)) "
                    "WHERE namespace_id=$1 AND source_path=$2",
                    nid,
                    path,
                    indexer.compute_doc_content_hash(body + "changed"),
                )
            except BaseException:
                pending.cancel()
                raise
        with pytest.raises(ValueError, match="generation mismatch"):
            await asyncio.wait_for(pending, 2)
    assert await pool.fetchval("SELECT count(*) FROM evidence_documents WHERE namespace_id=$1", nid) == 0


@pytest.mark.asyncio
async def test_false_raw_body_stamp_refuses_before_population(evidence_db, monkeypatch):
    from scrutator.db.repository import populate_exact_evidence_atomic

    pool, namespace, nid, body, path = await seed(evidence_db, monkeypatch)
    document = indexer._build_evidence_document(namespace, path, body)
    document["raw_content"] += "tampered"
    with pytest.raises(ValueError, match="body digest mismatch"):
        await populate_exact_evidence_atomic(namespace, document)
    assert await pool.fetchval("SELECT count(*) FROM evidence_documents WHERE namespace_id=$1", nid) == 0
