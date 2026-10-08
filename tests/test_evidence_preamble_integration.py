"""Real producer/database controls for narrowly proven legacy Markdown preambles."""

import hashlib

import pytest

from scrutator.chunker.splitters import compute_doc_id
from scrutator.config import settings
from scrutator.db.models import FetchRequest, OffsetRange, ParentOfChunkRange
from scrutator.db.repository import fetch_chunks_by_doc_id, fetch_evidence_raw_content
from scrutator.search import fetcher, indexer

pytest_plugins = ("tests.test_evidence_documents_integration",)


async def seed(evidence_db, monkeypatch):
    pool, namespace, nid, _ = evidence_db
    body = "---\r\ntitle: Fixture only\r\n---\r\n\nA short λ preamble.\n\n# Source\n\n" + "detail " * 500 + "\n"
    path = "fixture/preamble.md"
    monkeypatch.setattr(settings, "evidence_exact_bytes", False)
    await indexer.index_document(
        body,
        path,
        namespace=namespace,
        max_tokens=settings.evidence_producer_max_tokens,
        overlap_tokens=settings.evidence_producer_overlap_tokens,
    )
    monkeypatch.setattr(settings, "evidence_exact_bytes", True)
    monkeypatch.setattr(settings, "evidence_exact_namespaces", [namespace])
    rows = await pool.fetch("SELECT * FROM chunks WHERE namespace_id=$1 ORDER BY chunk_index", nid)
    return pool, namespace, nid, body, path, rows


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [(512, 50), (1024, 64)])
async def test_preamble_population_and_all_selectors_preserve_complete_generation(evidence_db, monkeypatch, profile):
    monkeypatch.setattr(settings, "evidence_producer_max_tokens", profile[0])
    monkeypatch.setattr(settings, "evidence_producer_overlap_tokens", profile[1])
    pool, namespace, nid, body, path, before = await seed(evidence_db, monkeypatch)
    await pool.execute(
        "INSERT INTO graph_edges(source_chunk_id,target_chunk_id,edge_type) VALUES($1,$2,'fixture-prefix')",
        before[0]["id"],
        before[1]["id"],
    )
    assert (await indexer.populate_exact_evidence(body, path, namespace)).strategy_used == "evidence_exact_created"
    assert (await indexer.populate_exact_evidence(body, path, namespace)).strategy_used == "evidence_exact_present"
    for by, value in [
        ("source_id", compute_doc_id(namespace, path)),
        ("chunk_id", str(before[0]["id"])),
        ("chunk_id", str(before[1]["id"])),
    ]:
        response = await fetcher.fetch(FetchRequest(by=by, id=value, range="full"), frozenset({nid}))
        assert response.content_exact and response.content == body
        assert response.content_hash == "sha256:" + hashlib.sha256(body.encode()).hexdigest()
        assert len(response.chunk_manifest) == len(before)
    doc_id = compute_doc_id(namespace, path)
    parent = await fetcher.fetch(
        FetchRequest(by="source_id", id=doc_id, range=ParentOfChunkRange(parent_of_chunk=str(before[0]["id"]))),
        frozenset({nid}),
    )
    assert parent.content_exact and parent.content == body
    sliced = await fetcher.fetch(
        FetchRequest(by="source_id", id=doc_id, range=OffsetRange(offset_start=2, offset_end=20)), frozenset({nid})
    )
    assert sliced.content_exact and sliced.content == body[2:20]
    assert sliced.content_hash == parent.content_hash
    with pytest.raises(Exception) as denied:
        await fetcher.fetch(FetchRequest(by="source_id", id=doc_id), frozenset())
    assert denied.value.status_code == 404
    assert before == await pool.fetch("SELECT * FROM chunks WHERE namespace_id=$1 ORDER BY chunk_index", nid)
    assert await pool.fetchval("SELECT count(*) FROM graph_edges WHERE source_chunk_id=$1", before[0]["id"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "content='forged'",
        "content_hash='forged'",
        "source_type='text'",
        "metadata=metadata-'section'",
        "metadata=jsonb_set(metadata,'{frontmatter}','{}'::jsonb)",
        "metadata=jsonb_set(metadata,'{heading_hierarchy}','[\"forged\"]'::jsonb)",
        "metadata=jsonb_set(metadata,'{section}','{}'::jsonb)",
        "parent_id=id",
        "chunk_index=99",
    ],
)
async def test_forged_preamble_refuses_population(evidence_db, monkeypatch, mutation):
    pool, namespace, nid, body, path, before = await seed(evidence_db, monkeypatch)
    await pool.execute(f"UPDATE chunks SET {mutation} WHERE id=$1", before[0]["id"])
    with pytest.raises(ValueError, match="generation mismatch"):
        await indexer.populate_exact_evidence(body, path, namespace)
    assert await pool.fetchval("SELECT count(*) FROM evidence_documents WHERE namespace_id=$1", nid) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "metadata=jsonb_set(metadata,'{section}','null'::jsonb)",
        "metadata=jsonb_set(metadata,'{section,doc_content_hash}','\"forged\"'::jsonb)",
        "metadata=jsonb_set(metadata,'{section,doc_id}','\"forged\"'::jsonb)",
        "content='changed'",
        "content_hash='changed'",
    ],
)
async def test_mixed_tail_generation_refuses_population(evidence_db, monkeypatch, mutation):
    pool, namespace, nid, body, path, before = await seed(evidence_db, monkeypatch)
    await pool.execute(f"UPDATE chunks SET {mutation} WHERE id=$1", before[1]["id"])
    with pytest.raises(ValueError, match="generation mismatch"):
        await indexer.populate_exact_evidence(body, path, namespace)
    assert await pool.fetchval("SELECT count(*) FROM evidence_documents WHERE namespace_id=$1", nid) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "content='changed'",
        "content_hash='changed'",
        "indexed_at=indexed_at+interval '1 second'",
        "metadata=metadata-'section'",
    ],
)
async def test_changed_preamble_invalidates_read_snapshot(evidence_db, monkeypatch, mutation):
    pool, namespace, nid, body, path, before = await seed(evidence_db, monkeypatch)
    await indexer.populate_exact_evidence(body, path, namespace)
    doc_id = compute_doc_id(namespace, path)
    old = await fetch_chunks_by_doc_id(doc_id, frozenset({nid}))
    await pool.execute(f"UPDATE chunks SET {mutation} WHERE id=$1", before[0]["id"])
    assert await fetch_evidence_raw_content(doc_id, frozenset({nid}), expected_rows=old) is None
    current = await fetcher.fetch(FetchRequest(by="source_id", id=doc_id, range="full"), frozenset({nid}))
    if "indexed_at=" not in mutation:
        assert not current.content_exact


@pytest.mark.asyncio
async def test_unknown_producer_profile_refuses(evidence_db, monkeypatch):
    pool, namespace, nid, body, path, _ = await seed(evidence_db, monkeypatch)
    monkeypatch.setattr(settings, "evidence_producer_max_tokens", 1)
    with pytest.raises(ValueError, match="generation mismatch"):
        await indexer.populate_exact_evidence(body, path, namespace)
    assert await pool.fetchval("SELECT count(*) FROM evidence_documents WHERE namespace_id=$1", nid) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["drop-tail", "drop-preamble", "all-unstamped", "missing-anchor"])
async def test_incomplete_generation_refuses(evidence_db, monkeypatch, operation):
    pool, namespace, nid, body, path, before = await seed(evidence_db, monkeypatch)
    if operation == "drop-tail":
        await pool.execute("DELETE FROM chunks WHERE id=$1", before[-1]["id"])
    elif operation == "drop-preamble":
        await pool.execute("DELETE FROM chunks WHERE id=$1", before[0]["id"])
    elif operation == "all-unstamped":
        await pool.execute(
            "UPDATE chunks SET metadata=jsonb_set(metadata,'{section}','null'::jsonb) WHERE namespace_id=$1", nid
        )
    else:
        await pool.execute("DELETE FROM chunks WHERE namespace_id=$1 AND chunk_index>0", nid)
    with pytest.raises(ValueError, match="generation mismatch"):
        await indexer.populate_exact_evidence(body, path, namespace)
    assert await pool.fetchval("SELECT count(*) FROM evidence_documents WHERE namespace_id=$1", nid) == 0
