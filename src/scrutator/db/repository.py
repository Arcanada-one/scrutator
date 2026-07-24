"""Database repository — CRUD and search queries using asyncpg."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np

from scrutator.db.connection import get_pool
from scrutator.db.models import (
    ChunkLookupResult,
    NamespaceInfo,
    NamespaceStats,
    SearchResult,
    doc_fields_from_metadata,
)

if TYPE_CHECKING:
    from scrutator.memory.models import MemoryStats


async def upsert_namespace(name: str, description: str | None = None) -> int:
    """Create namespace if not exists, return its id."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO namespaces (name, description)
            VALUES ($1, $2)
            ON CONFLICT (name) DO UPDATE SET description = COALESCE($2, namespaces.description)
            RETURNING id
            """,
            name,
            description,
        )
        return row["id"]


async def get_namespace_id(name: str) -> int | None:
    """Resolve an existing namespace without mutating it."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval("SELECT id FROM namespaces WHERE name = $1", name)


async def upsert_project(namespace_id: int, name: str, description: str | None = None) -> int:
    """Create project if not exists, return its id."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO projects (namespace_id, name, description)
            VALUES ($1, $2, $3)
            ON CONFLICT (namespace_id, name) DO UPDATE SET description = COALESCE($3, projects.description)
            RETURNING id
            """,
            namespace_id,
            name,
            description,
        )
        return row["id"]


async def insert_chunks(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    namespace_id: int,
    project_id: int | None = None,
) -> int:
    """Insert chunks with embeddings. Returns count of inserted rows."""
    if not chunks:
        return 0
    pool = await get_pool()
    inserted = 0
    async with pool.acquire() as conn:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            vector = np.array(embedding, dtype=np.float32)
            metadata_json = json.dumps(chunk.get("metadata", {}))
            await conn.execute(
                """
                INSERT INTO chunks (
                    namespace_id, project_id, source_path, source_type,
                    chunk_index, parent_id, content, content_hash,
                    embedding_dense, metadata, token_count, indexed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11, NOW())
                ON CONFLICT (namespace_id, source_path, chunk_index)
                DO UPDATE SET
                    content = EXCLUDED.content,
                    content_hash = EXCLUDED.content_hash,
                    embedding_dense = EXCLUDED.embedding_dense,
                    metadata = EXCLUDED.metadata,
                    token_count = EXCLUDED.token_count,
                    updated_at = NOW(),
                    indexed_at = NOW()
                """,
                namespace_id,
                project_id,
                chunk["source_path"],
                chunk["source_type"],
                chunk["chunk_index"],
                chunk.get("parent_id"),
                chunk["content"],
                chunk["content_hash"],
                vector,
                metadata_json,
                chunk.get("token_count", 0),
            )
            inserted += 1
    return inserted


async def insert_sparse_vectors(chunk_ids: list[str], sparse_weights: list[dict[str, float]]) -> int:
    """Insert sparse vectors for chunks. ON CONFLICT → update. Returns count."""
    if not chunk_ids:
        return 0
    pool = await get_pool()
    inserted = 0
    async with pool.acquire() as conn:
        for chunk_id, weights in zip(chunk_ids, sparse_weights, strict=True):
            await conn.execute(
                """
                INSERT INTO sparse_vectors (chunk_id, token_weights)
                VALUES ($1::uuid, $2::jsonb)
                ON CONFLICT (chunk_id)
                DO UPDATE SET token_weights = EXCLUDED.token_weights
                """,
                chunk_id,
                json.dumps(weights),
            )
            inserted += 1
    return inserted


_ATOMIC_CHUNK_UPSERT_SQL = """
    INSERT INTO chunks (
        id, namespace_id, project_id, source_path, source_type,
        chunk_index, parent_id, content, content_hash,
        embedding_dense, metadata, token_count, indexed_at
    ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7::uuid, $8, $9, $10, $11::jsonb, $12, NOW())
    ON CONFLICT (namespace_id, source_path, chunk_index)
    DO UPDATE SET
        project_id = EXCLUDED.project_id,
        source_type = EXCLUDED.source_type,
        parent_id = EXCLUDED.parent_id,
        content = EXCLUDED.content,
        content_hash = EXCLUDED.content_hash,
        embedding_dense = EXCLUDED.embedding_dense,
        metadata = EXCLUDED.metadata,
        token_count = EXCLUDED.token_count,
        updated_at = NOW(),
        indexed_at = NOW()
    RETURNING id
"""

_ATOMIC_SPARSE_UPSERT_SQL = """
    INSERT INTO sparse_vectors (chunk_id, token_weights)
    VALUES ($1::uuid, $2::jsonb)
    ON CONFLICT (chunk_id)
    DO UPDATE SET token_weights = EXCLUDED.token_weights
"""

_ATOMIC_DELETE_SOURCE_SQL = """
    DELETE FROM chunks
    WHERE namespace_id = $1 AND source_path = $2
"""

# SRCH-0038 1b: exact whole-document bytes for the skills namespace, upserted INSIDE the same
# `replace_source_chunks_atomic` transaction (crash-consistent with the chunk generation). The
# blob lives here, OUT of the GIN-indexed `chunks.metadata`, so the ~2704-byte jsonb_ops entry
# ceiling never applies. `raw_content` is byte-identical to the content `content_hash` hashes.
_ATOMIC_SOURCE_DOCUMENT_UPSERT_SQL = """
    INSERT INTO source_documents (namespace_id, source_path, doc_id, content_hash, raw_content, updated_at)
    VALUES ($1, $2, $3, $4, $5, NOW())
    ON CONFLICT (namespace_id, source_path)
    DO UPDATE SET
        doc_id = EXCLUDED.doc_id,
        content_hash = EXCLUDED.content_hash,
        raw_content = EXCLUDED.raw_content,
        updated_at = NOW()
"""

# SRCH-0039 (Mechanism C): exact whole-document bytes for the LARGE evidence corpus, upserted INSIDE
# the same `replace_source_chunks_atomic` transaction (crash-consistent with the chunk generation).
# A SEPARATE table from `source_documents` so evidence's flag-gated / larger / gracefully-degrading
# policy stays isolated from skills' always-exact / 256 KB-cap / fail-closed policy. `raw_content` is
# byte-identical to the content `content_hash` hashes, so `sha256(raw_content) == content_hash`.
_ATOMIC_EVIDENCE_DOCUMENT_UPSERT_SQL = """
    INSERT INTO evidence_documents (namespace_id, source_path, doc_id, content_hash, raw_content, updated_at)
    VALUES ($1, $2, $3, $4, $5, NOW())
    ON CONFLICT (namespace_id, source_path)
    DO UPDATE SET
        doc_id = EXCLUDED.doc_id,
        content_hash = EXCLUDED.content_hash,
        raw_content = EXCLUDED.raw_content,
        updated_at = NOW()
"""

# SRCH-0039 pre-merge review (write-side invalidation): when a chunk replacement re-stamps the
# chunks' `doc_content_hash` but the evidence upsert is SKIPPED (flag OFF, or a skills doc), any
# pre-existing evidence_documents row would go STALE (its raw_content no longer hashes to the new
# stamp). Delete it in the SAME transaction so a replacement can never leave a stale exact-bytes row.
_ATOMIC_EVIDENCE_DOCUMENT_DELETE_SQL = """
    DELETE FROM evidence_documents WHERE namespace_id = $1 AND source_path = $2
"""


def _validate_atomic_replacement(
    chunks: list[dict[str, Any]], embeddings: list[list[float]], sparse_weights: list[dict[str, float]]
) -> str:
    if len(chunks) != len(embeddings) or len(chunks) != len(sparse_weights):
        raise ValueError("chunk and embedding cardinalities must match")
    source_path = chunks[0]["source_path"]
    if any(chunk["source_path"] != source_path for chunk in chunks):
        raise ValueError("atomic source replacement accepts one source_path")
    emitted_ids: set[str] = set()
    for chunk in chunks:
        chunk_id = chunk.get("id")
        if not isinstance(chunk_id, str):
            raise ValueError("each atomic replacement chunk requires a UUID id")
        try:
            UUID(chunk_id)
        except ValueError as exc:
            raise ValueError("each atomic replacement chunk requires a UUID id") from exc
        if chunk_id in emitted_ids:
            raise ValueError("duplicate chunk id in atomic replacement")
        parent_id = chunk.get("parent_id")
        if parent_id is not None and parent_id not in emitted_ids:
            raise ValueError("parent_id must reference an earlier chunk in the same replacement")
        emitted_ids.add(chunk_id)
    return source_path


async def replace_source_chunks_atomic(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    sparse_weights: list[dict[str, float]],
    namespace_id: int,
    project_id: int | None = None,
    source_document: dict[str, Any] | None = None,
    evidence_document: dict[str, Any] | None = None,
) -> int:
    """Replace one source generation under an advisory lock and one transaction.

    ``source_document`` (SRCH-0038 1b, skills namespace only) carries the exact whole-document
    bytes ``{doc_id, source_path, content_hash, raw_content}``. When present it is upserted into
    ``source_documents`` INSIDE this same transaction, so the exact-source blob is crash-consistent
    with the chunk generation it belongs to. It is deliberately NOT written to ``chunks.metadata``
    (that GIN-indexed column hit the ~2704-byte jsonb_ops entry ceiling on real multi-KB skills).

    ``evidence_document`` (SRCH-0039 Mechanism C, non-skills namespaces, only when
    ``evidence_exact_bytes`` is ON) carries the same shape for the evidence corpus and is upserted
    into the SEPARATE ``evidence_documents`` table inside this same transaction — crash-consistent
    by construction. The two are mutually exclusive per document (a doc is skills XOR evidence)."""
    if not chunks:
        return 0
    source_path = _validate_atomic_replacement(chunks, embeddings, sparse_weights)

    pool = await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended($1::int::text || ':' || $2, 0))",
            namespace_id,
            source_path,
        )
        # Preserve legacy delete/reinsert FK-cascade semantics inside the
        # transactionally locked source replacement.
        await conn.execute(_ATOMIC_DELETE_SOURCE_SQL, namespace_id, source_path)
        if source_document is not None:
            await conn.execute(
                _ATOMIC_SOURCE_DOCUMENT_UPSERT_SQL,
                namespace_id,
                source_path,
                source_document["doc_id"],
                source_document["content_hash"],
                source_document["raw_content"],
            )
        if evidence_document is not None:
            await conn.execute(
                _ATOMIC_EVIDENCE_DOCUMENT_UPSERT_SQL,
                namespace_id,
                source_path,
                evidence_document["doc_id"],
                evidence_document["content_hash"],
                evidence_document["raw_content"],
            )
        else:
            # Write-side invalidation: no fresh exact bytes for this generation (flag OFF, or skills)
            # → drop any stale evidence row so fetch degrades gracefully to reassembly afterward.
            # No-op for skills source_paths (they have no evidence_documents row).
            await conn.execute(_ATOMIC_EVIDENCE_DOCUMENT_DELETE_SQL, namespace_id, source_path)
        for chunk, embedding, weights in zip(chunks, embeddings, sparse_weights, strict=True):
            chunk_index = chunk["chunk_index"]
            chunk_id = await conn.fetchval(
                _ATOMIC_CHUNK_UPSERT_SQL,
                chunk["id"],
                namespace_id,
                project_id,
                source_path,
                chunk["source_type"],
                chunk_index,
                chunk.get("parent_id"),
                chunk["content"],
                chunk["content_hash"],
                np.asarray(embedding, dtype=np.float32),
                json.dumps(chunk.get("metadata", {})),
                chunk.get("token_count", 0),
            )
            await conn.execute(
                _ATOMIC_SPARSE_UPSERT_SQL,
                chunk_id,
                json.dumps(weights),
            )
    return len(chunks)


async def get_chunk_ids_by_source(source_path: str, namespace_id: int | None = None) -> list[str]:
    """Get chunk IDs for a source path, ordered by chunk_index."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if namespace_id is None:
            rows = await conn.fetch(
                "SELECT id::text AS chunk_id FROM chunks WHERE source_path = $1 ORDER BY chunk_index",
                source_path,
            )
        else:
            rows = await conn.fetch(
                "SELECT id::text AS chunk_id FROM chunks WHERE namespace_id=$1 AND source_path=$2 ORDER BY chunk_index",
                namespace_id,
                source_path,
            )
    return [row["chunk_id"] for row in rows]


async def get_structured_graph_hash(namespace_id: int, source_path: str) -> str | None:
    """Return the last committed structured graph hash for one namespace/source."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT content_hash FROM structured_graph_sources WHERE namespace_id = $1 AND source_path = $2",
            namespace_id,
            source_path,
        )


async def get_source_graph_provenance(namespace_id: int, source_path: str) -> dict[str, list]:
    """Snapshot legacy graph IDs before reindexing removes their source chunks."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            WITH source_chunks AS (
                SELECT id
                FROM chunks c
                WHERE c.namespace_id = $1 AND c.source_path = $2
            )
            SELECT
                COALESCE((
                    SELECT array_agg(e.id::text ORDER BY e.id::text)
                    FROM entities e
                    WHERE e.namespace_id = $1 AND e.source_chunk_id IN (SELECT id FROM source_chunks)
                ), ARRAY[]::text[]) AS entity_ids,
                COALESCE((
                    SELECT array_agg(ee.id ORDER BY ee.id)
                    FROM entity_edges ee
                    JOIN entities source_entity ON source_entity.id = ee.source_entity_id
                    WHERE source_entity.namespace_id = $1
                      AND ee.source_chunk_id IN (SELECT id FROM source_chunks)
                ), ARRAY[]::int[]) AS edge_ids
            """,
            namespace_id,
            source_path,
        )
    return {"entity_ids": list(row["entity_ids"]), "edge_ids": list(row["edge_ids"])}


async def apply_structured_graph(
    namespace_id: int,
    source_path: str,
    content_hash: str,
    entities: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    source_chunk_id: str | None = None,
    prior_entity_ids: list[str] | None = None,
    prior_edge_ids: list[int] | None = None,
) -> dict[str, int | bool]:
    """Atomically upsert and converge a deterministic graph for one source."""
    pool = await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended($1::int::text || ':' || $2, 0))",
            namespace_id,
            source_path,
        )
        current_hash = await conn.fetchval(
            "SELECT content_hash FROM structured_graph_sources WHERE namespace_id = $1 AND source_path = $2",
            namespace_id,
            source_path,
        )
        if current_hash == content_hash:
            return {"entities_upserted": 0, "edges_upserted": 0, "idempotent_noop": True}

        # Seed provenance captured before chunk replacement, including legacy rows
        # that predate the source-association tables.
        if prior_entity_ids:
            await conn.execute(
                """
                INSERT INTO entity_sources (
                    entity_id, namespace_id, source_path, content_hash, source_chunk_id, updated_at
                )
                SELECT id, $1, $2, $3, NULL, NOW()
                FROM entities
                WHERE namespace_id = $1 AND id = ANY($4::uuid[])
                ON CONFLICT (entity_id, source_path) DO NOTHING
                """,
                namespace_id,
                source_path,
                current_hash or content_hash,
                prior_entity_ids,
            )
        if prior_edge_ids:
            await conn.execute(
                """
                INSERT INTO entity_edge_sources (
                    edge_id, namespace_id, source_path, content_hash, source_chunk_id, updated_at
                )
                SELECT ee.id, $1, $2, $3, NULL, NOW()
                FROM entity_edges ee
                JOIN entities source_entity ON source_entity.id = ee.source_entity_id
                WHERE source_entity.namespace_id = $1 AND ee.id = ANY($4::int[])
                ON CONFLICT (edge_id, source_path) DO NOTHING
                """,
                namespace_id,
                source_path,
                current_hash or content_hash,
                prior_edge_ids,
            )

        entity_ids_by_name: dict[str, str] = {}
        for entity in entities:
            row = await conn.fetchrow(
                """
                INSERT INTO entities (
                    namespace_id, name, entity_type, description, properties, source_chunk_id
                )
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::uuid)
                ON CONFLICT (namespace_id, name, entity_type)
                DO UPDATE SET
                    description = COALESCE(EXCLUDED.description, entities.description),
                    properties = COALESCE(entities.properties, '{}'::jsonb) || EXCLUDED.properties,
                    source_chunk_id = COALESCE(EXCLUDED.source_chunk_id, entities.source_chunk_id),
                    updated_at = NOW()
                RETURNING id::text AS id
                """,
                namespace_id,
                entity["name"],
                entity["entity_type"],
                entity.get("description"),
                json.dumps(entity.get("properties") or {}),
                source_chunk_id,
            )
            entity_id = row["id"]
            entity_ids_by_name[entity["name"]] = entity_id
            await conn.execute(
                """
                INSERT INTO entity_sources (
                    entity_id, namespace_id, source_path, content_hash, source_chunk_id, updated_at
                ) VALUES ($1::uuid, $2, $3, $4, $5::uuid, NOW())
                ON CONFLICT (entity_id, source_path) DO UPDATE SET
                    namespace_id = EXCLUDED.namespace_id,
                    content_hash = EXCLUDED.content_hash,
                    source_chunk_id = EXCLUDED.source_chunk_id,
                    updated_at = NOW()
                """,
                entity_id,
                namespace_id,
                source_path,
                content_hash,
                source_chunk_id,
            )

        edge_ids: list[int] = []
        for edge in edges:
            row = await conn.fetchrow(
                """
                INSERT INTO entity_edges (
                    source_entity_id, target_entity_id, relation, weight, source_chunk_id
                ) VALUES ($1::uuid, $2::uuid, $3, 1.0, $4::uuid)
                ON CONFLICT (source_entity_id, target_entity_id, relation)
                DO UPDATE SET
                    weight = 1.0,
                    source_chunk_id = COALESCE(EXCLUDED.source_chunk_id, entity_edges.source_chunk_id)
                RETURNING id
                """,
                entity_ids_by_name[edge["source"]],
                entity_ids_by_name[edge["target"]],
                edge["relation"],
                source_chunk_id,
            )
            edge_id = row["id"]
            edge_ids.append(edge_id)
            await conn.execute(
                """
                INSERT INTO entity_edge_sources (
                    edge_id, namespace_id, source_path, content_hash, source_chunk_id, updated_at
                ) VALUES ($1, $2, $3, $4, $5::uuid, NOW())
                ON CONFLICT (edge_id, source_path) DO UPDATE SET
                    namespace_id = EXCLUDED.namespace_id,
                    content_hash = EXCLUDED.content_hash,
                    source_chunk_id = EXCLUDED.source_chunk_id,
                    updated_at = NOW()
                """,
                edge_id,
                namespace_id,
                source_path,
                content_hash,
                source_chunk_id,
            )

        current_entity_ids = list(entity_ids_by_name.values())
        await conn.execute(
            """
            DELETE FROM entity_sources
            WHERE namespace_id = $1 AND source_path = $2
              AND NOT (entity_id = ANY($3::uuid[]))
            """,
            namespace_id,
            source_path,
            current_entity_ids,
        )
        removed_edge_rows = await conn.fetch(
            """
            DELETE FROM entity_edge_sources
            WHERE namespace_id = $1 AND source_path = $2
              AND NOT (edge_id = ANY($3::int[]))
            RETURNING edge_id
            """,
            namespace_id,
            source_path,
            edge_ids,
        )
        removed_edge_ids = [row["edge_id"] for row in removed_edge_rows]
        if removed_edge_ids:
            await conn.execute(
                """
                DELETE FROM entity_edges ee
                WHERE ee.id = ANY($1::int[])
                  AND NOT EXISTS (
                      SELECT 1 FROM entity_edge_sources ees WHERE ees.edge_id = ee.id
                  )
                """,
                removed_edge_ids,
            )

        # Publish the hash only after every graph mutation succeeds.
        await conn.execute(
            """
            INSERT INTO structured_graph_sources (namespace_id, source_path, content_hash, updated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (namespace_id, source_path) DO UPDATE SET
                content_hash = EXCLUDED.content_hash,
                updated_at = NOW()
            """,
            namespace_id,
            source_path,
            content_hash,
        )
        return {
            "entities_upserted": len(entities),
            "edges_upserted": len(edges),
            "idempotent_noop": False,
        }


def _command_count(status: str) -> int:
    """Extract asyncpg's affected-row count from a command status."""
    return int(status.rsplit(" ", 1)[-1])


async def delete_ltm_source(namespace_id: int, source_path: str) -> dict[str, int | bool]:
    """Atomically remove one source while preserving shared graph ownership."""
    pool = await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended($1::int::text || ':' || $2, 0))",
            namespace_id,
            source_path,
        )
        entity_rows = await conn.fetch(
            """
            SELECT candidate.entity_id::text AS entity_id
            FROM (
                SELECT es.entity_id
                FROM entity_sources es
                WHERE es.namespace_id = $1 AND es.source_path = $2
                UNION
                SELECT e.id
                FROM entities e
                JOIN chunks c ON c.id = e.source_chunk_id
                WHERE c.namespace_id = $1 AND c.source_path = $2
            ) candidate
            """,
            namespace_id,
            source_path,
        )
        edge_rows = await conn.fetch(
            """
            SELECT candidate.edge_id
            FROM (
                SELECT ees.edge_id
                FROM entity_edge_sources ees
                WHERE ees.namespace_id = $1 AND ees.source_path = $2
                UNION
                SELECT ee.id
                FROM entity_edges ee
                JOIN chunks c ON c.id = ee.source_chunk_id
                WHERE c.namespace_id = $1 AND c.source_path = $2
            ) candidate
            """,
            namespace_id,
            source_path,
        )
        candidate_entity_ids = [row["entity_id"] for row in entity_rows]
        candidate_edge_ids = [row["edge_id"] for row in edge_rows]

        edge_sources_deleted = _command_count(
            await conn.execute(
                "DELETE FROM entity_edge_sources WHERE namespace_id = $1 AND source_path = $2",
                namespace_id,
                source_path,
            )
        )
        entity_sources_deleted = _command_count(
            await conn.execute(
                "DELETE FROM entity_sources WHERE namespace_id = $1 AND source_path = $2",
                namespace_id,
                source_path,
            )
        )
        chunks_deleted = _command_count(
            await conn.execute(
                "DELETE FROM chunks WHERE namespace_id = $1 AND source_path = $2",
                namespace_id,
                source_path,
            )
        )
        # SRCH-0039 pre-merge review (NOTE A): remove the exact-bytes row too, so deleting a source
        # cannot orphan or later resurrect a stale evidence_documents row. Same transaction.
        await conn.execute(_ATOMIC_EVIDENCE_DOCUMENT_DELETE_SQL, namespace_id, source_path)
        edges_deleted = _command_count(
            await conn.execute(
                """
                DELETE FROM entity_edges ee
                WHERE ee.id = ANY($1::int[])
                  AND ee.source_chunk_id IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM entity_edge_sources ees WHERE ees.edge_id = ee.id
                  )
                """,
                candidate_edge_ids,
            )
        )
        entities_deleted = _command_count(
            await conn.execute(
                """
                DELETE FROM entities e
                WHERE e.id = ANY($1::uuid[])
                  AND NOT EXISTS (
                      SELECT 1 FROM entity_sources es WHERE es.entity_id = e.id
                  )
                  AND e.source_chunk_id IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM entity_edges ee
                      WHERE ee.source_entity_id = e.id OR ee.target_entity_id = e.id
                  )
                """,
                candidate_entity_ids,
            )
        )
        # The structured hash is the committed source marker. Remove it only
        # after every associated graph/chunk mutation has succeeded.
        hashes_deleted = _command_count(
            await conn.execute(
                "DELETE FROM structured_graph_sources WHERE namespace_id = $1 AND source_path = $2",
                namespace_id,
                source_path,
            )
        )
        counts = {
            "chunks_deleted": chunks_deleted,
            "entity_sources_deleted": entity_sources_deleted,
            "edge_sources_deleted": edge_sources_deleted,
            "edges_deleted": edges_deleted,
            "entities_deleted": entities_deleted,
        }
        return {**counts, "idempotent_noop": not any((*counts.values(), hashes_deleted))}


async def delete_by_source(source_path: str, namespace_id: int) -> int:
    """Delete all chunks for a given source path. Returns deleted count."""
    pool = await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        result = await conn.execute(
            "DELETE FROM chunks WHERE namespace_id=$1 AND source_path=$2",
            namespace_id,
            source_path,
        )
        # SRCH-0039 pre-merge review (NOTE A): drop the exact-bytes row in the same transaction so a
        # source delete/re-create can't leave a stale evidence_documents row behind.
        await conn.execute(_ATOMIC_EVIDENCE_DOCUMENT_DELETE_SQL, namespace_id, source_path)
        return int(result.split()[-1])


async def hybrid_search(
    query_embedding: list[float],
    query_text: str,
    namespace_id: int,
    limit: int = 10,
    query_sparse: dict[str, float] | None = None,
    fetch_multiplier: int = 3,
    return_pool: bool = False,
) -> list[SearchResult]:
    """Hybrid search: dense cosine + sparse lexical + FTS with RRF ranking.

    When query_sparse is provided, uses 3-way RRF (dense + sparse + FTS).
    Otherwise falls back to 2-way RRF (dense + FTS).

    SRCH-0029 M2 params:
    - fetch_multiplier: candidate pre-fetch factor (default 3 = byte-identical to prior behaviour).
      Set to settings.rerank_pool_multiplier when rerank_enabled=True (wider recall pool).
    - return_pool: when True, the SQL final LIMIT returns fetch_limit rows (full pool for reranker).
      When False (default), final LIMIT is `limit` (existing behaviour).
    """
    pool = await get_pool()
    vector = np.array(query_embedding, dtype=np.float32)
    fetch_limit = limit * fetch_multiplier
    # SQL $5 target: full pool for reranker, or final limit otherwise
    sql_final_limit = fetch_limit if return_pool else limit

    if query_sparse:
        # 3-way RRF: dense + sparse + FTS
        sparse_json = json.dumps(query_sparse)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH semantic AS (
                    SELECT c.id, ROW_NUMBER() OVER (
                        ORDER BY c.embedding_dense <=> $1, c.id ASC
                    ) AS rank
                    FROM chunks c
                    WHERE c.namespace_id = $2
                      AND c.embedding_dense IS NOT NULL
                    ORDER BY c.embedding_dense <=> $1, c.id ASC
                    LIMIT $3
                ),
                fulltext AS (
                    SELECT c.id, ROW_NUMBER() OVER (
                        ORDER BY ts_rank_cd(c.textsearch_ru, plainto_tsquery('russian', $4))
                               + ts_rank_cd(c.textsearch_en, plainto_tsquery('english', $4)) DESC,
                                 c.id ASC
                    ) AS rank
                    FROM chunks c
                    WHERE c.namespace_id = $2
                      AND (c.textsearch_ru @@ plainto_tsquery('russian', $4)
                           OR c.textsearch_en @@ plainto_tsquery('english', $4))
                    ORDER BY ts_rank_cd(c.textsearch_ru, plainto_tsquery('russian', $4))
                             + ts_rank_cd(c.textsearch_en, plainto_tsquery('english', $4)) DESC,
                               c.id ASC
                    LIMIT $3
                ),
                sparse_match AS (
                    SELECT sv.chunk_id AS id, ROW_NUMBER() OVER (
                        ORDER BY (
                            SELECT SUM(
                                COALESCE((sv.token_weights->>key)::real, 0) * value::real
                            )
                            FROM jsonb_each_text($6::jsonb) AS q(key, value)
                        ) DESC, sv.chunk_id ASC
                    ) AS rank
                    FROM sparse_vectors sv
                    JOIN chunks c ON c.id = sv.chunk_id
                    WHERE c.namespace_id = $2
                    ORDER BY (
                        SELECT SUM(
                            COALESCE((sv.token_weights->>key)::real, 0) * value::real
                        )
                        FROM jsonb_each_text($6::jsonb) AS q(key, value)
                    ) DESC, sv.chunk_id ASC
                    LIMIT $3
                ),
                ranked AS (
                    SELECT
                        COALESCE(s.id, COALESCE(f.id, sp.id)) AS chunk_id,
                        COALESCE(1.0 / (60 + s.rank), 0.0)
                            + COALESCE(1.0 / (60 + f.rank), 0.0)
                            + COALESCE(1.0 / (60 + sp.rank), 0.0) AS rrf_score
                    FROM semantic s
                    FULL OUTER JOIN fulltext f ON s.id = f.id
                    FULL OUTER JOIN sparse_match sp ON COALESCE(s.id, f.id) = sp.id
                    ORDER BY rrf_score DESC, chunk_id ASC
                    LIMIT $5
                )
                SELECT
                    r.chunk_id, r.rrf_score,
                    c.content, c.source_path, c.source_type, c.chunk_index,
                    c.metadata, n.name AS namespace_name,
                    p.name AS project_name
                FROM ranked r
                JOIN chunks c ON c.id = r.chunk_id
                JOIN namespaces n ON n.id = c.namespace_id
                LEFT JOIN projects p ON p.id = c.project_id
                ORDER BY r.rrf_score DESC, r.chunk_id ASC
                """,
                vector,
                namespace_id,
                fetch_limit,
                query_text,
                sql_final_limit,
                sparse_json,
            )
    else:
        # 2-way RRF: dense + FTS (backward-compatible)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH semantic AS (
                    SELECT c.id, ROW_NUMBER() OVER (
                        ORDER BY c.embedding_dense <=> $1, c.id ASC
                    ) AS rank
                    FROM chunks c
                    WHERE c.namespace_id = $2
                      AND c.embedding_dense IS NOT NULL
                    ORDER BY c.embedding_dense <=> $1, c.id ASC
                    LIMIT $3
                ),
                fulltext AS (
                    SELECT c.id, ROW_NUMBER() OVER (
                        ORDER BY ts_rank_cd(c.textsearch_ru, plainto_tsquery('russian', $4))
                               + ts_rank_cd(c.textsearch_en, plainto_tsquery('english', $4)) DESC,
                                 c.id ASC
                    ) AS rank
                    FROM chunks c
                    WHERE c.namespace_id = $2
                      AND (c.textsearch_ru @@ plainto_tsquery('russian', $4)
                           OR c.textsearch_en @@ plainto_tsquery('english', $4))
                    ORDER BY ts_rank_cd(c.textsearch_ru, plainto_tsquery('russian', $4))
                             + ts_rank_cd(c.textsearch_en, plainto_tsquery('english', $4)) DESC,
                               c.id ASC
                    LIMIT $3
                ),
                ranked AS (
                    SELECT
                        COALESCE(s.id, f.id) AS chunk_id,
                        COALESCE(1.0 / (60 + s.rank), 0.0)
                            + COALESCE(1.0 / (60 + f.rank), 0.0) AS rrf_score
                    FROM semantic s
                    FULL OUTER JOIN fulltext f ON s.id = f.id
                    ORDER BY rrf_score DESC, chunk_id ASC
                    LIMIT $5
                )
                SELECT
                    r.chunk_id, r.rrf_score,
                    c.content, c.source_path, c.source_type, c.chunk_index,
                    c.metadata, n.name AS namespace_name,
                    p.name AS project_name
                FROM ranked r
                JOIN chunks c ON c.id = r.chunk_id
                JOIN namespaces n ON n.id = c.namespace_id
                LEFT JOIN projects p ON p.id = c.project_id
                ORDER BY r.rrf_score DESC, r.chunk_id ASC
                """,
                vector,
                namespace_id,
                fetch_limit,
                query_text,
                sql_final_limit,
            )

    results = []
    for row in rows:
        meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"] or {})
        source_id, content_hash = doc_fields_from_metadata(meta)
        results.append(
            SearchResult(
                chunk_id=str(row["chunk_id"]),
                content=row["content"],
                source_path=row["source_path"],
                source_type=row["source_type"],
                chunk_index=row["chunk_index"],
                score=float(row["rrf_score"]),
                namespace=row["namespace_name"],
                project=row["project_name"],
                metadata=meta,
                heading_hierarchy=meta.get("heading_hierarchy", []),
                source_id=source_id,
                content_hash=content_hash,
            )
        )
    return results


async def insert_edges(edges: list[dict[str, Any]], allowed_namespace_ids: frozenset[int] | None = None) -> int:
    """Batch insert graph edges. ON CONFLICT → update weight. Returns count."""
    if not edges:
        return 0
    pool = await get_pool()
    inserted = 0
    async with pool.acquire() as conn:
        for edge in edges:
            if allowed_namespace_ids is not None:
                permitted = await conn.fetchval(
                    """
                    SELECT NOT EXISTS (
                        SELECT 1
                        FROM unnest($1::uuid[]) requested(id)
                        LEFT JOIN chunks c
                          ON c.id = requested.id
                         AND c.namespace_id = ANY($2::int[])
                        WHERE c.id IS NULL
                    )
                    """,
                    [edge["source_chunk_id"], edge["target_chunk_id"]],
                    sorted(allowed_namespace_ids),
                )
                if not permitted:
                    continue
            await conn.execute(
                """
                INSERT INTO graph_edges (source_chunk_id, target_chunk_id, edge_type, weight, created_by)
                VALUES ($1::uuid, $2::uuid, $3, $4, $5)
                ON CONFLICT (source_chunk_id, target_chunk_id, edge_type)
                DO UPDATE SET weight = EXCLUDED.weight
                """,
                edge["source_chunk_id"],
                edge["target_chunk_id"],
                edge["edge_type"],
                edge.get("weight", 1.0),
                edge.get("created_by", "dreamer"),
            )
            inserted += 1
    return inserted


async def get_edges_for_chunk(
    chunk_id: str, allowed_namespace_ids: frozenset[int] | None = None
) -> list[dict[str, Any]]:
    """Get all edges (inbound + outbound) for a chunk."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if allowed_namespace_ids is None:
            rows = await conn.fetch(
                """
            SELECT id, source_chunk_id::text, target_chunk_id::text,
                   edge_type, weight, created_by, created_at::text
            FROM graph_edges
            WHERE source_chunk_id = $1::uuid OR target_chunk_id = $1::uuid
            ORDER BY created_at DESC
                """,
                chunk_id,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT g.id, g.source_chunk_id::text, g.target_chunk_id::text,
                       g.edge_type, g.weight, g.created_by, g.created_at::text
                FROM graph_edges g
                JOIN chunks requested
                  ON requested.id = $1::uuid
                 AND requested.namespace_id = ANY($2::int[])
                JOIN chunks source ON source.id = g.source_chunk_id
                JOIN chunks target ON target.id = g.target_chunk_id
                WHERE (g.source_chunk_id = requested.id OR g.target_chunk_id = requested.id)
                  AND source.namespace_id = ANY($2::int[])
                  AND target.namespace_id = ANY($2::int[])
                ORDER BY g.created_at DESC
                """,
                chunk_id,
                sorted(allowed_namespace_ids),
            )
    return [dict(r) for r in rows]


async def delete_edges_by_creator(created_by: str, namespace_id: int | None = None) -> int:
    """Delete edges created by a specific agent. Optional namespace filter."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if namespace_id is not None:
            result = await conn.execute(
                """
                DELETE FROM graph_edges g
                USING chunks source, chunks target
                WHERE g.source_chunk_id = source.id
                  AND g.target_chunk_id = target.id
                  AND g.created_by = $1
                  AND source.namespace_id = $2
                  AND target.namespace_id = $2
                """,
                created_by,
                namespace_id,
            )
        else:
            result = await conn.execute(
                "DELETE FROM graph_edges WHERE created_by = $1",
                created_by,
            )
        return int(result.split()[-1])


async def find_similar_pairs(namespace_id: int, threshold: float = 0.92, limit: int = 50) -> list[dict[str, Any]]:
    """Find chunk pairs with cosine similarity > threshold within a namespace."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                a.id::text AS chunk_id_a,
                b.id::text AS chunk_id_b,
                1 - (a.embedding_dense <=> b.embedding_dense) AS similarity,
                a.source_path AS source_path_a,
                b.source_path AS source_path_b,
                LEFT(a.content, 200) AS content_a,
                LEFT(b.content, 200) AS content_b
            FROM chunks a
            JOIN chunks b ON a.id < b.id
                AND a.namespace_id = b.namespace_id
                AND a.source_path != b.source_path
            WHERE a.namespace_id = $1
              AND a.embedding_dense IS NOT NULL
              AND b.embedding_dense IS NOT NULL
              AND 1 - (a.embedding_dense <=> b.embedding_dense) > $2
            ORDER BY similarity DESC
            LIMIT $3
            """,
            namespace_id,
            threshold,
            limit,
        )
    return [dict(r) for r in rows]


async def get_orphan_chunks(namespace_id: int, limit: int = 50) -> list[dict[str, Any]]:
    """Find chunks with zero graph edges."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                c.id::text AS chunk_id,
                c.source_path,
                0 AS edge_count,
                c.created_at::text
            FROM chunks c
            LEFT JOIN graph_edges g_out ON g_out.source_chunk_id = c.id
            LEFT JOIN graph_edges g_in ON g_in.target_chunk_id = c.id
            WHERE c.namespace_id = $1
              AND g_out.id IS NULL
              AND g_in.id IS NULL
            ORDER BY c.created_at ASC
            LIMIT $2
            """,
            namespace_id,
            limit,
        )
    return [dict(r) for r in rows]


async def find_stale_chunks(namespace_id: int, stale_days: int = 90, limit: int = 50) -> list[dict[str, Any]]:
    """Find chunks not updated in stale_days days."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                c.id::text AS chunk_id,
                c.source_path,
                EXTRACT(DAY FROM NOW() - c.updated_at)::int AS days_since_update,
                COUNT(g.id)::int AS edge_count
            FROM chunks c
            LEFT JOIN graph_edges g ON g.source_chunk_id = c.id OR g.target_chunk_id = c.id
            WHERE c.namespace_id = $1
              AND c.updated_at < NOW() - MAKE_INTERVAL(days => $2)
            GROUP BY c.id, c.source_path, c.updated_at
            ORDER BY c.updated_at ASC
            LIMIT $3
            """,
            namespace_id,
            stale_days,
            limit,
        )
    return [dict(r) for r in rows]


async def get_edge_stats(namespace_id: int | None = None) -> dict[str, Any]:
    """Edge statistics: total count, breakdown by type."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if namespace_id is not None:
            total = await conn.fetchval(
                """
                SELECT COUNT(*)::int FROM graph_edges g
                JOIN chunks c ON g.source_chunk_id = c.id
                WHERE c.namespace_id = $1
                """,
                namespace_id,
            )
            by_type = await conn.fetch(
                """
                SELECT g.edge_type, COUNT(*)::int AS count,
                       AVG(g.weight)::real AS avg_weight
                FROM graph_edges g
                JOIN chunks c ON g.source_chunk_id = c.id
                WHERE c.namespace_id = $1
                GROUP BY g.edge_type ORDER BY count DESC
                """,
                namespace_id,
            )
        else:
            total = await conn.fetchval("SELECT COUNT(*)::int FROM graph_edges")
            by_type = await conn.fetch(
                """
                SELECT edge_type, COUNT(*)::int AS count,
                       AVG(weight)::real AS avg_weight
                FROM graph_edges GROUP BY edge_type ORDER BY count DESC
                """
            )
    return {
        "total_edges": total or 0,
        "by_type": [dict(r) for r in by_type],
    }


async def get_namespaces(namespace_ids: frozenset[int]) -> list[NamespaceInfo]:
    """List namespaces within namespace_ids, with chunk counts. Empty set -> empty result
    (SRCH-0023 V-AC-6 — never enumerate outside the caller's allowed-set)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT n.id, n.name, n.description,
                   COUNT(c.id)::int AS chunk_count
            FROM namespaces n
            LEFT JOIN chunks c ON c.namespace_id = n.id
            WHERE n.id = ANY($1::int[])
            GROUP BY n.id, n.name, n.description
            ORDER BY n.name
            """,
            list(namespace_ids),
        )
    return [
        NamespaceInfo(id=r["id"], name=r["name"], description=r["description"], chunk_count=r["chunk_count"])
        for r in rows
    ]


async def get_stats(namespace_ids: frozenset[int]) -> dict[str, Any]:
    """Get index statistics scoped to namespace_ids. Empty set -> zeroed stats
    (SRCH-0023 V-AC-6 — never enumerate outside the caller's allowed-set)."""
    pool = await get_pool()
    ns_id_list = list(namespace_ids)
    async with pool.acquire() as conn:
        total_chunks = await conn.fetchval(
            "SELECT COUNT(*)::int FROM chunks WHERE namespace_id = ANY($1::int[])", ns_id_list
        )
        total_namespaces = await conn.fetchval(
            "SELECT COUNT(*)::int FROM namespaces WHERE id = ANY($1::int[])", ns_id_list
        )
        total_projects = await conn.fetchval(
            "SELECT COUNT(*)::int FROM projects WHERE namespace_id = ANY($1::int[])", ns_id_list
        )
        ns_rows = await conn.fetch(
            """
            SELECT n.name,
                   COUNT(c.id)::int AS chunk_count,
                   COUNT(DISTINCT c.project_id)::int AS project_count
            FROM namespaces n
            LEFT JOIN chunks c ON c.namespace_id = n.id
            WHERE n.id = ANY($1::int[])
            GROUP BY n.name
            ORDER BY n.name
            """,
            ns_id_list,
        )
    return {
        "total_chunks": total_chunks or 0,
        "total_namespaces": total_namespaces or 0,
        "total_projects": total_projects or 0,
        "namespaces": [
            NamespaceStats(name=r["name"], chunk_count=r["chunk_count"], project_count=r["project_count"])
            for r in ns_rows
        ],
    }


async def get_chunks_by_source_path(
    source_path: str,
    namespace_id: int | None = None,
) -> list[ChunkLookupResult]:
    """Lookup chunks by source_path. Returns chunk info ordered by chunk_index."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if namespace_id is not None:
            rows = await conn.fetch(
                """
                SELECT id::text AS chunk_id, chunk_index, source_type, source_path,
                       LEFT(content, 200) AS content_preview, metadata
                FROM chunks
                WHERE source_path = $1 AND namespace_id = $2
                ORDER BY chunk_index
                """,
                source_path,
                namespace_id,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id::text AS chunk_id, chunk_index, source_type, source_path,
                       LEFT(content, 200) AS content_preview, metadata
                FROM chunks
                WHERE source_path = $1
                ORDER BY chunk_index
                """,
                source_path,
            )
    results = []
    for row in rows:
        meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"] or {})
        results.append(
            ChunkLookupResult(
                chunk_id=row["chunk_id"],
                chunk_index=row["chunk_index"],
                source_path=row["source_path"],
                source_type=row["source_type"],
                content_preview=row["content_preview"] or "",
                metadata=meta,
            )
        )
    return results


def _row_to_chunk_lookup(row: Any) -> ChunkLookupResult:
    meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"] or {})
    return ChunkLookupResult(
        chunk_id=row["chunk_id"],
        chunk_index=row["chunk_index"],
        source_path=row["source_path"],
        source_type=row["source_type"],
        content_preview=row["content_preview"] or "",
        metadata=meta,
    )


async def get_section_siblings_children(chunk_id: str, allowed_namespace_ids: frozenset[int]) -> dict[str, Any] | None:
    """Fetch the target chunk's document-scoped row set for section-context assembly.

    SRCH-0021 (V-AC-4): looks up the target chunk, then all chunks sharing its
    `metadata.section.doc_id`; navigator.build_section_context derives
    ancestors/self/siblings/children from the returned rows in-memory.
    Falls back to grouping by `(namespace_id, source_path)` when the target
    chunk has no `section` key yet (un-backfilled document — PRD Risk table).
    Returns None if chunk_id does not exist. Parameterized throughout — no
    f-string/`%`-format SQL (V-AC-8).
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        self_row = await conn.fetchrow(
            """
            SELECT id::text AS chunk_id, chunk_index, source_type, source_path,
                   namespace_id, LEFT(content, 200) AS content_preview, metadata
            FROM chunks
            WHERE id = $1::uuid AND namespace_id = ANY($2::int[])
            """,
            chunk_id,
            list(allowed_namespace_ids),
        )
        if self_row is None:
            return None

        namespace_id = self_row["namespace_id"]
        meta = (
            json.loads(self_row["metadata"])
            if isinstance(self_row["metadata"], str)
            else dict(self_row["metadata"] or {})
        )
        doc_id = (meta.get("section") or {}).get("doc_id") or None

        if doc_id:
            rows = await conn.fetch(
                """
                SELECT id::text AS chunk_id, chunk_index, source_type, source_path,
                       LEFT(content, 200) AS content_preview, metadata
                FROM chunks
                WHERE namespace_id = $1 AND metadata->'section'->>'doc_id' = $2
                ORDER BY chunk_index
                """,
                namespace_id,
                doc_id,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id::text AS chunk_id, chunk_index, source_type, source_path,
                       LEFT(content, 200) AS content_preview, metadata
                FROM chunks
                WHERE namespace_id = $1 AND source_path = $2
                ORDER BY chunk_index
                """,
                namespace_id,
                self_row["source_path"],
            )

    return {"doc_rows": [_row_to_chunk_lookup(row) for row in rows]}


# ── SRCH-0038: exact whole-document fetch-by-id (namespace-scoped, S2/S3) ─────────────

_FETCH_CHUNK_COLUMNS = """
    c.id::text AS chunk_id, c.chunk_index, c.content, c.content_hash,
    c.source_path, c.source_type, c.token_count,
    c.metadata, c.indexed_at::text AS indexed_at, n.name AS namespace
"""


def _fetch_row_to_dict(row: Any) -> dict[str, Any]:
    meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"] or {})
    return {
        "chunk_id": row["chunk_id"],
        "chunk_index": row["chunk_index"],
        "content": row["content"],
        "content_hash": row["content_hash"],
        "source_path": row["source_path"],
        "source_type": row["source_type"],
        "token_count": row["token_count"] or 0,
        "metadata": meta,
        "indexed_at": row["indexed_at"],
        "namespace": row["namespace"],
    }


async def fetch_source_raw_content(doc_id: str, allowed_namespace_ids: frozenset[int]) -> str | None:
    """Return the exact whole-document bytes for a skills doc (SRCH-0038 1b), namespace-scoped.

    Parameterized equality on the opaque namespace-scoped ``doc_id`` (S3, no path join) filtered to
    the caller's allowed namespaces (S2). An empty allowed-set or an unknown / cross-namespace
    ``doc_id`` returns ``None`` → the fetcher fails closed (409) rather than returning a hash-failing
    reassembly. Reads from ``source_documents`` (NOT the GIN-indexed ``chunks.metadata``)."""
    if not allowed_namespace_ids or not doc_id:
        return None
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT raw_content FROM source_documents WHERE doc_id = $1 AND namespace_id = ANY($2::int[])",
            doc_id,
            list(allowed_namespace_ids),
        )


async def fetch_evidence_raw_content(doc_id: str, allowed_namespace_ids: frozenset[int]) -> tuple[str, str] | None:
    """Return ``(raw_content, content_hash)`` for an evidence doc (SRCH-0039), namespace-scoped, or
    ``None`` when absent.

    Mirrors ``fetch_source_raw_content`` — the SAME fail-closed, namespace-gated predicate
    ``WHERE doc_id = $1 AND namespace_id = ANY($2::int[])`` so the byte-fetch and the authz check
    are one read (authorize-before-bytes: ``content_hash`` is never a request field, only resolved
    internally). Reads from the isolated ``evidence_documents`` table (NOT the GIN-indexed
    ``chunks.metadata``).

    The row's ``content_hash`` is returned ALONGSIDE the bytes so the fetcher can compare it to the
    current chunk stamp and reject a STALE row (a hash-to-hash comparison of two bound-at-write
    values, NOT a body re-hash) — defense-in-depth against a stale exact-bytes row (SRCH-0039
    pre-merge review). An empty allowed-set, an unknown / cross-namespace ``doc_id``, or a not-yet-
    backfilled doc returns ``None`` → the fetcher GRACEFULLY DEGRADES to reassembly
    (``content_exact=False``), deliberately NOT the skills fail-closed 409 (evidence row-absence is
    an expected pre-backfill state on the huge existing corpus)."""
    if not allowed_namespace_ids or not doc_id:
        return None
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT raw_content, content_hash FROM evidence_documents "
            "WHERE doc_id = $1 AND namespace_id = ANY($2::int[])",
            doc_id,
            list(allowed_namespace_ids),
        )
    if row is None:
        return None
    return row["raw_content"], row["content_hash"]


async def fetch_chunks_by_doc_id(doc_id: str, allowed_namespace_ids: frozenset[int]) -> list[dict[str, Any]]:
    """Return every chunk of a document, ordered by chunk_index, scoped to the caller's
    allowed namespaces (SRCH-0038 S2). Lookup is a parameterized equality on the opaque
    ``metadata->'section'->>'doc_id'`` — never a filesystem-path join (S3). An empty
    allowed-set (grace-window / zero-grant principal) fetches nothing (fail-closed); an
    unknown or cross-namespace ``doc_id`` returns ``[]`` → the endpoint answers 404 (no
    existence oracle)."""
    if not allowed_namespace_ids:
        return []
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT {_FETCH_CHUNK_COLUMNS}
            FROM chunks c
            JOIN namespaces n ON n.id = c.namespace_id
            WHERE c.metadata->'section'->>'doc_id' = $1
              AND c.namespace_id = ANY($2::int[])
            ORDER BY c.chunk_index
            """,
            doc_id,
            list(allowed_namespace_ids),
        )
    return [_fetch_row_to_dict(row) for row in rows]


async def fetch_chunks_by_chunk_id(chunk_id: str, allowed_namespace_ids: frozenset[int]) -> list[dict[str, Any]]:
    """Resolve a chunk UUID to its parent document, then return the whole document's chunks
    ordered by chunk_index (SRCH-0038, mirrors ``get_section_siblings_children`` scoping).
    Namespace-scoped throughout (S2); parameterized equality only (S3). Falls back to
    ``(namespace_id, source_path)`` grouping when the target chunk has no ``section`` key
    (un-backfilled). Returns ``[]`` when the chunk is unknown or outside the allowed-set."""
    if not allowed_namespace_ids:
        return []
    pool = await get_pool()
    async with pool.acquire() as conn:
        self_row = await conn.fetchrow(
            """
            SELECT namespace_id, source_path, metadata
            FROM chunks
            WHERE id = $1::uuid AND namespace_id = ANY($2::int[])
            """,
            chunk_id,
            list(allowed_namespace_ids),
        )
        if self_row is None:
            return []
        meta = (
            json.loads(self_row["metadata"])
            if isinstance(self_row["metadata"], str)
            else dict(self_row["metadata"] or {})
        )
        namespace_id = self_row["namespace_id"]
        doc_id = (meta.get("section") or {}).get("doc_id") or None
        if doc_id:
            rows = await conn.fetch(
                f"""
                SELECT {_FETCH_CHUNK_COLUMNS}
                FROM chunks c
                JOIN namespaces n ON n.id = c.namespace_id
                WHERE c.namespace_id = $1 AND c.metadata->'section'->>'doc_id' = $2
                ORDER BY c.chunk_index
                """,
                namespace_id,
                doc_id,
            )
        else:
            rows = await conn.fetch(
                f"""
                SELECT {_FETCH_CHUNK_COLUMNS}
                FROM chunks c
                JOIN namespaces n ON n.id = c.namespace_id
                WHERE c.namespace_id = $1 AND c.source_path = $2
                ORDER BY c.chunk_index
                """,
                namespace_id,
                self_row["source_path"],
            )
    return [_fetch_row_to_dict(row) for row in rows]


async def search_with_filters(
    query_text: str,
    namespace_id: int,
    source_type: str | None = None,
    actor: str | None = None,
    memory_type: str | None = None,
    include_expired: bool = False,
    importance_boost: bool = False,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Hybrid search with additional metadata filters for memory recall."""
    from scrutator.search.embedder import embed_single

    query_embedding = await embed_single(query_text)
    vector = np.array(query_embedding, dtype=np.float32)
    fetch_limit = limit * 3

    # Build dynamic WHERE clauses
    conditions: list[str] = []
    params: list[Any] = [vector, namespace_id, fetch_limit, query_text, limit]
    param_idx = 6  # next parameter index ($6, $7, ...)

    if source_type:
        conditions.append(f"c.source_type = ${param_idx}")
        params.append(source_type)
        param_idx += 1

    if actor:
        conditions.append(f"c.metadata->>'actor' = ${param_idx}")
        params.append(actor)
        param_idx += 1

    if memory_type:
        conditions.append(f"c.metadata->>'memory_type' = ${param_idx}")
        params.append(memory_type)
        param_idx += 1

    if not include_expired:
        conditions.append("(c.metadata->>'valid_until' IS NULL OR c.metadata->>'valid_until' > NOW()::text)")

    extra_where = ""
    if conditions:
        extra_where = " AND " + " AND ".join(conditions)

    boost_expr = "r.rrf_score"
    if importance_boost:
        boost_expr = "r.rrf_score * COALESCE((c.metadata->>'importance')::real, 0.5)"

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            WITH semantic AS (
                SELECT c.id, ROW_NUMBER() OVER (
                    ORDER BY c.embedding_dense <=> $1, c.id ASC
                ) AS rank
                FROM chunks c
                WHERE c.namespace_id = $2
                  AND c.embedding_dense IS NOT NULL
                  {extra_where}
                ORDER BY c.embedding_dense <=> $1, c.id ASC
                LIMIT $3
            ),
            fulltext AS (
                SELECT c.id, ROW_NUMBER() OVER (
                    ORDER BY ts_rank_cd(c.textsearch_ru, plainto_tsquery('russian', $4))
                           + ts_rank_cd(c.textsearch_en, plainto_tsquery('english', $4)) DESC,
                             c.id ASC
                ) AS rank
                FROM chunks c
                WHERE c.namespace_id = $2
                  AND (c.textsearch_ru @@ plainto_tsquery('russian', $4)
                       OR c.textsearch_en @@ plainto_tsquery('english', $4))
                  {extra_where}
                ORDER BY ts_rank_cd(c.textsearch_ru, plainto_tsquery('russian', $4))
                         + ts_rank_cd(c.textsearch_en, plainto_tsquery('english', $4)) DESC,
                           c.id ASC
                LIMIT $3
            ),
            ranked AS (
                SELECT
                    COALESCE(s.id, f.id) AS chunk_id,
                    COALESCE(1.0 / (60 + s.rank), 0.0)
                        + COALESCE(1.0 / (60 + f.rank), 0.0) AS rrf_score
                FROM semantic s
                FULL OUTER JOIN fulltext f ON s.id = f.id
                ORDER BY rrf_score DESC, chunk_id ASC
                LIMIT $5
            )
            SELECT
                r.chunk_id, {boost_expr} AS score,
                c.content, c.source_path, c.source_type, c.chunk_index,
                c.metadata, c.created_at::text,
                n.name AS namespace_name,
                p.name AS project_name
            FROM ranked r
            JOIN chunks c ON c.id = r.chunk_id
            JOIN namespaces n ON n.id = c.namespace_id
            LEFT JOIN projects p ON p.id = c.project_id
            ORDER BY score DESC, r.chunk_id ASC
            """,
            *params,
        )

    results = []
    for row in rows:
        meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"] or {})
        results.append(
            {
                "chunk_id": str(row["chunk_id"]),
                "content": row["content"],
                "source_path": row["source_path"],
                "source_type": row["source_type"],
                "chunk_index": row["chunk_index"],
                "score": float(row["score"]),
                "namespace": row["namespace_name"],
                "project": row["project_name"],
                "metadata": meta,
                "created_at": row["created_at"],
            }
        )
    return results


async def delete_memories_by_actor(actor: str, namespace_id: int | None = None) -> int:
    """Delete memory chunks by actor. Optional namespace filter."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if namespace_id is not None:
            result = await conn.execute(
                """
                DELETE FROM chunks
                WHERE source_type = 'memory'
                  AND metadata->>'actor' = $1
                  AND namespace_id = $2
                """,
                actor,
                namespace_id,
            )
        else:
            result = await conn.execute(
                """
                DELETE FROM chunks
                WHERE source_type = 'memory'
                  AND metadata->>'actor' = $1
                """,
                actor,
            )
        return int(result.split()[-1])


async def upsert_entity(
    namespace_id: int,
    name: str,
    entity_type: str,
    description: str | None = None,
    properties: dict | None = None,
    source_chunk_id: str | None = None,
) -> str:
    """Upsert a named entity. Returns entity UUID."""
    pool = await get_pool()
    props_json = json.dumps(properties or {})
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO entities (namespace_id, name, entity_type, description, properties, source_chunk_id)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::uuid)
            ON CONFLICT (namespace_id, name, entity_type)
            DO UPDATE SET
                description = COALESCE(EXCLUDED.description, entities.description),
                properties = entities.properties || EXCLUDED.properties,
                source_chunk_id = COALESCE(EXCLUDED.source_chunk_id, entities.source_chunk_id),
                updated_at = NOW()
            RETURNING id::text
            """,
            namespace_id,
            name,
            entity_type,
            description,
            props_json,
            source_chunk_id,
        )
        return row["id"]


async def upsert_entity_edge(
    source_entity_id: str,
    target_entity_id: str,
    relation: str,
    weight: float = 1.0,
    source_chunk_id: str | None = None,
) -> int:
    """Upsert an entity-to-entity edge. Returns edge id."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO entity_edges (source_entity_id, target_entity_id, relation, weight, source_chunk_id)
            VALUES ($1::uuid, $2::uuid, $3, $4, $5::uuid)
            ON CONFLICT (source_entity_id, target_entity_id, relation)
            DO UPDATE SET
                weight = EXCLUDED.weight,
                source_chunk_id = COALESCE(EXCLUDED.source_chunk_id, entity_edges.source_chunk_id)
            RETURNING id
            """,
            source_entity_id,
            target_entity_id,
            relation,
            weight,
            source_chunk_id,
        )
        return row["id"]


async def get_entities_for_chunks(chunk_ids: list[str]) -> dict[str, list[dict]]:
    """Get entities linked to given chunks. Returns {chunk_id: [entity_dict, ...]}."""
    if not chunk_ids:
        return {}
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT e.source_chunk_id::text AS chunk_id,
                   e.name, e.entity_type, e.description,
                   e.properties
            FROM entities e
            WHERE e.source_chunk_id = ANY($1::uuid[])
            ORDER BY e.name
            """,
            chunk_ids,
        )
    result: dict[str, list[dict]] = {}
    for row in rows:
        cid = row["chunk_id"]
        props = json.loads(row["properties"]) if isinstance(row["properties"], str) else dict(row["properties"] or {})
        entry = {
            "name": row["name"],
            "entity_type": row["entity_type"],
            "description": row["description"],
            "properties": props,
        }
        result.setdefault(cid, []).append(entry)
    return result


async def get_entity_edges_for_chunks(chunk_ids: list[str]) -> dict[str, list[dict]]:
    """Get entity edges linked to given chunks. Returns {chunk_id: [edge_dict, ...]}."""
    if not chunk_ids:
        return {}
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT ee.source_chunk_id::text AS chunk_id,
                   src.name AS source_name,
                   tgt.name AS target_name,
                   ee.relation, ee.weight
            FROM entity_edges ee
            JOIN entities src ON src.id = ee.source_entity_id
            JOIN entities tgt ON tgt.id = ee.target_entity_id
            WHERE ee.source_chunk_id = ANY($1::uuid[])
            ORDER BY ee.created_at
            """,
            chunk_ids,
        )
    result: dict[str, list[dict]] = {}
    for row in rows:
        cid = row["chunk_id"]
        entry = {
            "source_name": row["source_name"],
            "target_name": row["target_name"],
            "relation": row["relation"],
            "weight": float(row["weight"]),
        }
        result.setdefault(cid, []).append(entry)
    return result


async def get_entity_by_name(namespace_id: int, name: str, entity_type: str) -> str | None:
    """Look up entity UUID by name+type. Returns None if not found."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id::text FROM entities WHERE namespace_id = $1 AND name = $2 AND entity_type = $3",
            namespace_id,
            name,
            entity_type,
        )
    return row["id"] if row else None


async def create_ltm_job(namespace_id: int, source_path: str) -> str:
    """Create a new LTM ingest job. Returns job UUID."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO ltm_jobs (namespace_id, source_path, status)
            VALUES ($1, $2, 'pending')
            RETURNING id::text
            """,
            namespace_id,
            source_path,
        )
        return row["id"]


async def update_ltm_job(
    job_id: str,
    status: str | None = None,
    current_step: str | None = None,
    total_chunks: int | None = None,
    processed_chunks: int | None = None,
    error: str | None = None,
) -> None:
    """Update LTM job state."""
    pool = await get_pool()
    sets: list[str] = ["updated_at = NOW()"]
    params: list[Any] = []
    idx = 2  # $1 is job_id

    if status is not None:
        sets.append(f"status = ${idx}")
        params.append(status)
        idx += 1
    if current_step is not None:
        sets.append(f"current_step = ${idx}")
        params.append(current_step)
        idx += 1
    if total_chunks is not None:
        sets.append(f"total_chunks = ${idx}")
        params.append(total_chunks)
        idx += 1
    if processed_chunks is not None:
        sets.append(f"processed_chunks = ${idx}")
        params.append(processed_chunks)
        idx += 1
    if error is not None:
        sets.append(f"error = ${idx}")
        params.append(error)
        idx += 1

    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE ltm_jobs SET {', '.join(sets)} WHERE id = $1::uuid",
            job_id,
            *params,
        )


async def get_ltm_job(job_id: str) -> dict[str, Any] | None:
    """Get LTM job by id."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id::text, namespace_id, source_path, status, current_step,
                   total_chunks, processed_chunks, error,
                   created_at::text, updated_at::text
            FROM ltm_jobs WHERE id = $1::uuid
            """,
            job_id,
        )
    return dict(row) if row else None


async def list_entities(namespace_id: int, limit: int = 100) -> list[dict[str, Any]]:
    """List entities in a namespace with edge counts."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT e.id::text, e.name, e.entity_type, e.description,
                   e.properties, e.created_at::text,
                   (SELECT COUNT(*)::int FROM entity_edges ee
                    WHERE ee.source_entity_id = e.id OR ee.target_entity_id = e.id) AS edge_count
            FROM entities e
            WHERE e.namespace_id = $1
            ORDER BY e.name
            LIMIT $2
            """,
            namespace_id,
            limit,
        )
    results = []
    for r in rows:
        props = json.loads(r["properties"]) if isinstance(r["properties"], str) else dict(r["properties"] or {})
        results.append(
            {
                "id": r["id"],
                "name": r["name"],
                "entity_type": r["entity_type"],
                "description": r["description"],
                "properties": props,
                "edge_count": r["edge_count"],
                "created_at": r["created_at"],
            }
        )
    return results


async def get_entity_graph(namespace_id: int, entity_name: str | None = None) -> tuple[list[dict], list[dict]]:
    """Get entity graph as (nodes, edges). If entity_name given, return 1-hop neighborhood."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if entity_name:
            # 1-hop: the entity + its direct neighbors
            rows_edges = await conn.fetch(
                """
                SELECT src.name AS source_name, src.entity_type AS source_type,
                       tgt.name AS target_name, tgt.entity_type AS target_type,
                       ee.relation, ee.weight
                FROM entity_edges ee
                JOIN entities src ON src.id = ee.source_entity_id
                JOIN entities tgt ON tgt.id = ee.target_entity_id
                WHERE src.namespace_id = $1
                  AND (src.name = $2 OR tgt.name = $2)
                ORDER BY ee.created_at
                LIMIT 200
                """,
                namespace_id,
                entity_name,
            )
        else:
            rows_edges = await conn.fetch(
                """
                SELECT src.name AS source_name, src.entity_type AS source_type,
                       tgt.name AS target_name, tgt.entity_type AS target_type,
                       ee.relation, ee.weight
                FROM entity_edges ee
                JOIN entities src ON src.id = ee.source_entity_id
                JOIN entities tgt ON tgt.id = ee.target_entity_id
                WHERE src.namespace_id = $1
                ORDER BY ee.created_at
                LIMIT 500
                """,
                namespace_id,
            )

    # Build unique nodes from edges
    node_set: dict[str, dict] = {}
    edges: list[dict] = []
    for r in rows_edges:
        node_set[r["source_name"]] = {"name": r["source_name"], "type": r["source_type"]}
        node_set[r["target_name"]] = {"name": r["target_name"], "type": r["target_type"]}
        edges.append(
            {
                "source": r["source_name"],
                "target": r["target_name"],
                "relation": r["relation"],
                "weight": float(r["weight"]),
            }
        )

    return list(node_set.values()), edges


async def get_entity_names_for_namespace(namespace_id: int) -> list[str]:
    """Get all unique entity names in a namespace."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT name FROM entities WHERE namespace_id = $1 ORDER BY name",
            namespace_id,
        )
    return [row["name"] for row in rows]


async def _repoint_and_delete_alias(conn: Any, canonical_id: str, alias_id: str) -> None:
    """Repoint edges from alias to canonical, then delete alias entity."""
    # Repoint source edges (skip if would create duplicate)
    await conn.execute(
        """
        UPDATE entity_edges SET source_entity_id = $1::uuid
        WHERE source_entity_id = $2::uuid
        AND NOT EXISTS (
            SELECT 1 FROM entity_edges e2
            WHERE e2.source_entity_id = $1::uuid
              AND e2.target_entity_id = entity_edges.target_entity_id
              AND e2.relation = entity_edges.relation
        )
        """,
        canonical_id,
        alias_id,
    )
    # Repoint target edges
    await conn.execute(
        """
        UPDATE entity_edges SET target_entity_id = $1::uuid
        WHERE target_entity_id = $2::uuid
        AND NOT EXISTS (
            SELECT 1 FROM entity_edges e2
            WHERE e2.target_entity_id = $1::uuid
              AND e2.source_entity_id = entity_edges.source_entity_id
              AND e2.relation = entity_edges.relation
        )
        """,
        canonical_id,
        alias_id,
    )
    # Delete orphaned edges + alias entity
    await conn.execute(
        "DELETE FROM entity_edges WHERE source_entity_id = $1::uuid OR target_entity_id = $1::uuid",
        alias_id,
    )
    await conn.execute("DELETE FROM entities WHERE id = $1::uuid", alias_id)


async def merge_entity_aliases(namespace_id: int, canonical: str, aliases: list[str]) -> int:
    """Merge alias entities into canonical. Returns merged count."""
    if not aliases:
        return 0
    pool = await get_pool()
    merged = 0
    async with pool.acquire() as conn:
        canonical_row = await conn.fetchrow(
            "SELECT id::text FROM entities WHERE namespace_id = $1 AND name = $2 LIMIT 1",
            namespace_id,
            canonical,
        )
        if not canonical_row:
            return 0
        canonical_id = canonical_row["id"]

        for alias in aliases:
            if alias == canonical:
                continue
            alias_rows = await conn.fetch(
                "SELECT id::text FROM entities WHERE namespace_id = $1 AND name = $2",
                namespace_id,
                alias,
            )
            for alias_row in alias_rows:
                await _repoint_and_delete_alias(conn, canonical_id, alias_row["id"])
                merged += 1

    return merged


async def upsert_entity_event(
    namespace_id: int,
    entity_id: str,
    event_type: str,
    when_t: Any | None = None,
    valid_from: Any | None = None,
    valid_to: Any | None = None,
    description: str | None = None,
    properties: dict | None = None,
    source_chunk_id: str | None = None,
) -> str:
    """Upsert a temporal event. ON CONFLICT (namespace,entity,type,when_t) → update.
    Returns event UUID."""
    pool = await get_pool()
    props_json = json.dumps(properties or {})
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO entity_events
                (namespace_id, entity_id, event_type, when_t, valid_from, valid_to,
                 description, properties, source_chunk_id)
            VALUES ($1, $2::uuid, $3, $4, $5, $6, $7, $8::jsonb, $9::uuid)
            ON CONFLICT (namespace_id, entity_id, event_type, when_t)
            DO UPDATE SET
                valid_from = COALESCE(EXCLUDED.valid_from, entity_events.valid_from),
                valid_to = COALESCE(EXCLUDED.valid_to, entity_events.valid_to),
                description = COALESCE(EXCLUDED.description, entity_events.description),
                properties = entity_events.properties || EXCLUDED.properties,
                source_chunk_id = COALESCE(EXCLUDED.source_chunk_id, entity_events.source_chunk_id)
            RETURNING id::text
            """,
            namespace_id,
            entity_id,
            event_type,
            when_t,
            valid_from,
            valid_to,
            description,
            props_json,
            source_chunk_id,
        )
        return row["id"]


async def find_overlapping_events(
    namespace_id: int,
    entity_id: str,
    event_type: str,
    valid_from: Any,
    exclude_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return prior open events for (entity,type) whose interval overlaps the new
    valid_from. Used by auto-invalidate logic."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id::text, valid_from, valid_to, when_t
            FROM entity_events
            WHERE namespace_id = $1
              AND entity_id = $2::uuid
              AND event_type = $3
              AND ($5::uuid IS NULL OR id <> $5::uuid)
              AND valid_from IS NOT NULL
              AND valid_from < $4
              AND (valid_to IS NULL OR valid_to > $4)
            ORDER BY valid_from ASC
            """,
            namespace_id,
            entity_id,
            event_type,
            valid_from,
            exclude_id,
        )
    return [dict(r) for r in rows]


async def supersede_event(event_id: str, valid_to: Any, superseded_by: str) -> None:
    """Close an event's validity period and mark it superseded."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE entity_events
            SET valid_to = $2, superseded_by = $3::uuid
            WHERE id = $1::uuid AND valid_to IS NULL
            """,
            event_id,
            valid_to,
            superseded_by,
        )


async def get_events_for_entity(
    namespace_id: int,
    entity_name: str,
    include_superseded: bool = False,
) -> list[dict[str, Any]]:
    """List events for an entity (by name) within a namespace."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT ee.id::text, ee.event_type,
                   ee.when_t::text AS when_t, ee.valid_from::text AS valid_from,
                   ee.valid_to::text AS valid_to,
                   ee.description, ee.properties,
                   ee.source_chunk_id::text AS source_chunk_id,
                   ee.superseded_by::text AS superseded_by
            FROM entity_events ee
            JOIN entities e ON e.id = ee.entity_id
            WHERE e.namespace_id = $1
              AND e.name = $2
              AND ($3::bool OR ee.superseded_by IS NULL)
            ORDER BY ee.when_t NULLS LAST, ee.valid_from NULLS LAST
            """,
            namespace_id,
            entity_name,
            include_superseded,
        )
    out: list[dict[str, Any]] = []
    for r in rows:
        props = json.loads(r["properties"]) if isinstance(r["properties"], str) else dict(r["properties"] or {})
        out.append(
            {
                "id": r["id"],
                "event_type": r["event_type"],
                "when_t": r["when_t"],
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
                "description": r["description"],
                "properties": props,
                "source_chunk_id": r["source_chunk_id"],
                "superseded_by": r["superseded_by"],
            }
        )
    return out


async def get_chunk_events_summary(chunk_ids: list[str]) -> dict[str, list[dict]]:
    """Return events linked to each chunk_id (used for recall enrichment)."""
    if not chunk_ids:
        return {}
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT ee.source_chunk_id::text AS chunk_id,
                   ee.event_type,
                   ee.when_t,
                   ee.valid_from,
                   ee.valid_to,
                   e.name AS entity_name
            FROM entity_events ee
            JOIN entities e ON e.id = ee.entity_id
            WHERE ee.source_chunk_id = ANY($1::uuid[])
              AND ee.superseded_by IS NULL
            """,
            chunk_ids,
        )
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["chunk_id"], []).append(
            {
                "entity_name": r["entity_name"],
                "event_type": r["event_type"],
                "when_t": r["when_t"],
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
            }
        )
    return out


async def filter_chunks_by_temporal(
    chunk_ids: list[str],
    as_of: Any | None = None,
    time_range: tuple[Any, Any] | None = None,
) -> list[str]:
    """Apply temporal filter — return subset of chunk_ids whose events match.
    Chunks with NO events pass through (treated as timeless / always valid)."""
    if not chunk_ids:
        return []
    if as_of is None and time_range is None:
        return chunk_ids
    pool = await get_pool()
    async with pool.acquire() as conn:
        if as_of is not None:
            rows = await conn.fetch(
                """
                WITH input(chunk_id) AS (SELECT unnest($1::uuid[]))
                SELECT input.chunk_id::text AS cid
                FROM input
                LEFT JOIN entity_events ee ON ee.source_chunk_id = input.chunk_id
                GROUP BY input.chunk_id
                HAVING COUNT(ee.id) = 0
                    OR BOOL_OR(
                        ee.valid_from IS NOT NULL
                        AND ee.valid_from <= $2
                        AND (ee.valid_to IS NULL OR ee.valid_to > $2)
                    )
                """,
                chunk_ids,
                as_of,
            )
        else:
            t1, t2 = time_range
            rows = await conn.fetch(
                """
                WITH input(chunk_id) AS (SELECT unnest($1::uuid[]))
                SELECT input.chunk_id::text AS cid
                FROM input
                LEFT JOIN entity_events ee ON ee.source_chunk_id = input.chunk_id
                GROUP BY input.chunk_id
                HAVING COUNT(ee.id) = 0
                    OR BOOL_OR(
                        ee.valid_from IS NOT NULL
                        AND tstzrange(ee.valid_from, ee.valid_to, '[)')
                            && tstzrange($2::timestamptz, $3::timestamptz, '[)')
                    )
                """,
                chunk_ids,
                t1,
                t2,
            )
    return [r["cid"] for r in rows]


# ---- LTM-0013: Reflect layer --------------------------------------------------


async def create_reflect_run(namespace_id: int, model_used: str) -> str:
    """Open a new reflect_runs row in 'running' state. Returns run UUID."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO reflect_runs (namespace_id, model_used, status)
            VALUES ($1, $2, 'running')
            RETURNING id::text
            """,
            namespace_id,
            model_used,
        )
        return row["id"]


async def finalize_reflect_run(
    run_id: str,
    status: str,
    chunks_scanned: int,
    meta_facts_created: int,
    cost_usd: float,
    req_count: int,
    abort_reason: str | None,
) -> None:
    """Close the run with final counters."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE reflect_runs
            SET status = $2,
                chunks_scanned = $3,
                meta_facts_created = $4,
                cost_usd = $5,
                req_count = $6,
                abort_reason = $7,
                finished_at = NOW()
            WHERE id = $1::uuid
            """,
            run_id,
            status,
            chunks_scanned,
            meta_facts_created,
            cost_usd,
            req_count,
            abort_reason,
        )


async def fetch_chunks_for_reflect(
    namespace_id: int,
    since: Any | None,
    limit: int,
) -> dict[str, list[dict]]:
    """Return chunks grouped by primary linked entity name.

    Skips chunks without any linked entity (silent skip — see Fork A condition).
    Each group: [{chunk_id, content, entity_id}, ...].
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.id::text AS chunk_id,
                   c.content,
                   e.id::text AS entity_id,
                   e.name AS entity_name,
                   c.indexed_at
            FROM chunks c
            JOIN entities e
              ON e.source_chunk_id = c.id
             AND e.namespace_id = c.namespace_id
            WHERE c.namespace_id = $1
              AND ($2::timestamptz IS NULL OR c.indexed_at >= $2)
            ORDER BY c.indexed_at DESC, c.id
            LIMIT $3
            """,
            namespace_id,
            since,
            limit,
        )
    grouped: dict[str, list[dict]] = {}
    seen_chunk_per_entity: dict[str, set[str]] = {}
    for r in rows:
        ename = r["entity_name"]
        cid = r["chunk_id"]
        seen = seen_chunk_per_entity.setdefault(ename, set())
        if cid in seen:
            continue
        seen.add(cid)
        grouped.setdefault(ename, []).append({"chunk_id": cid, "content": r["content"], "entity_id": r["entity_id"]})
    return grouped


def dense_to_float32(value: Any) -> np.ndarray:
    """Convert an ``embedding_dense`` column value to a float32 ndarray.

    The pgvector asyncpg codec returns ``pgvector.Vector``, which is not
    iterable and which ``np.asarray`` wraps as a 0-d object scalar.
    """
    to_numpy = getattr(value, "to_numpy", None)
    if to_numpy is not None:
        return np.asarray(to_numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


async def fetch_chunks_for_reflect_cosine(
    namespace_id: int,
    since: Any | None,
    limit: int,
    threshold: float,
) -> dict[str, list[dict]]:
    """Return chunks clustered by dense-embedding cosine similarity (LTM-0018).

    Skips chunks where ``embedding_dense IS NULL``. Cluster keys are stable
    string tags ``"cluster_<root_index>"``; ``entity_id`` is ``None`` for every
    member (cosine path produces ``MetaFact.entity_ids=[]``).
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.id::text AS chunk_id,
                   c.content,
                   c.embedding_dense
            FROM chunks c
            WHERE c.namespace_id = $1
              AND c.embedding_dense IS NOT NULL
              AND ($2::timestamptz IS NULL OR c.indexed_at >= $2)
            ORDER BY c.id
            LIMIT $3
            """,
            namespace_id,
            since,
            limit,
        )
    if len(rows) < 2:
        return {}
    from scrutator.ltm.grouping import cluster_by_cosine

    vectors = np.asarray([dense_to_float32(r["embedding_dense"]) for r in rows], dtype=np.float32)
    index_groups = cluster_by_cosine(vectors, threshold)
    return {
        f"cluster_{root}": [
            {
                "chunk_id": rows[i]["chunk_id"],
                "content": rows[i]["content"],
                "entity_id": None,
            }
            for i in indices
        ]
        for root, indices in index_groups.items()
    }


async def insert_meta_fact(
    namespace_id: int,
    fact: Any,
    embedding: list[float] | None,
) -> str:
    """Insert a meta_facts row. Returns its UUID."""
    pool = await get_pool()
    vector = np.array(embedding, dtype=np.float32) if embedding else None
    props_json = json.dumps(fact.properties or {})
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO meta_facts (
                namespace_id, fact_type, content, source_chunk_ids, entity_ids,
                depth, model_used, reflect_run_id, embedding_dense, properties
            )
            VALUES ($1, $2, $3, $4::uuid[], $5::uuid[], $6, $7, $8::uuid, $9, $10::jsonb)
            RETURNING id::text
            """,
            namespace_id,
            str(fact.fact_type),
            fact.content,
            list(fact.source_chunk_ids),
            list(fact.entity_ids),
            fact.depth,
            fact.model_used,
            fact.reflect_run_id,
            vector,
            props_json,
        )
        return row["id"]


async def list_meta_facts_by_namespace(
    namespace_id: int,
    fact_type: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """List meta-facts in a namespace, optionally filtered by fact_type."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id::text, fact_type, content,
                   ARRAY(SELECT x::text FROM unnest(source_chunk_ids) AS x) AS source_chunk_ids,
                   ARRAY(SELECT x::text FROM unnest(entity_ids) AS x) AS entity_ids,
                   depth, derived_at::text, model_used,
                   reflect_run_id::text, properties
            FROM meta_facts
            WHERE namespace_id = $1
              AND ($2::text IS NULL OR fact_type = $2)
            ORDER BY derived_at DESC
            LIMIT $3
            """,
            namespace_id,
            fact_type,
            limit,
        )
    out: list[dict[str, Any]] = []
    for r in rows:
        props = json.loads(r["properties"]) if isinstance(r["properties"], str) else dict(r["properties"] or {})
        out.append(
            {
                "id": r["id"],
                "fact_type": r["fact_type"],
                "content": r["content"],
                "source_chunk_ids": list(r["source_chunk_ids"] or []),
                "entity_ids": list(r["entity_ids"] or []),
                "depth": r["depth"],
                "derived_at": r["derived_at"],
                "model_used": r["model_used"],
                "reflect_run_id": r["reflect_run_id"],
                "properties": props,
            }
        )
    return out


async def get_meta_facts_for_chunks(chunk_ids: list[str]) -> dict[str, list[dict]]:
    """Reverse lookup: chunk_id → meta-facts that reference it."""
    if not chunk_ids:
        return {}
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT mf.id::text, mf.fact_type, mf.content, mf.depth,
                   ARRAY(SELECT x::text FROM unnest(mf.source_chunk_ids) AS x) AS source_chunk_ids,
                   chunk_id::text AS for_chunk
            FROM meta_facts mf, unnest(mf.source_chunk_ids) AS chunk_id
            WHERE chunk_id = ANY($1::uuid[])
            """,
            chunk_ids,
        )
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["for_chunk"], []).append(
            {
                "id": r["id"],
                "fact_type": r["fact_type"],
                "content": r["content"],
                "depth": r["depth"],
                "source_chunk_ids": list(r["source_chunk_ids"] or []),
            }
        )
    return out


async def search_meta_facts(
    namespace_id: int,
    query_embedding: list[float],
    limit: int,
) -> list[dict[str, Any]]:
    """HNSW search over meta_facts.embedding_dense — cosine similarity."""
    if not query_embedding:
        return []
    pool = await get_pool()
    vector = np.array(query_embedding, dtype=np.float32)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id::text, fact_type, content,
                   ARRAY(SELECT x::text FROM unnest(source_chunk_ids) AS x) AS source_chunk_ids,
                   ARRAY(SELECT x::text FROM unnest(entity_ids) AS x) AS entity_ids,
                   depth, derived_at::text, model_used,
                   reflect_run_id::text,
                   1 - (embedding_dense <=> $2) AS score
            FROM meta_facts
            WHERE namespace_id = $1
              AND embedding_dense IS NOT NULL
            ORDER BY embedding_dense <=> $2
            LIMIT $3
            """,
            namespace_id,
            vector,
            limit,
        )
    return [
        {
            "id": r["id"],
            "fact_type": r["fact_type"],
            "content": r["content"],
            "source_chunk_ids": list(r["source_chunk_ids"] or []),
            "entity_ids": list(r["entity_ids"] or []),
            "depth": r["depth"],
            "derived_at": r["derived_at"],
            "model_used": r["model_used"],
            "reflect_run_id": r["reflect_run_id"],
            "score": float(r["score"]),
        }
        for r in rows
    ]


async def memory_stats(namespace_ids: frozenset[int] | None = None) -> MemoryStats:
    """Get memory statistics grouped by namespace, actor, type."""
    from scrutator.memory.models import MemoryStats

    pool = await get_pool()
    async with pool.acquire() as conn:
        allowed = sorted(namespace_ids) if namespace_ids is not None else None
        total = await conn.fetchval(
            """SELECT COUNT(*)::int FROM chunks
               WHERE source_type = 'memory'
                 AND ($1::int[] IS NULL OR namespace_id = ANY($1::int[]))""",
            allowed,
        )

        ns_rows = await conn.fetch(
            """
            SELECT n.name, COUNT(*)::int AS cnt
            FROM chunks c JOIN namespaces n ON n.id = c.namespace_id
            WHERE c.source_type = 'memory'
              AND ($1::int[] IS NULL OR c.namespace_id = ANY($1::int[]))
            GROUP BY n.name ORDER BY cnt DESC
            """,
            allowed,
        )

        actor_rows = await conn.fetch(
            """
            SELECT c.metadata->>'actor' AS actor, COUNT(*)::int AS cnt
            FROM chunks c
            WHERE c.source_type = 'memory' AND c.metadata->>'actor' IS NOT NULL
              AND ($1::int[] IS NULL OR c.namespace_id = ANY($1::int[]))
            GROUP BY actor ORDER BY cnt DESC
            """,
            allowed,
        )

        type_rows = await conn.fetch(
            """
            SELECT c.metadata->>'memory_type' AS mtype, COUNT(*)::int AS cnt
            FROM chunks c
            WHERE c.source_type = 'memory' AND c.metadata->>'memory_type' IS NOT NULL
              AND ($1::int[] IS NULL OR c.namespace_id = ANY($1::int[]))
            GROUP BY mtype ORDER BY cnt DESC
            """,
            allowed,
        )

    return MemoryStats(
        total_memories=total or 0,
        by_namespace={r["name"]: r["cnt"] for r in ns_rows},
        by_actor={r["actor"]: r["cnt"] for r in actor_rows},
        by_type={r["mtype"]: r["cnt"] for r in type_rows},
    )
