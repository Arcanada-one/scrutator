"""Database repository — CRUD and search queries using asyncpg."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from scrutator.db.connection import get_pool
from scrutator.db.models import ChunkLookupResult, NamespaceInfo, NamespaceStats, SearchResult

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
                ON CONFLICT (source_path, chunk_index)
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


async def get_chunk_ids_by_source(source_path: str) -> list[str]:
    """Get chunk IDs for a source path, ordered by chunk_index."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id::text AS chunk_id FROM chunks WHERE source_path = $1 ORDER BY chunk_index",
            source_path,
        )
    return [row["chunk_id"] for row in rows]


async def delete_by_source(source_path: str) -> int:
    """Delete all chunks for a given source path. Returns deleted count."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM chunks WHERE source_path = $1", source_path)
        return int(result.split()[-1])


async def hybrid_search(
    query_embedding: list[float],
    query_text: str,
    namespace_id: int | None = None,
    limit: int = 10,
    query_sparse: dict[str, float] | None = None,
) -> list[SearchResult]:
    """Hybrid search: dense cosine + sparse lexical + FTS with RRF ranking.

    When query_sparse is provided, uses 3-way RRF (dense + sparse + FTS).
    Otherwise falls back to 2-way RRF (dense + FTS).
    """
    pool = await get_pool()
    vector = np.array(query_embedding, dtype=np.float32)
    fetch_limit = limit * 3

    if query_sparse:
        # 3-way RRF: dense + sparse + FTS
        sparse_json = json.dumps(query_sparse)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH semantic AS (
                    SELECT c.id, ROW_NUMBER() OVER (
                        ORDER BY c.embedding_dense <=> $1
                    ) AS rank
                    FROM chunks c
                    WHERE ($2::int IS NULL OR c.namespace_id = $2)
                      AND c.embedding_dense IS NOT NULL
                    ORDER BY c.embedding_dense <=> $1
                    LIMIT $3
                ),
                fulltext AS (
                    SELECT c.id, ROW_NUMBER() OVER (
                        ORDER BY ts_rank_cd(c.textsearch_ru, plainto_tsquery('russian', $4))
                               + ts_rank_cd(c.textsearch_en, plainto_tsquery('english', $4)) DESC
                    ) AS rank
                    FROM chunks c
                    WHERE ($2::int IS NULL OR c.namespace_id = $2)
                      AND (c.textsearch_ru @@ plainto_tsquery('russian', $4)
                           OR c.textsearch_en @@ plainto_tsquery('english', $4))
                    LIMIT $3
                ),
                sparse_match AS (
                    SELECT sv.chunk_id AS id, ROW_NUMBER() OVER (
                        ORDER BY (
                            SELECT SUM(
                                COALESCE((sv.token_weights->>key)::real, 0) * value::real
                            )
                            FROM jsonb_each_text($6::jsonb) AS q(key, value)
                        ) DESC
                    ) AS rank
                    FROM sparse_vectors sv
                    JOIN chunks c ON c.id = sv.chunk_id
                    WHERE ($2::int IS NULL OR c.namespace_id = $2)
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
                    ORDER BY rrf_score DESC
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
                ORDER BY r.rrf_score DESC
                """,
                vector,
                namespace_id,
                fetch_limit,
                query_text,
                limit,
                sparse_json,
            )
    else:
        # 2-way RRF: dense + FTS (backward-compatible)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH semantic AS (
                    SELECT c.id, ROW_NUMBER() OVER (
                        ORDER BY c.embedding_dense <=> $1
                    ) AS rank
                    FROM chunks c
                    WHERE ($2::int IS NULL OR c.namespace_id = $2)
                      AND c.embedding_dense IS NOT NULL
                    ORDER BY c.embedding_dense <=> $1
                    LIMIT $3
                ),
                fulltext AS (
                    SELECT c.id, ROW_NUMBER() OVER (
                        ORDER BY ts_rank_cd(c.textsearch_ru, plainto_tsquery('russian', $4))
                               + ts_rank_cd(c.textsearch_en, plainto_tsquery('english', $4)) DESC
                    ) AS rank
                    FROM chunks c
                    WHERE ($2::int IS NULL OR c.namespace_id = $2)
                      AND (c.textsearch_ru @@ plainto_tsquery('russian', $4)
                           OR c.textsearch_en @@ plainto_tsquery('english', $4))
                    LIMIT $3
                ),
                ranked AS (
                    SELECT
                        COALESCE(s.id, f.id) AS chunk_id,
                        COALESCE(1.0 / (60 + s.rank), 0.0)
                            + COALESCE(1.0 / (60 + f.rank), 0.0) AS rrf_score
                    FROM semantic s
                    FULL OUTER JOIN fulltext f ON s.id = f.id
                    ORDER BY rrf_score DESC
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
                ORDER BY r.rrf_score DESC
                """,
                vector,
                namespace_id,
                fetch_limit,
                query_text,
                limit,
            )

    results = []
    for row in rows:
        meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"] or {})
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
            )
        )
    return results


async def insert_edges(edges: list[dict[str, Any]]) -> int:
    """Batch insert graph edges. ON CONFLICT → update weight. Returns count."""
    if not edges:
        return 0
    pool = await get_pool()
    inserted = 0
    async with pool.acquire() as conn:
        for edge in edges:
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


async def get_edges_for_chunk(chunk_id: str) -> list[dict[str, Any]]:
    """Get all edges (inbound + outbound) for a chunk."""
    pool = await get_pool()
    async with pool.acquire() as conn:
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
    return [dict(r) for r in rows]


async def delete_edges_by_creator(created_by: str, namespace_id: int | None = None) -> int:
    """Delete edges created by a specific agent. Optional namespace filter."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if namespace_id is not None:
            result = await conn.execute(
                """
                DELETE FROM graph_edges g
                USING chunks c
                WHERE g.source_chunk_id = c.id
                  AND g.created_by = $1
                  AND c.namespace_id = $2
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


async def get_namespaces() -> list[NamespaceInfo]:
    """List all namespaces with chunk counts."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT n.id, n.name, n.description,
                   COUNT(c.id)::int AS chunk_count
            FROM namespaces n
            LEFT JOIN chunks c ON c.namespace_id = n.id
            GROUP BY n.id, n.name, n.description
            ORDER BY n.name
            """
        )
    return [
        NamespaceInfo(id=r["id"], name=r["name"], description=r["description"], chunk_count=r["chunk_count"])
        for r in rows
    ]


async def get_stats() -> dict[str, Any]:
    """Get index statistics."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        total_chunks = await conn.fetchval("SELECT COUNT(*)::int FROM chunks")
        total_namespaces = await conn.fetchval("SELECT COUNT(*)::int FROM namespaces")
        total_projects = await conn.fetchval("SELECT COUNT(*)::int FROM projects")
        ns_rows = await conn.fetch(
            """
            SELECT n.name,
                   COUNT(c.id)::int AS chunk_count,
                   COUNT(DISTINCT c.project_id)::int AS project_count
            FROM namespaces n
            LEFT JOIN chunks c ON c.namespace_id = n.id
            GROUP BY n.name
            ORDER BY n.name
            """
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


async def search_with_filters(
    query_text: str,
    namespace_id: int | None = None,
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
                    ORDER BY c.embedding_dense <=> $1
                ) AS rank
                FROM chunks c
                WHERE ($2::int IS NULL OR c.namespace_id = $2)
                  AND c.embedding_dense IS NOT NULL
                  {extra_where}
                ORDER BY c.embedding_dense <=> $1
                LIMIT $3
            ),
            fulltext AS (
                SELECT c.id, ROW_NUMBER() OVER (
                    ORDER BY ts_rank_cd(c.textsearch_ru, plainto_tsquery('russian', $4))
                           + ts_rank_cd(c.textsearch_en, plainto_tsquery('english', $4)) DESC
                ) AS rank
                FROM chunks c
                WHERE ($2::int IS NULL OR c.namespace_id = $2)
                  AND (c.textsearch_ru @@ plainto_tsquery('russian', $4)
                       OR c.textsearch_en @@ plainto_tsquery('english', $4))
                  {extra_where}
                LIMIT $3
            ),
            ranked AS (
                SELECT
                    COALESCE(s.id, f.id) AS chunk_id,
                    COALESCE(1.0 / (60 + s.rank), 0.0)
                        + COALESCE(1.0 / (60 + f.rank), 0.0) AS rrf_score
                FROM semantic s
                FULL OUTER JOIN fulltext f ON s.id = f.id
                ORDER BY rrf_score DESC
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
            ORDER BY score DESC
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
            DO UPDATE SET weight = EXCLUDED.weight
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
                properties = entity_events.properties || EXCLUDED.properties
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


async def memory_stats() -> MemoryStats:
    """Get memory statistics grouped by namespace, actor, type."""
    from scrutator.memory.models import MemoryStats

    pool = await get_pool()
    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*)::int FROM chunks WHERE source_type = 'memory'")

        ns_rows = await conn.fetch(
            """
            SELECT n.name, COUNT(*)::int AS cnt
            FROM chunks c JOIN namespaces n ON n.id = c.namespace_id
            WHERE c.source_type = 'memory'
            GROUP BY n.name ORDER BY cnt DESC
            """
        )

        actor_rows = await conn.fetch(
            """
            SELECT c.metadata->>'actor' AS actor, COUNT(*)::int AS cnt
            FROM chunks c
            WHERE c.source_type = 'memory' AND c.metadata->>'actor' IS NOT NULL
            GROUP BY actor ORDER BY cnt DESC
            """
        )

        type_rows = await conn.fetch(
            """
            SELECT c.metadata->>'memory_type' AS mtype, COUNT(*)::int AS cnt
            FROM chunks c
            WHERE c.source_type = 'memory' AND c.metadata->>'memory_type' IS NOT NULL
            GROUP BY mtype ORDER BY cnt DESC
            """
        )

    return MemoryStats(
        total_memories=total or 0,
        by_namespace={r["name"]: r["cnt"] for r in ns_rows},
        by_actor={r["actor"]: r["cnt"] for r in actor_rows},
        by_type={r["mtype"]: r["cnt"] for r in type_rows},
    )
