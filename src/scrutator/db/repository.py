"""Database repository — CRUD and search queries using asyncpg."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from scrutator.db.connection import get_pool
from scrutator.db.models import NamespaceInfo, NamespaceStats, SearchResult


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
) -> list[SearchResult]:
    """Hybrid search: dense cosine + FTS with RRF ranking."""
    pool = await get_pool()
    vector = np.array(query_embedding, dtype=np.float32)
    fetch_limit = limit * 3

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
