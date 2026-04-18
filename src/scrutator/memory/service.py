"""Memory service — index, recall, and manage memories."""

from __future__ import annotations

import time
import uuid

from scrutator.db import repository
from scrutator.memory.models import (
    MemoryBulkResponse,
    MemoryIndexResponse,
    MemoryRecallRequest,
    MemoryRecallResponse,
    MemoryRecallResult,
    MemoryRecord,
    MemoryStats,
)
from scrutator.search.embedder import embed_single


def _memory_source_path(namespace: str, project: str | None, actor: str, memory_id: str) -> str:
    """Build canonical source_path for a memory chunk."""
    project_part = project or "_"
    return f"memory://{namespace}/{project_part}/{actor}/{memory_id}"


def _memory_metadata(record: MemoryRecord, memory_id: str) -> dict:
    """Build metadata dict from MemoryRecord."""
    meta: dict = {
        "memory_id": memory_id,
        "actor": record.actor,
        "memory_type": record.memory_type,
        "importance": record.importance,
        "tags": record.tags,
    }
    if record.valid_from:
        meta["valid_from"] = record.valid_from
    if record.valid_until:
        meta["valid_until"] = record.valid_until
    if record.source_ref:
        meta["source_ref"] = record.source_ref
    return meta


async def index_memory(record: MemoryRecord) -> MemoryIndexResponse:
    """Index a single memory as a chunk with source_type='memory'."""
    memory_id = str(uuid.uuid4())
    source_path = _memory_source_path(record.namespace, record.project, record.actor, memory_id)
    metadata = _memory_metadata(record, memory_id)

    embedding = await embed_single(record.content)
    namespace_id = await repository.upsert_namespace(record.namespace)
    project_id = None
    if record.project:
        project_id = await repository.upsert_project(namespace_id, record.project)

    chunk = {
        "source_path": source_path,
        "source_type": "memory",
        "chunk_index": 0,
        "content": record.content,
        "content_hash": "",
        "metadata": metadata,
        "token_count": 0,
    }

    await repository.insert_chunks([chunk], [embedding], namespace_id, project_id)

    return MemoryIndexResponse(memory_id=memory_id, chunk_id=memory_id, namespace=record.namespace)


async def bulk_index(records: list[MemoryRecord]) -> MemoryBulkResponse:
    """Index multiple memories in batch."""
    memory_ids: list[str] = []
    for record in records:
        result = await index_memory(record)
        memory_ids.append(result.memory_id)
    return MemoryBulkResponse(indexed=len(memory_ids), memory_ids=memory_ids)


async def recall(request: MemoryRecallRequest) -> MemoryRecallResponse:
    """Search memories with filters and importance boosting."""
    start = time.monotonic()

    namespace_id = None
    if request.namespace:
        namespace_id = await repository.upsert_namespace(request.namespace)

    results = await repository.search_with_filters(
        query_text=request.query,
        namespace_id=namespace_id,
        source_type="memory",
        actor=request.actor,
        memory_type=request.memory_type,
        include_expired=request.include_expired,
        importance_boost=request.importance_boost,
        limit=request.limit,
    )

    recall_results = []
    for r in results:
        if r["score"] < request.min_score:
            continue
        meta = r.get("metadata", {})
        recall_results.append(
            MemoryRecallResult(
                memory_id=meta.get("memory_id", r["chunk_id"]),
                content=r["content"],
                actor=meta.get("actor", "unknown"),
                memory_type=meta.get("memory_type", "fact"),
                importance=meta.get("importance", 0.5),
                score=r["score"],
                namespace=r["namespace"],
                project=r.get("project"),
                tags=meta.get("tags", []),
                valid_from=meta.get("valid_from"),
                valid_until=meta.get("valid_until"),
                source_ref=meta.get("source_ref"),
                created_at=r.get("created_at"),
            )
        )

    elapsed = (time.monotonic() - start) * 1000
    return MemoryRecallResponse(
        results=recall_results,
        total=len(recall_results),
        query=request.query,
        search_time_ms=round(elapsed, 2),
    )


async def get_memory_stats() -> MemoryStats:
    """Get memory statistics grouped by namespace, actor, type."""
    return await repository.memory_stats()
