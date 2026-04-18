"""Dream analyzer — periodic knowledge base analysis and systematization."""

from __future__ import annotations

import time

from scrutator.db import repository
from scrutator.dream.models import (
    BoostScore,
    CrossReference,
    DreamAnalysisRequest,
    DreamAnalysisResult,
    DuplicatePair,
    OrphanChunk,
    StaleChunk,
)


async def _resolve_namespace_id(namespace: str) -> int | None:
    """Look up namespace_id by name. Returns None if not found."""
    namespaces = await repository.get_namespaces()
    for ns in namespaces:
        if ns.name == namespace:
            return ns.id
    return None


async def find_semantic_duplicates(namespace_id: int, threshold: float, limit: int) -> list[DuplicatePair]:
    """Find chunk pairs with cosine similarity above threshold."""
    pairs = await repository.find_similar_pairs(namespace_id, threshold, limit)
    return [
        DuplicatePair(
            chunk_id_a=p["chunk_id_a"],
            chunk_id_b=p["chunk_id_b"],
            similarity=float(p["similarity"]),
            source_path_a=p["source_path_a"],
            source_path_b=p["source_path_b"],
            content_preview_a=p["content_a"][:200],
            content_preview_b=p["content_b"][:200],
        )
        for p in pairs
    ]


async def find_cross_references(
    namespace_id: int, min_similarity: float, dedup_threshold: float, limit: int
) -> list[CrossReference]:
    """Find related but unlinked chunks (similarity in [min, dedup_threshold))."""
    pairs = await repository.find_similar_pairs(namespace_id, min_similarity, limit * 2)
    result = []
    for p in pairs:
        sim = float(p["similarity"])
        if sim >= dedup_threshold:
            continue
        result.append(
            CrossReference(
                chunk_id_a=p["chunk_id_a"],
                chunk_id_b=p["chunk_id_b"],
                similarity=sim,
                source_path_a=p["source_path_a"],
                source_path_b=p["source_path_b"],
            )
        )
        if len(result) >= limit:
            break
    return result


async def find_orphan_chunks(namespace_id: int, limit: int) -> list[OrphanChunk]:
    """Find chunks with zero graph edges."""
    rows = await repository.get_orphan_chunks(namespace_id, limit)
    return [
        OrphanChunk(
            chunk_id=r["chunk_id"],
            source_path=r["source_path"],
            edge_count=r["edge_count"],
            created_at=str(r["created_at"]),
        )
        for r in rows
    ]


async def find_stale_chunks(namespace_id: int, stale_days: int, limit: int) -> list[StaleChunk]:
    """Find chunks not updated in stale_days days."""
    rows = await repository.find_stale_chunks(namespace_id, stale_days, limit)
    return [
        StaleChunk(
            chunk_id=r["chunk_id"],
            source_path=r["source_path"],
            days_since_update=r["days_since_update"],
            edge_count=r["edge_count"],
        )
        for r in rows
    ]


async def compute_boost_scores(namespace_id: int, limit: int) -> list[BoostScore]:
    """Compute relevance boost based on edge connectivity."""
    edge_stats = await repository.get_edge_stats(namespace_id)
    total_edges = edge_stats["total_edges"]
    if total_edges == 0:
        return []

    stats = await repository.get_stats()
    total_chunks = stats["total_chunks"]
    if total_chunks == 0:
        return []

    by_type = edge_stats.get("by_type", [])
    boosts = []
    for entry in by_type[:limit]:
        boosts.append(
            BoostScore(
                chunk_id="aggregate",
                source_path=f"edge_type:{entry['edge_type']}",
                edge_count=entry["count"],
                avg_edge_weight=float(entry.get("avg_weight", 1.0)),
                boost_score=min(1.0, entry["count"] / max(total_edges, 1)),
            )
        )
    return boosts


async def analyze(request: DreamAnalysisRequest) -> DreamAnalysisResult:
    """Run all dream analyzers for a namespace."""
    start = time.monotonic()

    namespace_id = await _resolve_namespace_id(request.namespace)
    if namespace_id is None:
        return DreamAnalysisResult(
            namespace=request.namespace,
            duplicates=[],
            cross_references=[],
            orphans=[],
            stale=[],
            boosts=[],
            stats={"total_chunks": 0, "total_edges": 0, "analysis_time_ms": 0, "error": "namespace_not_found"},
        )

    limit = request.max_results_per_type

    duplicates = await find_semantic_duplicates(namespace_id, request.dedup_threshold, limit)
    cross_refs = await find_cross_references(namespace_id, request.min_similarity, request.dedup_threshold, limit)
    orphans = await find_orphan_chunks(namespace_id, limit)
    stale = await find_stale_chunks(namespace_id, request.stale_days, limit)

    boosts: list[BoostScore] = []
    if request.include_boost:
        boosts = await compute_boost_scores(namespace_id, limit)

    edge_stats = await repository.get_edge_stats(namespace_id)
    stats_data = await repository.get_stats()

    elapsed = int((time.monotonic() - start) * 1000)

    return DreamAnalysisResult(
        namespace=request.namespace,
        duplicates=duplicates,
        cross_references=cross_refs,
        orphans=orphans,
        stale=stale,
        boosts=boosts,
        stats={
            "total_chunks": stats_data["total_chunks"],
            "total_edges": edge_stats["total_edges"],
            "analysis_time_ms": elapsed,
            "duplicates_found": len(duplicates),
            "cross_refs_found": len(cross_refs),
            "orphans_found": len(orphans),
            "stale_found": len(stale),
        },
    )
