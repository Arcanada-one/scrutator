"""Hybrid search engine — embed query, RRF ranking, source attribution."""

from __future__ import annotations

import time

from scrutator.db.models import SearchResponse, SearchResult
from scrutator.db.repository import hybrid_search, search_with_filters, upsert_namespace
from scrutator.search.embedder import embed_single


async def search(
    query: str,
    namespace: str | None = None,
    project: str | None = None,
    source_type: str | None = None,
    limit: int = 10,
    min_score: float = 0.0,
    include_content: bool = True,
) -> SearchResponse:
    """Execute hybrid search: embed query → dense+FTS → RRF → results."""
    start = time.monotonic()

    # Resolve namespace id if specified
    namespace_id = None
    if namespace:
        namespace_id = await upsert_namespace(namespace)

    if source_type:
        # Use filtered search when source_type is specified
        raw = await search_with_filters(
            query_text=query,
            namespace_id=namespace_id,
            source_type=source_type,
            limit=limit,
        )
        results = [
            SearchResult(
                chunk_id=r["chunk_id"],
                content=r["content"],
                source_path=r["source_path"],
                source_type=r["source_type"],
                chunk_index=r["chunk_index"],
                score=r["score"],
                namespace=r["namespace"],
                project=r.get("project"),
                metadata=r.get("metadata", {}),
                heading_hierarchy=r.get("metadata", {}).get("heading_hierarchy", []),
            )
            for r in raw
        ]
    else:
        # Standard hybrid search (backward-compatible)
        query_embedding = await embed_single(query)
        results = await hybrid_search(
            query_embedding=query_embedding,
            query_text=query,
            namespace_id=namespace_id,
            limit=limit,
        )

    # Apply filters
    if min_score > 0:
        results = [r for r in results if r.score >= min_score]

    if not include_content:
        for r in results:
            r.content = ""

    elapsed_ms = (time.monotonic() - start) * 1000

    return SearchResponse(
        results=results,
        total=len(results),
        query=query,
        search_time_ms=round(elapsed_ms, 2),
    )
