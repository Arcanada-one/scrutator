"""Hybrid search engine — embed query, RRF ranking, source attribution."""

from __future__ import annotations

import time

from scrutator.db.models import SearchResponse
from scrutator.db.repository import hybrid_search, upsert_namespace
from scrutator.search.embedder import embed_single


async def search(
    query: str,
    namespace: str | None = None,
    project: str | None = None,
    limit: int = 10,
    min_score: float = 0.0,
    include_content: bool = True,
) -> SearchResponse:
    """Execute hybrid search: embed query → dense+FTS → RRF → results."""
    start = time.monotonic()

    # 1. Embed the query
    query_embedding = await embed_single(query)

    # 2. Resolve namespace id if specified
    namespace_id = None
    if namespace:
        namespace_id = await upsert_namespace(namespace)

    # 3. Execute hybrid search
    results = await hybrid_search(
        query_embedding=query_embedding,
        query_text=query,
        namespace_id=namespace_id,
        limit=limit,
    )

    # 4. Apply filters
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
