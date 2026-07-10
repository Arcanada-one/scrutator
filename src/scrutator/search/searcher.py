"""Hybrid search engine — embed query, RRF ranking, source attribution."""

from __future__ import annotations

import time
from typing import Literal

from scrutator.config import settings
from scrutator.db.models import Citation, GroupedSearchResult, SearchResponse, SearchResult
from scrutator.db.repository import hybrid_search, search_with_filters
from scrutator.search.embedder import embed_single, embed_sparse
from scrutator.search.reranker import rerank


def _build_citation(r: SearchResult, score_kind: str) -> Citation:
    """Build a Citation from a SearchResult's own fields (no extra DB round-trip)."""
    return Citation(
        chunk_id=r.chunk_id,
        source_path=r.source_path,
        source_type=r.source_type,
        chunk_index=r.chunk_index,
        heading_hierarchy=r.heading_hierarchy,
        relevance_score=r.score,
        score_kind=score_kind,  # type: ignore[arg-type]
    )


def _group_key(r: SearchResult, group_by: Literal["document", "section"]) -> str:
    """SRCH-0021 D-REQ-05: fold key. Falls back to source_path when un-backfilled
    (no `section` key yet) — same degrade-gracefully posture as the nav endpoints."""
    section = r.metadata.get("section") if r.metadata else None
    if group_by == "document":
        return (section or {}).get("doc_id") or r.source_path
    return (section or {}).get("section_key") or r.source_path


def _fold_by_group(results: list[SearchResult], group_by: Literal["document", "section"]) -> list[GroupedSearchResult]:
    """Post-fusion, in-memory fold — never touches the RRF query/order upstream (V-AC-5)."""
    order: list[str] = []
    groups: dict[str, list[SearchResult]] = {}
    for r in results:
        key = _group_key(r, group_by)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)

    grouped: list[GroupedSearchResult] = []
    for key in order:
        members = groups[key]
        representative = max(members, key=lambda m: m.score)
        section = representative.metadata.get("section") if representative.metadata else None
        grouped.append(
            GroupedSearchResult(
                group_key=key,
                doc_id=(section or {}).get("doc_id", ""),
                score=max(m.score for m in members),
                representative=representative,
                member_chunk_ids=[m.chunk_id for m in members],
                member_count=len(members),
            )
        )
    return grouped


async def search(
    query: str,
    namespace_id: int,
    project: str | None = None,
    source_type: str | None = None,
    limit: int = 10,
    min_score: float = 0.0,
    include_content: bool = True,
    group_by: Literal["document", "section"] | None = None,
) -> SearchResponse:
    """Execute hybrid search: embed query → dense+FTS → RRF → optional ColBERT rerank → results.

    M1 (SRCH-0029): every SearchResult carries a populated Citation (always-on, near-zero cost).
    M2 (SRCH-0029): when settings.rerank_enabled=True, widens the fetch pool and reranks via
    ColBERT MaxSim late-interaction. Default OFF — measure-first per consilium condition 2.

    SRCH-0023: namespace_id is mandatory and MUST be resolved by the caller (via
    `auth.dependency.resolve_namespace_selector` against the authenticated principal's
    allowed-namespace set) — this function never auto-provisions or trusts a raw namespace
    string. Read paths never call `upsert_namespace`.

    SRCH-0021 (D-REQ-05/06): group_by is opt-in and post-fusion only — absent (None, the
    default) leaves every prior code path byte-identical (V-AC-6).
    """
    start = time.monotonic()

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
        # M1: populate citation on filtered results (score_kind=rrf — RRF order)
        for r in results:
            r.citation = _build_citation(r, "rrf")
    else:
        # Hybrid search: dense + sparse + FTS
        query_embedding = await embed_single(query)
        try:
            sparse_results = await embed_sparse([query])
            query_sparse = sparse_results[0] if sparse_results else None
        except Exception:
            query_sparse = None  # Fallback to 2-way RRF if sparse fails

        if settings.rerank_enabled:
            # M2: widen pool, return full pool for ColBERT rerank
            results = await hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                namespace_id=namespace_id,
                limit=limit,
                query_sparse=query_sparse,
                fetch_multiplier=settings.rerank_pool_multiplier,
                return_pool=True,
            )
            # Rerank via ColBERT MaxSim (sets .score and .citation on returned results)
            results = await rerank(query=query, candidates=results, top_k=limit)
        else:
            # M2 OFF: byte-identical behaviour (fetch_limit = limit * 3, return top-limit)
            results = await hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                namespace_id=namespace_id,
                limit=limit,
                query_sparse=query_sparse,
            )
            # M1: populate citation on hybrid results (score_kind=rrf)
            for r in results:
                r.citation = _build_citation(r, "rrf")

    # Apply filters
    if min_score > 0:
        results = [r for r in results if r.score >= min_score]

    if not include_content:
        for r in results:
            r.content = ""

    elapsed_ms = (time.monotonic() - start) * 1000

    if group_by:
        grouped = _fold_by_group(results, group_by)
        return SearchResponse(
            results=grouped,
            total=len(grouped),
            query=query,
            search_time_ms=round(elapsed_ms, 2),
        )

    return SearchResponse(
        results=results,
        total=len(results),
        query=query,
        search_time_ms=round(elapsed_ms, 2),
    )
