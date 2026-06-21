"""ColBERT late-interaction reranker for /v1/search (SRCH-0029 M2).

Distinct from ltm/pipeline.py's LLM-based reranker (that operates on
RecallResult objects on the /v1/ltm/recall path via Model Connector).
This module reranks SearchResult objects using ColBERT MaxSim, computed
client-side via numpy over token vectors returned by the Embedding API.

MaxSim formula (per ColBERT paper):
  Q: (n_q_tokens × dim)  — L2-normalised query token matrix
  D: (n_d_tokens × dim)  — L2-normalised doc token matrix
  sim = Q @ D.T           — (n_q × n_d) cosine similarities
  score = sim.max(axis=1).sum()  — max per query token, sum across query
"""

from __future__ import annotations

import logging

import numpy as np

from scrutator.config import settings
from scrutator.db.models import Citation, SearchResult
from scrutator.search.embedder import EmbeddingError, embed_colbert

logger = logging.getLogger(__name__)


def _maxsim(q_vecs: np.ndarray, d_vecs: np.ndarray) -> float:
    """Compute ColBERT MaxSim score between query and document token matrices.

    Args:
        q_vecs: (n_q, dim) float32 array — query token vectors (need not be unit-normed)
        d_vecs: (n_d, dim) float32 array — document token vectors (need not be unit-normed)

    Returns:
        Scalar MaxSim score: sum over query tokens of max cosine similarity to any doc token.
    """
    # L2-normalise (handle zero-vectors gracefully)
    q_norms = np.linalg.norm(q_vecs, axis=1, keepdims=True)
    q_norms = np.where(q_norms == 0, 1.0, q_norms)
    q_normed = q_vecs / q_norms

    d_norms = np.linalg.norm(d_vecs, axis=1, keepdims=True)
    d_norms = np.where(d_norms == 0, 1.0, d_norms)
    d_normed = d_vecs / d_norms

    # (n_q × n_d) cosine similarities
    sim = q_normed @ d_normed.T

    # For each query token: max similarity over doc tokens; sum
    return float(sim.max(axis=1).sum())


async def rerank(query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
    """ColBERT late-interaction rerank of candidates. Returns top_k by MaxSim, descending.

    On embedding failure: logs WARNING and returns candidates[:top_k] unchanged
    (soft-fail, mirrors the sparse-fallback pattern in searcher.py).

    Pool cap: only the top min(len(candidates), settings.rerank_colbert_max_pool)
    candidates (by current RRF score) are sent to ColBERT. Candidates beyond
    the cap are appended in RRF order after the reranked pool.

    Sets result.score and result.citation.relevance_score / score_kind on returned results.
    """
    if not candidates:
        return []

    max_pool = settings.rerank_colbert_max_pool
    if len(candidates) > max_pool:
        # Split: top max_pool go through ColBERT; remainder appended in RRF order
        rerank_pool = candidates[:max_pool]
        remainder = candidates[max_pool:]
    else:
        rerank_pool = candidates
        remainder = []

    try:
        # Embed query (1 call) and all pool docs (1 batched call)
        query_results = await embed_colbert([query])
        query_token_vecs = np.array(query_results[0], dtype=np.float32)  # (n_q, dim)

        doc_texts = [c.content for c in rerank_pool]
        doc_results = await embed_colbert(doc_texts)

        # Score each candidate
        scored: list[tuple[float, SearchResult]] = []
        for candidate, token_vecs_list in zip(rerank_pool, doc_results, strict=True):
            doc_token_vecs = np.array(token_vecs_list, dtype=np.float32)  # (n_d, dim)
            score = _maxsim(query_token_vecs, doc_token_vecs)

            # Build or update Citation
            citation = Citation(
                chunk_id=candidate.chunk_id,
                source_path=candidate.source_path,
                source_type=candidate.source_type,
                chunk_index=candidate.chunk_index,
                heading_hierarchy=candidate.heading_hierarchy,
                relevance_score=score,
                score_kind="colbert_rerank",
            )
            # Mutate a copy to avoid aliasing
            updated = candidate.model_copy(update={"score": score, "citation": citation})
            scored.append((score, updated))

        # Sort descending by MaxSim
        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [r for _, r in scored]

        # Append remainder (in original RRF order, citation stays None or existing)
        combined = reranked + remainder
        return combined[:top_k]

    except EmbeddingError as exc:
        logger.warning("ColBERT rerank failed (soft-fail, returning RRF order): %s", exc)
        return candidates[:top_k]
    except Exception as exc:
        logger.warning("ColBERT rerank unexpected error (soft-fail, returning RRF order): %s", exc)
        return candidates[:top_k]
