"""Tests for SRCH-0029 M2: ColBERT reranker.

Covers:
- MaxSim numeric correctness on small fixture
- embed_colbert parses colbert_vecs API response
- Soft-fail: embed_colbert raises → rerank returns original top_k (RRF order preserved)
- Pool-cap: limit * multiplier > max_pool → uses max_pool cap
- score_kind set to 'colbert_rerank' on reranked results
- relevance_score set to MaxSim on reranked results
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from scrutator.db.models import SearchResult


def _make_result(chunk_id: str, score: float, content: str = "text") -> SearchResult:
    """Factory for test SearchResult objects."""
    return SearchResult(
        chunk_id=chunk_id,
        content=content,
        source_path=f"docs/{chunk_id}.md",
        source_type="md",
        chunk_index=0,
        score=score,
        namespace="arcanada",
    )


class TestMaxSimNumeric:
    """Unit test MaxSim computation on known small vectors."""

    def test_maxsim_simple(self):
        """MaxSim over 2 query tokens vs 2 doc tokens with known values."""
        from scrutator.search.reranker import _maxsim

        # Query: 2 tokens of dim 4 (unit-normed)
        q = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        # Doc: 2 tokens of dim 4 (unit-normed)
        d = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

        # Q[0] · D[0]=1.0, Q[0] · D[1]=0.0  → max=1.0
        # Q[1] · D[0]=0.0, Q[1] · D[1]=0.0  → max=0.0
        # MaxSim = 1.0 + 0.0 = 1.0
        score = _maxsim(q, d)
        assert score == pytest.approx(1.0)

    def test_maxsim_perfect_match(self):
        """Identical Q and D → MaxSim = n_q_tokens (each token similarity = 1.0)."""
        from scrutator.search.reranker import _maxsim

        n = 3
        q = np.eye(n, dtype=np.float32)  # 3 orthogonal unit tokens
        d = np.eye(n, dtype=np.float32)
        score = _maxsim(q, d)
        assert score == pytest.approx(float(n))

    def test_maxsim_normalises_input(self):
        """Non-unit vectors are normalised before MaxSim (internal to _maxsim)."""
        from scrutator.search.reranker import _maxsim

        # 2-dim, scale by 2 — should normalise to unit and still give same result
        q = np.array([[2.0, 0.0]], dtype=np.float32)
        d = np.array([[2.0, 0.0]], dtype=np.float32)
        score = _maxsim(q, d)
        assert score == pytest.approx(1.0)

    def test_maxsim_orthogonal(self):
        """Orthogonal Q and D → MaxSim = 0."""
        from scrutator.search.reranker import _maxsim

        q = np.array([[1.0, 0.0]], dtype=np.float32)
        d = np.array([[0.0, 1.0]], dtype=np.float32)
        score = _maxsim(q, d)
        assert score == pytest.approx(0.0, abs=1e-6)


class TestEmbedColbert:
    """Tests for embedder.embed_colbert."""

    @pytest.mark.asyncio
    async def test_embed_colbert_returns_token_vecs(self):
        from scrutator.search.embedder import embed_colbert

        # Simulate API response: 1 text → 4 tokens × 1024 dim
        colbert_vecs = [[0.1] * 1024 for _ in range(4)]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "object": "list",
            "data": [{"colbert_vecs": colbert_vecs, "index": 0}],
        }
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch("scrutator.search.embedder.get_client", return_value=mock_client):
            result = await embed_colbert(["hello world"])

        assert len(result) == 1  # 1 text
        assert len(result[0]) == 4  # 4 tokens
        assert len(result[0][0]) == 1024  # 1024-dim

    @pytest.mark.asyncio
    async def test_embed_colbert_empty_input(self):
        from scrutator.search.embedder import embed_colbert

        result = await embed_colbert([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_colbert_error_raises_embedding_error(self):
        from scrutator.search.embedder import EmbeddingError, embed_colbert

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with (
            patch("scrutator.search.embedder.get_client", return_value=mock_client),
            pytest.raises(EmbeddingError, match="503"),
        ):
            await embed_colbert(["test"])

    @pytest.mark.asyncio
    async def test_embed_colbert_uses_colbert_endpoint(self):
        """Verify the correct /v1/embeddings/colbert endpoint is called."""
        from scrutator.search.embedder import embed_colbert

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"colbert_vecs": [[0.1] * 1024], "index": 0}]}
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch("scrutator.search.embedder.get_client", return_value=mock_client):
            await embed_colbert(["test"])

        call_url = mock_client.post.call_args[0][0]
        assert call_url.endswith("/v1/embeddings/colbert"), f"Expected colbert endpoint, got: {call_url}"

    @pytest.mark.asyncio
    async def test_embed_colbert_multi_texts(self):
        """Two texts → two colbert_vecs entries."""
        from scrutator.search.embedder import embed_colbert

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"colbert_vecs": [[0.1] * 1024, [0.2] * 1024], "index": 0},
                {"colbert_vecs": [[0.3] * 1024], "index": 1},
            ]
        }
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch("scrutator.search.embedder.get_client", return_value=mock_client):
            result = await embed_colbert(["hello", "world"])

        assert len(result) == 2
        assert len(result[0]) == 2  # 2 tokens
        assert len(result[1]) == 1  # 1 token

    @pytest.mark.asyncio
    async def test_embed_colbert_pages_at_provider_batch_cap(self):
        """The live ColBERT endpoint rejects batches larger than 16."""
        from scrutator.search.embedder import embed_colbert

        mock_client = AsyncMock()

        def response_for_page(*_args, **kwargs):
            page = kwargs["json"]["input"]
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {
                "data": [
                    {"colbert_vecs": [[float(text.removeprefix("text-"))] * 1024], "index": index}
                    for index, text in enumerate(page)
                ]
            }
            return response

        mock_client.post.side_effect = response_for_page

        with patch("scrutator.search.embedder.get_client", return_value=mock_client):
            result = await embed_colbert([f"text-{index}" for index in range(20)])

        assert [call.kwargs["json"]["input"] for call in mock_client.post.await_args_list] == [
            [f"text-{index}" for index in range(16)],
            [f"text-{index}" for index in range(16, 20)],
        ]
        assert [vectors[0][0] for vectors in result] == [float(index) for index in range(20)]


class TestReranker:
    """Tests for search/reranker.rerank."""

    @pytest.mark.asyncio
    async def test_rerank_reorders_by_maxsim(self):
        """High-similarity candidate rises to top after rerank."""
        from scrutator.search.reranker import rerank

        # 3 candidates; candidate B should rank highest after rerank
        candidates = [
            _make_result("A", score=0.05, content="unrelated"),  # RRF rank 1
            _make_result("B", score=0.03, content="exact match"),  # RRF rank 2
            _make_result("C", score=0.02, content="noise"),  # RRF rank 3
        ]

        # Query colbert: 1 token matching "exact" direction
        query_vecs = [[[1.0, 0.0, 0.0, 0.0] * 256]]  # 1 token, 1024-dim (repeated)
        # Candidate B colbert: token very similar to query token
        b_vecs = [[[1.0, 0.0, 0.0, 0.0] * 256]]
        a_vecs = [[[0.0, 1.0, 0.0, 0.0] * 256]]
        c_vecs = [[[0.0, 0.0, 1.0, 0.0] * 256]]

        def fake_embed_colbert(texts):
            # query + candidates are called separately; match on text
            mapping = {
                "exact match": b_vecs[0],
                "unrelated": a_vecs[0],
                "noise": c_vecs[0],
            }
            return [mapping.get(t, [[0.0] * 1024]) for t in texts]

        with (
            patch(
                "scrutator.search.reranker.embed_colbert",
                new=AsyncMock(side_effect=lambda texts: fake_embed_colbert(texts)),
            ),
            patch(
                "scrutator.search.reranker.embed_colbert",
                new=AsyncMock(side_effect=[query_vecs, [a_vecs[0], b_vecs[0], c_vecs[0]]]),
            ),
        ):
            results = await rerank(query="test", candidates=candidates, top_k=3)

        # Candidate B (1.0 · 1.0 dot product = highest maxsim) should be first
        assert results[0].chunk_id == "B"

    @pytest.mark.asyncio
    async def test_rerank_soft_fail_on_embed_error(self):
        """If embed_colbert raises, rerank returns original top_k in RRF order (soft-fail)."""
        from scrutator.search.embedder import EmbeddingError
        from scrutator.search.reranker import rerank

        candidates = [
            _make_result("A", score=0.05),
            _make_result("B", score=0.03),
            _make_result("C", score=0.02),
        ]

        with patch(
            "scrutator.search.reranker.embed_colbert",
            new=AsyncMock(side_effect=EmbeddingError("ColBERT API down")),
        ):
            results = await rerank(query="test", candidates=candidates, top_k=2)

        # On failure: return first top_k in original order
        assert len(results) == 2
        assert results[0].chunk_id == "A"
        assert results[1].chunk_id == "B"

    @pytest.mark.asyncio
    async def test_rerank_soft_fail_populates_rrf_citation(self):
        """On soft-fail, returned results must carry citation with score_kind='rrf' (invariant).

        Guards the SRCH-0029 compliance fix: rerank-ON soft-fail must not return
        citation=None, which would break the M1 frozen Citation contract (ARCA-0180).
        """
        from scrutator.search.embedder import EmbeddingError
        from scrutator.search.reranker import rerank

        candidates = [
            _make_result("A", score=0.05),
            _make_result("B", score=0.03),
        ]
        # candidates have citation=None (as hybrid_search returns them before rerank runs)
        assert all(c.citation is None for c in candidates)

        with patch(
            "scrutator.search.reranker.embed_colbert",
            new=AsyncMock(side_effect=EmbeddingError("ColBERT API down")),
        ):
            results = await rerank(query="test", candidates=candidates, top_k=2)

        # M1 invariant: citation must be present and use rrf score on soft-fail
        for r in results:
            assert r.citation is not None, f"citation must not be None on soft-fail (chunk_id={r.chunk_id})"
            assert r.citation.score_kind == "rrf", (
                f"score_kind must be 'rrf' on soft-fail, got {r.citation.score_kind!r}"
            )
            assert r.citation.relevance_score == r.score, "citation.relevance_score must mirror RRF .score on soft-fail"
            assert r.citation.chunk_id == r.chunk_id

    @pytest.mark.asyncio
    async def test_rerank_soft_fail_unexpected_exception_populates_rrf_citation(self):
        """On unexpected exception in rerank, citation invariant is still upheld."""
        from scrutator.search.reranker import rerank

        candidates = [_make_result("X", score=0.04)]

        with patch(
            "scrutator.search.reranker.embed_colbert",
            new=AsyncMock(side_effect=RuntimeError("unexpected")),
        ):
            results = await rerank(query="test", candidates=candidates, top_k=1)

        assert len(results) == 1
        assert results[0].citation is not None
        assert results[0].citation.score_kind == "rrf"

    @pytest.mark.asyncio
    async def test_rerank_sets_score_kind_colbert_rerank(self):
        """After successful rerank, score_kind='colbert_rerank' on all returned results."""
        from scrutator.search.reranker import rerank

        candidates = [_make_result("A", score=0.05, content="test content")]

        # Single token query, single token doc, orthogonal → maxsim = 0.0 (still valid)
        query_vecs = [[[1.0] + [0.0] * 1023]]  # 1 token, 1024-dim
        doc_vecs = [[[0.0] * 1023 + [1.0]]]  # 1 token, 1024-dim (orthogonal)

        with patch(
            "scrutator.search.reranker.embed_colbert",
            new=AsyncMock(side_effect=[query_vecs, [doc_vecs[0]]]),
        ):
            results = await rerank(query="test", candidates=candidates, top_k=1)

        assert len(results) == 1
        assert results[0].citation is not None
        assert results[0].citation.score_kind == "colbert_rerank"

    @pytest.mark.asyncio
    async def test_rerank_sets_relevance_score_to_maxsim(self):
        """citation.relevance_score is the MaxSim score, not the original RRF score."""
        from scrutator.search.reranker import rerank

        original_rrf = 0.033
        candidates = [_make_result("A", score=original_rrf, content="test content")]

        # Perfect match: 1 token, same direction → MaxSim = 1.0
        token = [1.0] + [0.0] * 1023
        query_vecs = [[token]]
        doc_vecs = [[token]]

        with patch(
            "scrutator.search.reranker.embed_colbert",
            new=AsyncMock(side_effect=[query_vecs, [doc_vecs[0]]]),
        ):
            results = await rerank(query="test", candidates=candidates, top_k=1)

        assert results[0].citation.relevance_score == pytest.approx(1.0, abs=1e-5)
        # The .score field is also updated to maxsim
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.asyncio
    async def test_rerank_pool_cap_applied(self):
        """When candidates > max_pool, only max_pool candidates are sent to ColBERT."""
        from scrutator.search.reranker import rerank

        # 35 candidates; max_pool = 30
        candidates = [_make_result(f"c{i}", score=0.05 - i * 0.001) for i in range(35)]

        # embed_colbert should be called with query (1 text) and at most 30 doc texts
        captured_doc_texts = []

        async def fake_embed(texts: list[str]):
            if len(texts) == 1 and not any(c.content == texts[0] for c in candidates):
                # query embed
                return [[[1.0] + [0.0] * 1023]]
            else:
                # doc embed
                captured_doc_texts.extend(texts)
                return [[[1.0] + [0.0] * 1023] for _ in texts]

        with (
            patch("scrutator.search.reranker.embed_colbert", new=AsyncMock(side_effect=fake_embed)),
            patch("scrutator.search.reranker.settings") as mock_settings,
        ):
            mock_settings.rerank_colbert_max_pool = 30
            results = await rerank(query="a query that won't match content", candidates=candidates, top_k=5)

        # Only top 30 by RRF were sent to ColBERT; result is top_k
        assert len(captured_doc_texts) <= 30
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_rerank_returns_top_k_only(self):
        """rerank returns exactly top_k results even when pool is larger."""
        from scrutator.search.reranker import rerank

        candidates = [_make_result(f"c{i}", score=0.05 - i * 0.001, content=f"text{i}") for i in range(10)]
        top_k = 3

        token = [1.0] + [0.0] * 1023
        # All same vectors → MaxSim all ~1.0, ordering arbitrary but length = top_k
        with patch(
            "scrutator.search.reranker.embed_colbert",
            new=AsyncMock(
                side_effect=[
                    [[token]],  # query: 1 text → 1 token
                    [[token] for _ in range(len(candidates))],  # docs
                ]
            ),
        ):
            results = await rerank(query="test", candidates=candidates, top_k=top_k)

        assert len(results) == top_k
