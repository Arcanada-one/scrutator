"""Tests for ARAS-0057 Task 2: pre-ranking maturity floor in SQL candidate arms.

RED phase: every test here must fail before the implementation lands.

Reuses the repository's mocked-pool SQL-shape pattern from
``tests/test_repository_tenant_scoping.py`` and
``tests/test_repository_deterministic_ordering.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_pool_mock(mock_conn):
    transaction = AsyncMock()
    transaction.__aenter__ = AsyncMock(return_value=None)
    transaction.__aexit__ = AsyncMock(return_value=False)
    mock_conn.transaction = MagicMock(return_value=transaction)
    mock_conn.fetchval = AsyncMock(return_value="force_custom_plan")
    mock_pool = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire.return_value = ctx
    return mock_pool


# ── SearchRequest.maturity model tests ────────────────────────────────────────


class TestSearchRequestMaturity:
    def test_maturity_defaults_to_none(self):
        from scrutator.db.models import SearchRequest

        req = SearchRequest(query="test")
        assert req.maturity is None

    def test_maturity_rejects_invalid_enum(self):
        from scrutator.db.models import SearchRequest

        with pytest.raises(ValueError):
            SearchRequest(query="test", maturity="unknown")


# ── Maturity floor mapping helper ─────────────────────────────────────────────


class TestMaturityFloorMapping:
    """The ordered floors are exact:
    - draft -> draft, validated, production
    - validated -> validated, production
    - production -> production
    """

    def test_floor_production(self):
        from scrutator.db.repository import _maturity_values_for_floor

        assert _maturity_values_for_floor("production") == ["production"]

    def test_floor_validated(self):
        from scrutator.db.repository import _maturity_values_for_floor

        assert _maturity_values_for_floor("validated") == ["validated", "production"]

    def test_floor_draft(self):
        from scrutator.db.repository import _maturity_values_for_floor

        assert _maturity_values_for_floor("draft") == ["draft", "validated", "production"]

    def test_floor_none_is_error(self):
        from scrutator.db.repository import _maturity_values_for_floor

        with pytest.raises(ValueError):
            _maturity_values_for_floor(None)


# ── SQL-shape: 2-way RRF (dense + FTS) maturity predicate ─────────────────────


class TestHybridSearchMaturityFloor:
    """Maturity floor placed inside every candidate CTE before its LIMIT
    when a floor is supplied; absent when None."""

    @pytest.mark.asyncio
    async def test_2way_no_floor_no_maturity_predicate(self):
        """With no maturity floor, the SQL and params contain no maturity predicate."""
        from scrutator.db.repository import hybrid_search

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            await hybrid_search(
                query_embedding=[0.1] * 1024,
                query_text="q",
                namespace_id=5,
                maturity=None,
            )

        sql = mock_conn.fetch.call_args[0][0]
        params = mock_conn.fetch.call_args[0]
        assert "maturity" not in sql, "no maturity predicate when floor is None"
        # No maturity param values injected into the param list
        for p in params:
            if isinstance(p, list):
                assert p != ["draft", "validated", "production"]

    @pytest.mark.asyncio
    async def test_2way_production_floor_injects_predicate(self):
        """With maturity='production', each candidate CTE gets a maturity predicate
        before its LIMIT, and only 'production' is in the bound array."""
        from scrutator.db.repository import hybrid_search

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            await hybrid_search(
                query_embedding=[0.1] * 1024,
                query_text="q",
                namespace_id=5,
                maturity="production",
            )

        sql = mock_conn.fetch.call_args[0][0]
        assert "maturity" in sql, "maturity predicate must be in SQL"

    @pytest.mark.asyncio
    async def test_2way_draft_floor_binds_three_values(self):
        """With maturity='draft', the bound text[] parameter contains all three
        permitted values: draft, validated, production."""
        from scrutator.db.repository import hybrid_search

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            await hybrid_search(
                query_embedding=[0.1] * 1024,
                query_text="q",
                namespace_id=5,
                maturity="draft",
            )

        params = mock_conn.fetch.call_args[0]
        # The bound array should contain all three values
        found = False
        for p in params:
            if isinstance(p, list) and "draft" in p:
                assert sorted(p) == ["draft", "production", "validated"]
                found = True
                break
        assert found, "bound maturity array not found in params"

    @pytest.mark.asyncio
    async def test_2way_no_is_null_or_escape(self):
        """Never add an IS NULL OR escape hatch in the maturity predicate."""
        from scrutator.db.repository import hybrid_search

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            await hybrid_search(
                query_embedding=[0.1] * 1024,
                query_text="q",
                namespace_id=5,
                maturity="production",
            )

        sql = mock_conn.fetch.call_args[0][0]
        assert "IS NULL OR" not in sql.split("maturity")[-1] if "maturity" in sql else True

    @pytest.mark.asyncio
    async def test_maturity_before_limit_in_cte(self):
        """The maturity predicate must appear BEFORE each candidate CTE's LIMIT."""
        from scrutator.db.repository import hybrid_search

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            await hybrid_search(
                query_embedding=[0.1] * 1024,
                query_text="q",
                namespace_id=5,
                maturity="production",
            )

        sql = mock_conn.fetch.call_args[0][0]
        # In each CTE: maturity predicate should appear before LIMIT
        for cte in ["semantic", "fulltext"]:
            cte_start = sql.find("AS (")
            if cte_start == -1:
                continue
            # Find the CTE by name
            cte_idx = sql.find(f"{cte} AS (")
            if cte_idx == -1:
                continue
            limit_idx = sql.find("LIMIT", cte_idx)
            maturity_idx = sql.find("maturity", cte_idx)
            assert maturity_idx != -1, f"maturity predicate missing from {cte} CTE"
            assert maturity_idx < limit_idx, f"maturity predicate must be before LIMIT in {cte} CTE"


# ── SQL-shape: 3-way RRF (dense + FTS + sparse) maturity predicate ────────────


class TestHybridSearch3WayMaturityFloor:
    """The three-way path also places the maturity predicate in all candidate arms."""

    @pytest.mark.asyncio
    async def test_3way_no_floor_no_maturity_predicate(self):
        """With no maturity floor, the 3-way SQL path contains no maturity predicate."""
        from scrutator.db.repository import hybrid_search

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            await hybrid_search(
                query_embedding=[0.1] * 1024,
                query_text="q",
                namespace_id=5,
                query_sparse={"tok": 1.0},
                maturity=None,
            )

        sql = mock_conn.fetch.call_args[0][0]
        assert "maturity" not in sql, "no maturity predicate when floor is None"

    @pytest.mark.asyncio
    async def test_3way_production_floor_in_all_three_ctes(self):
        """With maturity='production', all three CTEs (semantic, fulltext, sparse_match)
        get the maturity predicate before their LIMIT."""
        from scrutator.db.repository import hybrid_search

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            await hybrid_search(
                query_embedding=[0.1] * 1024,
                query_text="q",
                namespace_id=5,
                query_sparse={"tok": 1.0},
                maturity="production",
            )

        sql = mock_conn.fetch.call_args[0][0]
        assert "maturity" in sql, "maturity predicate must be in SQL"
        # Each of the three CTEs should have the predicate
        for cte in ["semantic", "fulltext", "sparse_match"]:
            cte_idx = sql.find(f"{cte} AS (")
            if cte_idx == -1:
                # sparse_match is odd — its alias is "sparse_match AS" without parens
                cte_idx = sql.find(cte)
            assert cte_idx != -1, f"CTE {cte} not found in SQL"
            limit_idx = sql.find("LIMIT", cte_idx)
            maturity_idx = sql.find("maturity", cte_idx)
            assert maturity_idx != -1, f"maturity predicate missing from {cte} CTE in 3-way path"
            assert maturity_idx < limit_idx, f"maturity predicate must be before LIMIT in {cte} CTE"


# ── SQL-shape: search_with_filters maturity predicate ─────────────────────────


class TestSearchWithFiltersMaturityFloor:
    """The search_with_filters path also places the maturity predicate in both
    candidate arms (semantic, fulltext) before ranking."""

    @pytest.mark.asyncio
    async def test_filtered_no_floor_no_maturity_predicate(self):
        """With no maturity floor, the filtered SQL path contains no maturity predicate."""
        from scrutator.db.repository import search_with_filters

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with (
            patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool),
            patch("scrutator.search.embedder.embed_single", new_callable=AsyncMock, return_value=[0.1] * 1024),
        ):
            await search_with_filters(
                query_text="q",
                namespace_id=5,
                source_type="md",
                maturity=None,
            )

        sql = mock_conn.fetch.call_args[0][0]
        assert "maturity" not in sql, "no maturity predicate when floor is None"

    @pytest.mark.asyncio
    async def test_filtered_production_floor_in_both_ctes(self):
        """With maturity='production', both filtered CTEs get the maturity predicate."""
        from scrutator.db.repository import search_with_filters

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with (
            patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool),
            patch("scrutator.search.embedder.embed_single", new_callable=AsyncMock, return_value=[0.1] * 1024),
        ):
            await search_with_filters(
                query_text="q",
                namespace_id=5,
                source_type="md",
                maturity="production",
            )

        sql = mock_conn.fetch.call_args[0][0]
        for cte in ["semantic", "fulltext"]:
            cte_idx = sql.find(f"{cte} AS (")
            if cte_idx == -1:
                continue
            limit_idx = sql.find("LIMIT", cte_idx)
            maturity_idx = sql.find("maturity", cte_idx)
            assert maturity_idx != -1, f"maturity predicate missing from {cte} CTE in filtered path"
            assert maturity_idx < limit_idx, f"maturity predicate must be before LIMIT in {cte} CTE"


# ── API endpoint: early 422 validation + maturity forwarding ──────────────────


class TestSearchEndpointMaturityValidation:
    """Behavioral endpoint tests: maturity validation happens BEFORE
    namespace resolution, and valid maturity+namespace forwards correctly."""

    @pytest.fixture(autouse=True)
    def _override_auth(self):
        from scrutator.auth.capabilities import NamespaceCapability, require_feeder_capability
        from scrutator.health import app

        app.dependency_overrides[require_feeder_capability] = lambda: NamespaceCapability(
            namespaces=frozenset({"skills", "arcanada"})
        )
        yield
        app.dependency_overrides.pop(require_feeder_capability, None)

    def test_maturity_non_skills_namespace_returns_422(self):
        """Maturity + non-skills namespace → 422 BEFORE any authz/namespace resolution."""
        from fastapi.testclient import TestClient

        from scrutator.health import app

        client = TestClient(app)
        resp = client.post(
            "/v1/search",
            json={"query": "test", "namespace": "arcanada", "maturity": "production"},
        )
        assert resp.status_code == 422
        assert "maturity" in resp.text.lower()

    def test_maturity_without_namespace_returns_422(self):
        """Maturity without an explicit namespace → 422 (omitted namespace ≠ skills)."""
        from fastapi.testclient import TestClient

        from scrutator.health import app

        client = TestClient(app)
        resp = client.post(
            "/v1/search",
            json={"query": "test", "maturity": "production"},
        )
        assert resp.status_code == 422
        assert "maturity" in resp.text.lower()


# ── search() forwarding: all paths forward maturity ───────────────────────────


class TestSearchForwardingMaturity:
    """search() must forward maturity to every backend path."""

    @pytest.mark.asyncio
    async def test_hybrid_two_way_forwards_maturity(self):
        from scrutator.search.searcher import search

        with (
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_search,
            patch("scrutator.search.searcher.settings") as mock_settings,
        ):
            mock_settings.rerank_enabled = False
            mock_embed.return_value = [0.1] * 1024
            mock_search.return_value = []

            await search(query="q", namespace_id=1, maturity="validated")

        _, kwargs = mock_search.call_args
        assert kwargs.get("maturity") == "validated"

    @pytest.mark.asyncio
    async def test_hybrid_three_way_forwards_maturity(self):
        from scrutator.search.searcher import search

        with (
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.searcher.embed_sparse", new_callable=AsyncMock) as mock_sparse,
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_search,
            patch("scrutator.search.searcher.settings") as mock_settings,
        ):
            mock_settings.rerank_enabled = False
            mock_embed.return_value = [0.1] * 1024
            mock_sparse.return_value = [{"tok": 0.5}]
            mock_search.return_value = []

            await search(query="q", namespace_id=1, maturity="draft")

        _, kwargs = mock_search.call_args
        assert kwargs.get("maturity") == "draft"

    @pytest.mark.asyncio
    async def test_rerank_path_forwards_maturity(self):
        from scrutator.search.searcher import search

        with (
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_search,
            patch("scrutator.search.searcher.rerank", new_callable=AsyncMock) as mock_rerank,
            patch("scrutator.search.searcher.settings") as mock_settings,
        ):
            mock_settings.rerank_enabled = True
            mock_settings.rerank_pool_multiplier = 4
            mock_embed.return_value = [0.1] * 1024
            mock_search.return_value = []
            mock_rerank.return_value = []

            await search(query="q", namespace_id=1, maturity="production")

        _, kwargs = mock_search.call_args
        assert kwargs.get("maturity") == "production"
        assert kwargs.get("return_pool") is True

    @pytest.mark.asyncio
    async def test_filtered_path_forwards_maturity(self):
        from scrutator.search.searcher import search

        with (
            patch("scrutator.search.searcher.search_with_filters", new_callable=AsyncMock) as mock_filtered,
        ):
            mock_filtered.return_value = []

            await search(query="q", namespace_id=1, source_type="md", maturity="draft")

        _, kwargs = mock_filtered.call_args
        assert kwargs.get("maturity") == "draft"


# ── Locked SQL predicates: exact placeholder numbers and argument counts ──────


class TestLockedSqlPredicates:
    """The maturity predicate must use exact placeholder numbers, appear in every
    candidate CTE arm, and must never have an IS NULL OR escape hatch."""

    @pytest.mark.asyncio
    async def test_2way_predicate_uses_placeholder_6(self):
        from scrutator.db.repository import hybrid_search

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            await hybrid_search(
                query_embedding=[0.1] * 1024,
                query_text="q",
                namespace_id=5,
                maturity="production",
            )

        sql = mock_conn.fetch.call_args[0][0]
        assert "metadata->>'maturity' = ANY($6::text[])" in sql

    @pytest.mark.asyncio
    async def test_2way_predicate_occurs_twice(self):
        from scrutator.db.repository import hybrid_search

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            await hybrid_search(
                query_embedding=[0.1] * 1024,
                query_text="q",
                namespace_id=5,
                maturity="production",
            )

        sql = mock_conn.fetch.call_args[0][0]
        count = sql.count("metadata->>'maturity' = ANY")
        assert count == 2, f"expected 2 predicate occurrences in 2-way SQL, got {count}"

    @pytest.mark.asyncio
    async def test_3way_predicate_uses_placeholder_7(self):
        from scrutator.db.repository import hybrid_search

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            await hybrid_search(
                query_embedding=[0.1] * 1024,
                query_text="q",
                namespace_id=5,
                query_sparse={"tok": 1.0},
                maturity="production",
            )

        sql = mock_conn.fetch.call_args[0][0]
        assert "metadata->>'maturity' = ANY($7::text[])" in sql

    @pytest.mark.asyncio
    async def test_3way_predicate_occurs_three_times(self):
        from scrutator.db.repository import hybrid_search

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            await hybrid_search(
                query_embedding=[0.1] * 1024,
                query_text="q",
                namespace_id=5,
                query_sparse={"tok": 1.0},
                maturity="production",
            )

        sql = mock_conn.fetch.call_args[0][0]
        count = sql.count("metadata->>'maturity' = ANY")
        assert count == 3, f"expected 3 predicate occurrences in 3-way SQL, got {count}"

    @pytest.mark.asyncio
    async def test_no_is_null_or_escape_in_maturity_predicate(self):
        """Never allow an IS NULL OR escape hatch — targeted regex check."""
        import re

        from scrutator.db.repository import hybrid_search

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            await hybrid_search(
                query_embedding=[0.1] * 1024,
                query_text="q",
                namespace_id=5,
                maturity="production",
            )

        sql = mock_conn.fetch.call_args[0][0]
        # IS NULL OR near a maturity predicate is the escape we reject
        assert not re.search(r"IS\s+NULL\s+OR.*maturity", sql, re.IGNORECASE), (
            "IS NULL OR escape hatch must not appear near maturity predicate"
        )

    @pytest.mark.asyncio
    async def test_exact_argument_count_2way(self):
        """2-way with maturity: exactly 6 positional args (vector, ns, limit, text, final, [maturities])."""
        from scrutator.db.repository import hybrid_search

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            await hybrid_search(
                query_embedding=[0.1] * 1024,
                query_text="q",
                namespace_id=5,
                maturity="production",
            )

        args = mock_conn.fetch.call_args[0]
        assert len(args) == 7, f"2-way with maturity expects 7 args (1 SQL + 6 params), got {len(args)}"
        # Last arg is the maturity list
        assert args[6] == ["production"]

    @pytest.mark.asyncio
    async def test_exact_argument_count_3way(self):
        """3-way with maturity: exactly 7 positional args (vector, ns, limit, text, final, sparse, [maturities])."""
        from scrutator.db.repository import hybrid_search

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            await hybrid_search(
                query_embedding=[0.1] * 1024,
                query_text="q",
                namespace_id=5,
                query_sparse={"tok": 1.0},
                maturity="production",
            )

        args = mock_conn.fetch.call_args[0]
        assert len(args) == 8, f"3-way with maturity expects 8 args (1 SQL + 7 params), got {len(args)}"
        # Last arg is the maturity list
        assert args[7] == ["production"]

    @pytest.mark.asyncio
    async def test_exact_argument_count_2way_no_maturity(self):
        """2-way without maturity: exactly 5 positional args."""
        from scrutator.db.repository import hybrid_search

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            await hybrid_search(
                query_embedding=[0.1] * 1024,
                query_text="q",
                namespace_id=5,
                maturity=None,
            )

        args = mock_conn.fetch.call_args[0]
        assert len(args) == 6, f"2-way without maturity expects 6 args (1 SQL + 5 params), got {len(args)}"


# ── Endpoint: valid skills namespace forwards maturity ───────────────────────


class TestSearchEndpointValidSkillsForwardsMaturity:
    """A valid request with namespace=skills and maturity=production must forward
    maturity through to search() exactly, with the auth/namespace layers mocked."""

    @pytest.mark.asyncio
    async def test_valid_skills_request_forwards_maturity_to_search(self):
        from fastapi.testclient import TestClient

        from scrutator.auth.dependency import require_tenant_context
        from scrutator.auth.models import TenantContext
        from scrutator.db.models import SearchResponse
        from scrutator.health import app

        # Override auth so the endpoint allows the request through
        app.dependency_overrides[require_tenant_context] = lambda: TenantContext(
            principal_id="test-principal",
            principal_type="service",
            allowed_namespace_ids=frozenset({42}),
            allowed_namespace_names=frozenset({"skills", "arcanada"}),
        )

        mock_search = AsyncMock()
        mock_search.return_value = SearchResponse(results=[], total=0, query="test", search_time_ms=1.0)

        mock_resolve = AsyncMock()
        mock_resolve.return_value = 1

        try:
            with (
                patch("scrutator.health.search", mock_search),
                patch("scrutator.health.resolve_namespace_selector", mock_resolve),
            ):
                client = TestClient(app)
                resp = client.post(
                    "/v1/search",
                    json={
                        "query": "summarize",
                        "namespace": "skills",
                        "maturity": "production",
                    },
                )

            assert resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.text[:200]}"
            mock_search.assert_awaited_once()
            _, kwargs = mock_search.call_args
            assert kwargs.get("maturity") == "production", f"maturity not forwarded; kwargs={list(kwargs.keys())}"
        finally:
            app.dependency_overrides.pop(require_tenant_context, None)


# ── Locked filtered-search SQL predicates ─────────────────────────────────────


class TestLockedFilteredSqlPredicates:
    """The search_with_filters path must use exact placeholder offsets when both
    source_type and maturity are supplied, with no IS NULL OR escape near maturity."""

    @pytest.mark.asyncio
    async def test_filtered_source_type_at_6_maturity_at_7(self):
        """source_type=$6, maturity=$7 when both are supplied."""
        from scrutator.db.repository import search_with_filters

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with (
            patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool),
            patch(
                "scrutator.search.embedder.embed_single",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            ),
        ):
            await search_with_filters(
                query_text="q",
                namespace_id=5,
                source_type="md",
                maturity="production",
            )

        sql = mock_conn.fetch.call_args[0][0]
        assert "c.source_type = $6" in sql
        assert "metadata->>'maturity' = ANY($7::text[])" in sql

    @pytest.mark.asyncio
    async def test_filtered_predicate_occurs_exactly_twice(self):
        """Maturity predicate appears in both semantic and fulltext CTEs."""
        from scrutator.db.repository import search_with_filters

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with (
            patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool),
            patch(
                "scrutator.search.embedder.embed_single",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            ),
        ):
            await search_with_filters(
                query_text="q",
                namespace_id=5,
                source_type="md",
                maturity="production",
            )

        sql = mock_conn.fetch.call_args[0][0]
        count = sql.count("metadata->>'maturity' = ANY")
        assert count == 2, f"expected 2 predicate occurrences in filtered SQL, got {count}"

    @pytest.mark.asyncio
    async def test_filtered_exact_argument_count_with_maturity(self):
        """filtered + source_type + maturity: SQL + 7 params = 8 args."""
        from scrutator.db.repository import search_with_filters

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with (
            patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool),
            patch(
                "scrutator.search.embedder.embed_single",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            ),
        ):
            await search_with_filters(
                query_text="q",
                namespace_id=5,
                source_type="md",
                maturity="production",
            )

        args = mock_conn.fetch.call_args[0]
        assert len(args) == 8, f"filtered + src_type + maturity expects 8 args, got {len(args)}"
        assert args[6] == "md", f"arg[6] (source_type) expected 'md', got {args[6]!r}"
        assert args[7] == ["production"], f"arg[7] (maturity) expected ['production'], got {args[7]!r}"

    @pytest.mark.asyncio
    async def test_filtered_exact_argument_count_no_maturity(self):
        """filtered + source_type, no maturity: SQL + 6 params = 7 args."""
        from scrutator.db.repository import search_with_filters

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with (
            patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool),
            patch(
                "scrutator.search.embedder.embed_single",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            ),
        ):
            await search_with_filters(
                query_text="q",
                namespace_id=5,
                source_type="md",
                maturity=None,
            )

        args = mock_conn.fetch.call_args[0]
        assert len(args) == 7, f"filtered + src_type, no maturity expects 7 args, got {len(args)}"
        sql = mock_conn.fetch.call_args[0][0]
        assert "maturity" not in sql, "no maturity predicate when floor is None"

    @pytest.mark.asyncio
    async def test_filtered_no_is_null_or_escape_near_maturity(self):
        """No IS NULL OR escape hatch near maturity in filtered path."""
        import re

        from scrutator.db.repository import search_with_filters

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with (
            patch("scrutator.db.repository.get_pool", new_callable=AsyncMock, return_value=mock_pool),
            patch(
                "scrutator.search.embedder.embed_single",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            ),
        ):
            await search_with_filters(
                query_text="q",
                namespace_id=5,
                source_type="md",
                maturity="production",
            )

        sql = mock_conn.fetch.call_args[0][0]
        # Split after each maturity predicate occurrence; the fragment immediately
        # following must not contain an IS NULL OR escape before the closing AND/LIMIT.
        fragments = sql.split("metadata->>'maturity'")
        for fragment in fragments[1:]:  # skip text before first occurrence
            # Look at the next ~80 chars after the predicate key
            tail = fragment[:80]
            if re.search(r"IS\s+NULL\s+OR", tail, re.IGNORECASE):
                raise AssertionError(
                    f"IS NULL OR escape hatch found near maturity predicate in filtered path: ...{tail}..."
                )


# ── Deterministic two-way forwarding: no sparse → query_sparse=None ───────────


class TestTwoWayForwardingNoSparse:
    """When embed_sparse returns no vectors, query_sparse must be None in the
    hybrid_search call, forcing the 2-way RRF path."""

    @pytest.mark.asyncio
    async def test_two_way_forwards_query_sparse_none_when_no_sparse_vector(self):
        from scrutator.search.searcher import search

        with (
            patch("scrutator.search.searcher.embed_single", new_callable=AsyncMock) as mock_embed,
            patch("scrutator.search.searcher.embed_sparse", new_callable=AsyncMock) as mock_sparse,
            patch("scrutator.search.searcher.hybrid_search", new_callable=AsyncMock) as mock_search,
            patch("scrutator.search.searcher.settings") as mock_settings,
        ):
            mock_settings.rerank_enabled = False
            mock_embed.return_value = [0.1] * 1024
            # No sparse vector — embed_sparse returns empty list
            mock_sparse.return_value = []
            mock_search.return_value = []

            await search(query="q", namespace_id=1, maturity="draft")

        mock_sparse.assert_awaited_once_with(["q"])
        _, kwargs = mock_search.call_args
        assert kwargs.get("query_sparse") is None, "query_sparse must be None when embed_sparse returns no vectors"
        assert kwargs.get("maturity") == "draft"
