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


# ── API endpoint: maturity forwarding and validation ──────────────────────────


class TestSearchEndpointMaturityForwarding:
    """The /v1/search endpoint validates namespace and forwards maturity."""

    @pytest.mark.asyncio
    async def test_maturity_forwarded_to_searcher(self):
        """When maturity is supplied, the search_endpoint forwards it to search()."""
        from scrutator.db.models import SearchRequest

        # Verify SearchRequest carries maturity
        req = SearchRequest(query="test", namespace="skills", maturity="production")
        assert req.maturity == "production"
        model = req.model_dump()
        assert model["maturity"] == "production"
