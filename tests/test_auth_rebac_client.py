"""Tests for SRCH-0023 Step 0/1: rebac_client.py principal -> allowed-namespace resolution.

Fail-closed contract: if both the live OpenFGA check and the local FK-cache fallback miss,
the principal resolves to an empty set (deny), never "all namespaces".
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scrutator.auth.rebac_client import resolve_allowed_namespaces


def _make_pool_mock(mock_conn):
    """Properly configured asyncpg pool mock with async context manager (repo convention)."""
    mock_pool = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire.return_value = ctx
    return mock_pool


class TestFkCacheFallback:
    """OpenFGA not configured (empty URL) -> falls back to principal_namespace_grants."""

    @pytest.mark.asyncio
    async def test_fk_cache_returns_granted_namespaces(self):
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [{"namespace_id": 1}, {"namespace_id": 3}]
        mock_pool = _make_pool_mock(mock_conn)

        with (
            patch("scrutator.auth.rebac_client.settings") as mock_settings,
            patch("scrutator.auth.rebac_client.get_pool", new_callable=AsyncMock, return_value=mock_pool),
        ):
            mock_settings.auth_arcana_openfga_url = ""
            mock_settings.auth_arcana_openfga_store_id = ""
            result = await resolve_allowed_namespaces("svc-1")

        assert result == frozenset({1, 3})

    @pytest.mark.asyncio
    async def test_no_grants_denies_empty_set(self):
        """A principal with no FK-cache rows resolves to empty (deny), never all-namespaces."""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_pool_mock(mock_conn)

        with (
            patch("scrutator.auth.rebac_client.settings") as mock_settings,
            patch("scrutator.auth.rebac_client.get_pool", new_callable=AsyncMock, return_value=mock_pool),
        ):
            mock_settings.auth_arcana_openfga_url = ""
            mock_settings.auth_arcana_openfga_store_id = ""
            result = await resolve_allowed_namespaces("svc-orphan")

        assert result == frozenset()


class TestOpenFgaPrimaryPath:
    @pytest.mark.asyncio
    async def test_live_openfga_success_skips_fk_cache(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"objects": ["namespace:arcanada", "namespace:ltm-bench"]}

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [{"id": 1}, {"id": 2}]
        mock_pool = _make_pool_mock(mock_conn)

        with (
            patch("scrutator.auth.rebac_client.settings") as mock_settings,
            patch("scrutator.auth.rebac_client.httpx.AsyncClient", return_value=mock_client),
            patch("scrutator.auth.rebac_client.get_pool", new_callable=AsyncMock, return_value=mock_pool),
        ):
            mock_settings.auth_arcana_openfga_url = "https://auth.arcanada.ai/openfga"
            mock_settings.auth_arcana_openfga_store_id = "store-1"
            result = await resolve_allowed_namespaces("user-1")

        assert result == frozenset({1, 2})
        # FK-cache fallback query must NOT be consulted when OpenFGA succeeds
        mock_conn.fetch.assert_called_once()
        args = mock_conn.fetch.call_args[0]
        assert "namespaces" in args[0]

    @pytest.mark.asyncio
    async def test_openfga_unreachable_falls_back_to_fk_cache(self):
        import httpx

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("refused")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [{"namespace_id": 5}]
        mock_pool = _make_pool_mock(mock_conn)

        with (
            patch("scrutator.auth.rebac_client.settings") as mock_settings,
            patch("scrutator.auth.rebac_client.httpx.AsyncClient", return_value=mock_client),
            patch("scrutator.auth.rebac_client.get_pool", new_callable=AsyncMock, return_value=mock_pool),
        ):
            mock_settings.auth_arcana_openfga_url = "https://auth.arcanada.ai/openfga"
            mock_settings.auth_arcana_openfga_store_id = "store-1"
            result = await resolve_allowed_namespaces("svc-2")

        assert result == frozenset({5})

    @pytest.mark.asyncio
    async def test_openfga_non_200_falls_back_to_fk_cache(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 503

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [{"namespace_id": 9}]
        mock_pool = _make_pool_mock(mock_conn)

        with (
            patch("scrutator.auth.rebac_client.settings") as mock_settings,
            patch("scrutator.auth.rebac_client.httpx.AsyncClient", return_value=mock_client),
            patch("scrutator.auth.rebac_client.get_pool", new_callable=AsyncMock, return_value=mock_pool),
        ):
            mock_settings.auth_arcana_openfga_url = "https://auth.arcanada.ai/openfga"
            mock_settings.auth_arcana_openfga_store_id = "store-1"
            result = await resolve_allowed_namespaces("svc-3")

        assert result == frozenset({9})
