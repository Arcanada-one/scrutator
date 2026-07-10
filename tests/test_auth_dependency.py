"""Tests for SRCH-0023 Step 1/2: require_tenant_context + resolve_namespace_selector.

Covers V-AC-1, V-AC-2, V-AC-5: fail-closed auth dependency + namespace-selector validation
(in-set / out-of-set / omitted-single-tenant / omitted-multi-tenant / omitted-zero-tenant).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from scrutator.auth.models import TenantContext
from scrutator.auth.verifier import Unauthenticated


def _make_pool_mock(mock_conn):
    mock_pool = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire.return_value = ctx
    return mock_pool


def _request(auth_header: str | None):
    req = MagicMock()
    req.headers = {"authorization": auth_header} if auth_header else {}
    req.method = "POST"
    req.url.path = "/v1/search"
    return req


class TestRequireTenantContextFailClosed:
    @pytest.mark.asyncio
    async def test_missing_auth_denies_when_enforce_true(self):
        from scrutator.auth.dependency import require_tenant_context

        with patch("scrutator.auth.dependency.settings") as mock_settings:
            mock_settings.auth_enforce = True
            with pytest.raises(HTTPException) as exc_info:
                await require_tenant_context(_request(None))
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_token_denies_when_enforce_true(self):
        from scrutator.auth.dependency import require_tenant_context

        with (
            patch("scrutator.auth.dependency.settings") as mock_settings,
            patch(
                "scrutator.auth.dependency.verify_bearer_token",
                new_callable=AsyncMock,
                side_effect=Unauthenticated("bad token"),
            ),
        ):
            mock_settings.auth_enforce = True
            with pytest.raises(HTTPException) as exc_info:
                await require_tenant_context(_request("Bearer bad"))
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_auth_grace_window_returns_empty_context_not_401(self):
        """SCRUTATOR_AUTH_ENFORCE=False (grace window): logs would-deny, does NOT reject,
        but grants an EMPTY allowed-set (never all-namespaces)."""
        from scrutator.auth.dependency import require_tenant_context

        with patch("scrutator.auth.dependency.settings") as mock_settings:
            mock_settings.auth_enforce = False
            ctx = await require_tenant_context(_request(None))

        assert isinstance(ctx, TenantContext)
        assert ctx.allowed_namespace_ids == frozenset()
        assert ctx.allowed_namespace_names == frozenset()

    @pytest.mark.asyncio
    async def test_valid_token_resolves_allowed_namespaces(self):
        from scrutator.auth.dependency import require_tenant_context

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [{"name": "arcanada"}]
        mock_pool = _make_pool_mock(mock_conn)

        with (
            patch("scrutator.auth.dependency.settings") as mock_settings,
            patch(
                "scrutator.auth.dependency.verify_bearer_token",
                new_callable=AsyncMock,
                return_value=("svc-1", "service"),
            ),
            patch(
                "scrutator.auth.dependency.resolve_allowed_namespaces",
                new_callable=AsyncMock,
                return_value=frozenset({1}),
            ),
            patch("scrutator.auth.dependency.get_pool", new_callable=AsyncMock, return_value=mock_pool),
        ):
            mock_settings.auth_enforce = True
            ctx = await require_tenant_context(_request("Bearer arc_api_x"))

        assert ctx.principal_id == "svc-1"
        assert ctx.allowed_namespace_ids == frozenset({1})
        assert ctx.allowed_namespace_names == frozenset({"arcanada"})


class TestResolveNamespaceSelector:
    @pytest.mark.asyncio
    async def test_omitted_single_tenant_resolves_that_tenant(self):
        from scrutator.auth.dependency import resolve_namespace_selector

        ctx = TenantContext(
            principal_id="svc-1",
            principal_type="service",
            allowed_namespace_ids=frozenset({7}),
            allowed_namespace_names=frozenset({"arcanada"}),
        )
        result = await resolve_namespace_selector(ctx, None)
        assert result == 7

    @pytest.mark.asyncio
    async def test_omitted_multi_tenant_is_ambiguous_400(self):
        from scrutator.auth.dependency import resolve_namespace_selector

        ctx = TenantContext(
            principal_id="svc-1",
            principal_type="service",
            allowed_namespace_ids=frozenset({7, 8}),
            allowed_namespace_names=frozenset({"arcanada", "ltm-bench"}),
        )
        with pytest.raises(HTTPException) as exc_info:
            await resolve_namespace_selector(ctx, None)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_omitted_zero_tenant_is_403(self):
        from scrutator.auth.dependency import resolve_namespace_selector

        ctx = TenantContext(
            principal_id="svc-orphan",
            principal_type="service",
            allowed_namespace_ids=frozenset(),
            allowed_namespace_names=frozenset(),
        )
        with pytest.raises(HTTPException) as exc_info:
            await resolve_namespace_selector(ctx, None)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_out_of_set_namespace_is_403_never_widened(self):
        from scrutator.auth.dependency import resolve_namespace_selector

        ctx = TenantContext(
            principal_id="svc-1",
            principal_type="service",
            allowed_namespace_ids=frozenset({7}),
            allowed_namespace_names=frozenset({"arcanada"}),
        )
        with pytest.raises(HTTPException) as exc_info:
            await resolve_namespace_selector(ctx, "someone-elses-namespace")
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_in_set_namespace_resolves_to_its_id(self):
        from scrutator.auth.dependency import resolve_namespace_selector

        ctx = TenantContext(
            principal_id="svc-1",
            principal_type="service",
            allowed_namespace_ids=frozenset({7, 8}),
            allowed_namespace_names=frozenset({"arcanada", "ltm-bench"}),
        )
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 8}
        mock_pool = _make_pool_mock(mock_conn)

        with patch("scrutator.auth.dependency.get_pool", new_callable=AsyncMock, return_value=mock_pool):
            result = await resolve_namespace_selector(ctx, "ltm-bench")

        assert result == 8
