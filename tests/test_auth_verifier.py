"""Tests for SRCH-0023 Step 1: bearer credential verification (src/scrutator/auth/verifier.py).

Covers V-AC-5 / V-AC-10 unit cases: valid/expired/bad-signature/JWKS-down token handling.
Fail-closed contract: every failure mode raises Unauthenticated, never returns a principal.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest

from scrutator.auth.verifier import Unauthenticated, verify_bearer_token


class TestMissingOrMalformedHeader:
    @pytest.mark.asyncio
    async def test_missing_header_denies(self):
        with pytest.raises(Unauthenticated):
            await verify_bearer_token(None)

    @pytest.mark.asyncio
    async def test_empty_header_denies(self):
        with pytest.raises(Unauthenticated):
            await verify_bearer_token("")

    @pytest.mark.asyncio
    async def test_wrong_scheme_denies(self):
        with pytest.raises(Unauthenticated):
            await verify_bearer_token("Basic abc123")

    @pytest.mark.asyncio
    async def test_bearer_with_no_token_denies(self):
        with pytest.raises(Unauthenticated):
            await verify_bearer_token("Bearer ")


class TestServiceTokenIntrospection:
    @pytest.mark.asyncio
    async def test_valid_service_token_returns_principal(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"active": True, "principal_id": "svc-42"}

        with patch("scrutator.auth.verifier.settings") as mock_settings:
            mock_settings.auth_arcana_introspect_url = "https://auth.arcanada.ai/introspect"
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            with patch("scrutator.auth.verifier.httpx.AsyncClient", return_value=mock_client):
                principal_id, principal_type = await verify_bearer_token("Bearer arc_api_abcdef")

        assert principal_id == "svc-42"
        assert principal_type == "service"

    @pytest.mark.asyncio
    async def test_inactive_token_denies(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"active": False}

        with patch("scrutator.auth.verifier.settings") as mock_settings:
            mock_settings.auth_arcana_introspect_url = "https://auth.arcanada.ai/introspect"
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            with (
                patch("scrutator.auth.verifier.httpx.AsyncClient", return_value=mock_client),
                pytest.raises(Unauthenticated),
            ):
                await verify_bearer_token("Bearer arc_api_revoked")

    @pytest.mark.asyncio
    async def test_introspection_unreachable_denies_fail_closed(self):
        import httpx

        with patch("scrutator.auth.verifier.settings") as mock_settings:
            mock_settings.auth_arcana_introspect_url = "https://auth.arcanada.ai/introspect"
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("connection refused")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            with (
                patch("scrutator.auth.verifier.httpx.AsyncClient", return_value=mock_client),
                pytest.raises(Unauthenticated),
            ):
                await verify_bearer_token("Bearer arc_api_whatever")

    @pytest.mark.asyncio
    async def test_introspection_non_200_denies(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("scrutator.auth.verifier.settings") as mock_settings:
            mock_settings.auth_arcana_introspect_url = "https://auth.arcanada.ai/introspect"
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            with (
                patch("scrutator.auth.verifier.httpx.AsyncClient", return_value=mock_client),
                pytest.raises(Unauthenticated),
            ):
                await verify_bearer_token("Bearer arc_api_whatever")

    @pytest.mark.asyncio
    async def test_introspection_url_not_configured_denies(self):
        with patch("scrutator.auth.verifier.settings") as mock_settings:
            mock_settings.auth_arcana_introspect_url = ""
            with pytest.raises(Unauthenticated):
                await verify_bearer_token("Bearer arc_api_whatever")


class TestOidcJwksVerification:
    @pytest.mark.asyncio
    async def test_valid_oidc_token_returns_principal(self):
        signing_key = MagicMock()
        signing_key.key = "fake-public-key"

        with patch("scrutator.auth.verifier.settings") as mock_settings:
            mock_settings.auth_arcana_jwks_url = "https://auth.arcanada.ai/.well-known/jwks.json"
            mock_jwks_client = MagicMock()
            mock_jwks_client.get_signing_key_from_jwt.return_value = signing_key
            with (
                patch("scrutator.auth.verifier._get_jwks_client", return_value=mock_jwks_client),
                patch("scrutator.auth.verifier.jwt.decode", return_value={"sub": "user-7", "exp": 9999999999}),
            ):
                principal_id, principal_type = await verify_bearer_token("Bearer eyJhbGciOiJSUzI1NiJ9.fake.token")

        assert principal_id == "user-7"
        assert principal_type == "user"

    @pytest.mark.asyncio
    async def test_expired_token_denies(self):
        signing_key = MagicMock()
        signing_key.key = "fake-public-key"

        with patch("scrutator.auth.verifier.settings") as mock_settings:
            mock_settings.auth_arcana_jwks_url = "https://auth.arcanada.ai/.well-known/jwks.json"
            mock_jwks_client = MagicMock()
            mock_jwks_client.get_signing_key_from_jwt.return_value = signing_key
            with (
                patch("scrutator.auth.verifier._get_jwks_client", return_value=mock_jwks_client),
                patch("scrutator.auth.verifier.jwt.decode", side_effect=jwt.ExpiredSignatureError("expired")),
                pytest.raises(Unauthenticated),
            ):
                await verify_bearer_token("Bearer eyJhbGciOiJSUzI1NiJ9.expired.token")

    @pytest.mark.asyncio
    async def test_bad_signature_denies(self):
        signing_key = MagicMock()
        signing_key.key = "fake-public-key"

        with patch("scrutator.auth.verifier.settings") as mock_settings:
            mock_settings.auth_arcana_jwks_url = "https://auth.arcanada.ai/.well-known/jwks.json"
            mock_jwks_client = MagicMock()
            mock_jwks_client.get_signing_key_from_jwt.return_value = signing_key
            with (
                patch("scrutator.auth.verifier._get_jwks_client", return_value=mock_jwks_client),
                patch("scrutator.auth.verifier.jwt.decode", side_effect=jwt.InvalidSignatureError("bad sig")),
                pytest.raises(Unauthenticated),
            ):
                await verify_bearer_token("Bearer eyJhbGciOiJSUzI1NiJ9.tampered.token")

    @pytest.mark.asyncio
    async def test_jwks_unreachable_denies_fail_closed(self):
        """JWKS endpoint down MUST deny, never fail-open."""
        with patch("scrutator.auth.verifier.settings") as mock_settings:
            mock_settings.auth_arcana_jwks_url = "https://auth.arcanada.ai/.well-known/jwks.json"
            with (
                patch(
                    "scrutator.auth.verifier._get_jwks_client",
                    side_effect=ConnectionError("JWKS host unreachable"),
                ),
                pytest.raises(Unauthenticated),
            ):
                await verify_bearer_token("Bearer eyJhbGciOiJSUzI1NiJ9.whatever.token")

    @pytest.mark.asyncio
    async def test_jwks_url_not_configured_denies(self):
        with patch("scrutator.auth.verifier.settings") as mock_settings:
            mock_settings.auth_arcana_jwks_url = ""
            with pytest.raises(Unauthenticated):
                await verify_bearer_token("Bearer eyJhbGciOiJSUzI1NiJ9.whatever.token")

    @pytest.mark.asyncio
    async def test_missing_sub_claim_denies(self):
        signing_key = MagicMock()
        signing_key.key = "fake-public-key"

        with patch("scrutator.auth.verifier.settings") as mock_settings:
            mock_settings.auth_arcana_jwks_url = "https://auth.arcanada.ai/.well-known/jwks.json"
            mock_jwks_client = MagicMock()
            mock_jwks_client.get_signing_key_from_jwt.return_value = signing_key
            with (
                patch("scrutator.auth.verifier._get_jwks_client", return_value=mock_jwks_client),
                patch("scrutator.auth.verifier.jwt.decode", return_value={"exp": 9999999999}),
                pytest.raises(Unauthenticated),
            ):
                await verify_bearer_token("Bearer eyJhbGciOiJSUzI1NiJ9.no-sub.token")
