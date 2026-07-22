"""Tests for SRCH-0023 Step 1: bearer credential verification (src/scrutator/auth/verifier.py).

Covers V-AC-5 / V-AC-10 unit cases: valid/expired/bad-signature/JWKS-down token handling.
Fail-closed contract: every failure mode raises Unauthenticated, never returns a principal.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
from pydantic import ValidationError

import scrutator.auth.verifier as verifier_module
from scrutator.auth.verifier import Unauthenticated, verify_bearer_token, verify_oidc_token
from scrutator.config import Settings

LTM_M2M_ISSUER = "https://auth.arcanada.ai"
LTM_M2M_AUDIENCE = "urn:arcanada:scrutator:ltm"
LTM_M2M_SCOPE = "kb:ltm.read"
LTM_M2M_CLIENT_ID = "muneral-kb-sync"
LTM_M2M_OBSERVER_CLIENT_ID = "kb-observer"
LTM_M2M_AGENT_CLIENT_ID = "arcana-agent-kb-reader"


def _ltm_claims(**overrides):
    now = int(time.time())
    claims = {
        "iss": LTM_M2M_ISSUER,
        "aud": LTM_M2M_AUDIENCE,
        "sub": LTM_M2M_CLIENT_ID,
        "client_id": LTM_M2M_CLIENT_ID,
        "scope": LTM_M2M_SCOPE,
        "iat": now,
        "nbf": now,
        "exp": now + 300,
    }
    claims.update(overrides)
    return claims


async def _verify_ltm_claims(claims, *, algorithm="EdDSA"):
    if algorithm == "EdDSA":
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
    else:
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
    token = jwt.encode(claims, private_key, algorithm=algorithm)
    signing_key = MagicMock()
    signing_key.key = public_key
    mock_jwks_client = MagicMock()
    mock_jwks_client.get_signing_key_from_jwt.return_value = signing_key
    with patch("scrutator.auth.verifier._get_jwks_client", return_value=mock_jwks_client):
        return await verify_bearer_token(f"Bearer {token}")


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
        mock_resp.json.return_value = {
            "active": True,
            "principal_id": "svc-42",
            "audience": "urn:test:scrutator",
            "scope": "kb:read",
        }

        with patch("scrutator.auth.verifier.settings") as mock_settings:
            mock_settings.auth_arcana_introspect_url = "https://auth.arcanada.ai/introspect"
            mock_settings.auth_service_audience = "urn:test:scrutator"
            mock_settings.auth_service_scope = "kb:read"
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
    async def test_non_boolean_active_value_denies(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "active": "false",
            "principal_id": "svc-42",
            "audience": "urn:test:scrutator",
            "scope": "kb:read",
        }
        with patch("scrutator.auth.verifier.settings") as mock_settings:
            mock_settings.auth_arcana_introspect_url = "https://auth.arcanada.ai/introspect"
            mock_settings.auth_service_audience = "urn:test:scrutator"
            mock_settings.auth_service_scope = "kb:read"
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            with (
                patch("scrutator.auth.verifier.httpx.AsyncClient", return_value=mock_client),
                pytest.raises(Unauthenticated, match="inactive"),
            ):
                await verify_bearer_token("Bearer arc_api_malformed")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response",
        [
            {"active": True, "principal_id": "svc-42", "audience": "other", "scope": "kb:read"},
            {
                "active": True,
                "principal_id": "svc-42",
                "audience": "urn:test:scrutator",
                "scope": "admin",
            },
        ],
    )
    async def test_wrong_service_resource_profile_denies(self, response):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = response
        with patch("scrutator.auth.verifier.settings") as mock_settings:
            mock_settings.auth_arcana_introspect_url = "https://auth.arcanada.ai/introspect"
            mock_settings.auth_service_audience = "urn:test:scrutator"
            mock_settings.auth_service_scope = "kb:read"
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            with (
                patch("scrutator.auth.verifier.httpx.AsyncClient", return_value=mock_client),
                pytest.raises(Unauthenticated),
            ):
                await verify_bearer_token("Bearer arc_api_wrong-resource")

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
    def test_jwks_fetch_identifies_scrutator_to_edge_policy(self):
        with patch("scrutator.auth.verifier.PyJWKClient") as client_type:
            verifier_module._jwks_client = None
            verifier_module._jwks_client_url = None
            verifier_module._get_jwks_client("https://auth.arcanada.ai/.well-known/jwks.json")

        client_type.assert_called_once_with(
            "https://auth.arcanada.ai/.well-known/jwks.json",
            cache_keys=True,
            lifespan=300,
            headers={"User-Agent": "Arcanada-Scrutator-JWKS/1.0"},
        )

    @pytest.mark.asyncio
    async def test_valid_oidc_token_returns_principal(self):
        signing_key = MagicMock()
        signing_key.key = "fake-public-key"

        with patch("scrutator.auth.verifier.settings") as mock_settings:
            mock_settings.auth_arcana_jwks_url = "https://auth.arcanada.ai/.well-known/jwks.json"
            mock_settings.auth_oidc_issuer = "https://auth.arcanada.ai"
            mock_settings.auth_oidc_audience = "urn:test:scrutator"
            mock_settings.auth_oidc_scope = "kb:read"
            mock_jwks_client = MagicMock()
            mock_jwks_client.get_signing_key_from_jwt.return_value = signing_key
            with (
                patch("scrutator.auth.verifier._get_jwks_client", return_value=mock_jwks_client),
                patch("scrutator.auth.verifier.jwt.get_unverified_header", return_value={"alg": "RS256"}),
                patch(
                    "scrutator.auth.verifier.jwt.decode",
                    return_value={
                        "sub": "user-7",
                        "exp": 9999999999,
                        "iss": "https://auth.arcanada.ai",
                        "aud": "urn:test:scrutator",
                        "scope": "openid kb:read",
                    },
                ),
            ):
                principal_id, principal_type = await verify_bearer_token("Bearer eyJhbGciOiJSUzI1NiJ9.fake.token")

        assert principal_id == "user-7"
        assert principal_type == "user"

    @pytest.mark.asyncio
    async def test_oidc_profile_without_scope_denies_before_jwks_lookup(self):
        with patch("scrutator.auth.verifier.settings") as mock_settings:
            mock_settings.auth_arcana_jwks_url = "https://auth.arcanada.ai/.well-known/jwks.json"
            mock_settings.auth_oidc_issuer = "https://auth.arcanada.ai"
            mock_settings.auth_oidc_audience = "urn:test:scrutator"
            mock_settings.auth_oidc_scope = ""
            with (
                patch("scrutator.auth.verifier._get_jwks_client") as jwks,
                pytest.raises(Unauthenticated, match="resource profile not configured"),
            ):
                await verify_oidc_token("token")
        jwks.assert_not_called()

    @pytest.mark.asyncio
    async def test_oidc_token_without_required_scope_denies(self):
        signing_key = MagicMock()
        signing_key.key = "fake-public-key"
        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = signing_key
        with patch("scrutator.auth.verifier.settings") as mock_settings:
            mock_settings.auth_arcana_jwks_url = "https://auth.arcanada.ai/.well-known/jwks.json"
            mock_settings.auth_oidc_issuer = "https://auth.arcanada.ai"
            mock_settings.auth_oidc_audience = "urn:test:scrutator"
            mock_settings.auth_oidc_scope = "kb:read"
            with (
                patch("scrutator.auth.verifier._get_jwks_client", return_value=mock_jwks_client),
                patch(
                    "scrutator.auth.verifier.jwt.decode",
                    return_value={"sub": "user-7", "scope": "openid profile"},
                ),
                pytest.raises(Unauthenticated, match="scope mismatch"),
            ):
                await verify_oidc_token("token")

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


class TestLtmM2mJwksVerification:
    @pytest.mark.asyncio
    async def test_valid_exact_profile_returns_service_principal(self):
        assert await _verify_ltm_claims(_ltm_claims()) == (LTM_M2M_CLIENT_ID, "service")

    @pytest.mark.asyncio
    async def test_valid_observer_profile_returns_separate_service_principal(self):
        claims = _ltm_claims(sub=LTM_M2M_OBSERVER_CLIENT_ID, client_id=LTM_M2M_OBSERVER_CLIENT_ID)
        assert await _verify_ltm_claims(claims) == (LTM_M2M_OBSERVER_CLIENT_ID, "service")

    @pytest.mark.asyncio
    async def test_valid_agent_reader_profile_returns_separate_service_principal(self):
        claims = _ltm_claims(sub=LTM_M2M_AGENT_CLIENT_ID, client_id=LTM_M2M_AGENT_CLIENT_ID)
        assert await _verify_ltm_claims(claims) == (LTM_M2M_AGENT_CLIENT_ID, "service")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("overrides", "error"),
        [
            ({"client_id": "other-agent"}, "client binding mismatch"),
            ({"aud": "urn:arcanada:scrutator:admin"}, "verification failed"),
            ({"scope": "kb:ltm.write"}, "scope mismatch"),
            ({"exp_offset": 301}, "lifetime mismatch"),
            ({"sub": "other-agent"}, "client binding mismatch"),
            ({"sub": "*", "client_id": "*"}, "client binding mismatch"),
        ],
    )
    async def test_agent_reader_wrong_client_resource_scope_lifetime_or_subject_denies(self, overrides, error):
        claims = _ltm_claims(sub=LTM_M2M_AGENT_CLIENT_ID, client_id=LTM_M2M_AGENT_CLIENT_ID)
        if "exp_offset" in overrides:
            claims["exp"] = claims["iat"] + overrides["exp_offset"]
        else:
            claims.update(overrides)
        with pytest.raises(Unauthenticated, match=error):
            await _verify_ltm_claims(claims)

    @pytest.mark.asyncio
    async def test_agent_reader_client_claim_list_is_rejected_fail_closed(self):
        client_ids = [LTM_M2M_AGENT_CLIENT_ID, "other-agent"]
        claims = _ltm_claims(
            sub=client_ids,
            client_id=client_ids,
        )
        with pytest.raises(Unauthenticated):
            await _verify_ltm_claims(claims)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("claim", "value"),
        [
            ("iss", "https://auth.arcanada.one"),
            ("aud", "https://auth.arcanada.ai/api/v1/admin"),
            ("aud", "https://support.arcanada.one/api/v1/admin"),
            ("scope", "namespace:read"),
            ("scope", f"{LTM_M2M_SCOPE} namespace:read"),
            ("client_id", "other-service"),
            ("sub", "other-service"),
        ],
    )
    async def test_wrong_identity_resource_or_scope_denies(self, claim, value):
        with pytest.raises(Unauthenticated):
            await _verify_ltm_claims(_ltm_claims(**{claim: value}))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("claim", ["exp", "iat", "nbf", "iss", "aud", "sub", "client_id", "scope"])
    async def test_every_required_claim_is_mandatory(self, claim):
        claims = _ltm_claims()
        del claims[claim]
        with pytest.raises(Unauthenticated):
            await _verify_ltm_claims(claims)

    @pytest.mark.asyncio
    async def test_sub_must_equal_client_id_even_for_noncanonical_values(self):
        with pytest.raises(Unauthenticated):
            await _verify_ltm_claims(_ltm_claims(sub="service-a", client_id="service-b"))

    @pytest.mark.asyncio
    async def test_audience_array_is_rejected_in_exact_profile(self):
        with pytest.raises(Unauthenticated):
            await _verify_ltm_claims(_ltm_claims(aud=[LTM_M2M_AUDIENCE]))

    @pytest.mark.asyncio
    async def test_token_lifetime_longer_than_five_minutes_denies(self):
        claims = _ltm_claims()
        claims["exp"] = claims["iat"] + 301
        with pytest.raises(Unauthenticated):
            await _verify_ltm_claims(claims)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("claim", ["iat", "nbf", "exp"])
    async def test_boolean_temporal_claim_denies(self, claim):
        with pytest.raises(Unauthenticated):
            await _verify_ltm_claims(_ltm_claims(**{claim: True}))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", ["not-a-number", 1.5, None])
    async def test_non_integer_nbf_denies(self, value):
        with pytest.raises(Unauthenticated):
            await _verify_ltm_claims(_ltm_claims(nbf=value))

    @pytest.mark.asyncio
    async def test_nbf_must_equal_iat(self):
        claims = _ltm_claims()
        claims["nbf"] = claims["iat"] - 1
        with pytest.raises(Unauthenticated):
            await _verify_ltm_claims(claims)

    @pytest.mark.asyncio
    async def test_rs256_algorithm_confusion_denies_without_legacy_fallback(self):
        with pytest.raises(Unauthenticated):
            await _verify_ltm_claims(_ltm_claims(), algorithm="RS256")

    @pytest.mark.asyncio
    async def test_malformed_agent_reader_profile_never_falls_back_to_interactive(self):
        claims = _ltm_claims(
            aud="urn:arcanada:other-service",
            sub=LTM_M2M_AGENT_CLIENT_ID,
            client_id=LTM_M2M_AGENT_CLIENT_ID,
            scope="other:read",
        )
        interactive = AsyncMock(return_value=(LTM_M2M_AGENT_CLIENT_ID, "user"))
        with (
            patch("scrutator.auth.verifier.verify_oidc_token", interactive),
            pytest.raises(Unauthenticated),
        ):
            await _verify_ltm_claims(claims, algorithm="RS256")
        interactive.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "identity_claim",
        [
            [LTM_M2M_AGENT_CLIENT_ID, "other-agent"],
            {"client": LTM_M2M_AGENT_CLIENT_ID},
        ],
        ids=["list", "dict"],
    )
    async def test_non_scalar_unverified_identity_claims_deny_without_interactive_fallback(self, identity_claim):
        claims = _ltm_claims(
            aud="urn:arcanada:other-service",
            sub=identity_claim,
            client_id=identity_claim,
            scope="other:read",
        )
        interactive = AsyncMock(return_value=(LTM_M2M_AGENT_CLIENT_ID, "user"))
        with (
            patch("scrutator.auth.verifier.verify_oidc_token", interactive),
            pytest.raises(Unauthenticated),
        ):
            await _verify_ltm_claims(claims, algorithm="RS256")
        interactive.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_malformed_muneral_profile_never_falls_back_to_interactive(self):
        claims = _ltm_claims()
        del claims["aud"]
        del claims["scope"]
        interactive = AsyncMock(return_value=(LTM_M2M_CLIENT_ID, "user"))
        with (
            patch("scrutator.auth.verifier.verify_oidc_token", interactive),
            pytest.raises(Unauthenticated),
        ):
            await _verify_ltm_claims(claims)
        interactive.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_other_eddsa_m2m_audience_never_falls_back_to_interactive(self):
        claims = _ltm_claims(
            aud="https://support.arcanada.one/api/v1/admin",
            sub="control-support-m2m",
            client_id="control-support-m2m",
            scope="support:tickets.admin",
        )
        interactive = AsyncMock(return_value=("control-support-m2m", "user"))
        with (
            patch("scrutator.auth.verifier.verify_oidc_token", interactive),
            pytest.raises(Unauthenticated),
        ):
            await _verify_ltm_claims(claims)
        interactive.assert_not_awaited()


class TestLtmM2mSettings:
    def test_exact_trust_profile_defaults_are_pinned(self):
        configured = Settings()
        assert configured.auth_ltm_issuer == LTM_M2M_ISSUER
        assert configured.auth_ltm_audience == LTM_M2M_AUDIENCE
        assert configured.auth_ltm_scope == LTM_M2M_SCOPE
        assert configured.auth_ltm_client_id == LTM_M2M_CLIENT_ID
        assert configured.auth_ltm_observer_client_id == LTM_M2M_OBSERVER_CLIENT_ID
        assert configured.auth_ltm_agent_client_id == LTM_M2M_AGENT_CLIENT_ID
        assert configured.auth_ltm_max_token_lifetime_seconds == 300

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("auth_ltm_issuer", "https://auth.arcanada.one"),
            ("auth_ltm_audience", "https://auth.arcanada.ai/api/v1/admin"),
            ("auth_ltm_scope", "namespace:read"),
            ("auth_ltm_client_id", "other-service"),
            ("auth_ltm_observer_client_id", "other-service"),
            ("auth_ltm_agent_client_id", "other-service"),
            ("auth_ltm_agent_client_id", "*"),
            ("auth_ltm_agent_client_id", [LTM_M2M_AGENT_CLIENT_ID]),
            ("auth_ltm_max_token_lifetime_seconds", 3600),
        ],
    )
    def test_trust_profile_cannot_be_widened_by_configuration(self, field, value):
        with pytest.raises(ValidationError):
            Settings(**{field: value})
