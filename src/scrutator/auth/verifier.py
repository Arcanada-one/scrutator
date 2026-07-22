"""Bearer credential verification (SRCH-0023 Step 1).

Three credential profiles, per the Auth Arcana mandate (no bespoke Scrutator key system):
- `arc_api_*` service token — verified via Auth Arcana's introspection endpoint.
- Dedicated Scrutator LTM M2M JWT — strict EdDSA issuer/audience/client/scope profile.
- Legacy interactive OIDC access token — separate RS256/ES256 JWKS profile.

Fail-closed: any verification error (bad signature, expired, JWKS/introspection unreachable,
missing claim) raises `Unauthenticated` — never returns a principal. Never fail-open to the
legacy unauthenticated behaviour.
"""

from __future__ import annotations

import httpx
import jwt
from jwt import PyJWKClient

from scrutator.config import settings


class Unauthenticated(Exception):
    """Raised when a bearer credential cannot be verified. Always fail-closed."""


_jwks_client: PyJWKClient | None = None
_jwks_client_url: str | None = None

LTM_M2M_ISSUER = "https://auth.arcanada.ai"
LTM_M2M_AUDIENCE = "urn:arcanada:scrutator:ltm"
LTM_M2M_SCOPE = "kb:ltm.read"
LTM_M2M_CLIENT_ID = "muneral-kb-sync"
LTM_M2M_OBSERVER_CLIENT_ID = "kb-observer"
LTM_M2M_AGENT_CLIENT_ID = "arcana-agent-kb-reader"
_LTM_REQUIRED_CLAIMS = ("exp", "iat", "nbf", "iss", "aud", "sub", "client_id", "scope")


def _get_jwks_client(jwks_url: str) -> PyJWKClient:
    """Return a cached PyJWKClient for the given URL, recreating it if the URL changes."""
    global _jwks_client, _jwks_client_url  # noqa: PLW0603
    if _jwks_client is None or _jwks_client_url != jwks_url:
        _jwks_client = PyJWKClient(
            jwks_url,
            cache_keys=True,
            lifespan=300,
            headers={"User-Agent": "Arcanada-Scrutator-JWKS/1.0"},
        )
        _jwks_client_url = jwks_url
    return _jwks_client


async def verify_service_token(token: str) -> tuple[str, str]:
    """Verify an `arc_api_*` service token via Auth Arcana introspection.

    Returns (principal_id, "service"). Raises Unauthenticated on any failure
    (network error, non-200, inactive token, missing principal_id) — fail-closed.
    """
    if not settings.auth_arcana_introspect_url:
        raise Unauthenticated("Auth Arcana introspection endpoint not configured")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                settings.auth_arcana_introspect_url,
                json={"token": token},
                headers={"Authorization": f"Bearer {token}"},
            )
    except httpx.HTTPError as exc:
        raise Unauthenticated(f"introspection request failed: {type(exc).__name__}") from exc
    if resp.status_code != 200:
        raise Unauthenticated(f"introspection returned status {resp.status_code}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise Unauthenticated("introspection returned invalid JSON") from exc
    if not isinstance(data, dict) or data.get("active") is not True:
        raise Unauthenticated("service token inactive")
    principal_id = data.get("principal_id")
    if not principal_id:
        raise Unauthenticated("introspection response missing principal_id")
    if not settings.auth_service_audience or not settings.auth_service_scope:
        raise Unauthenticated("service-token resource profile not configured")
    if data.get("audience") != settings.auth_service_audience:
        raise Unauthenticated("service token audience mismatch")
    if data.get("scope") != settings.auth_service_scope:
        raise Unauthenticated("service token scope mismatch")
    return principal_id, "service"


async def verify_oidc_token(token: str) -> tuple[str, str]:
    """Verify a legacy interactive OIDC token. Returns (principal_id, "user").

    Any JWKS lookup failure (host unreachable, network error) or JWT validation failure
    (expired, bad signature, malformed) raises Unauthenticated — fail-closed, never
    fail-open to an unverified principal.
    """
    if not settings.auth_arcana_jwks_url:
        raise Unauthenticated("Auth Arcana JWKS endpoint not configured")
    if not settings.auth_oidc_issuer or not settings.auth_oidc_audience or not settings.auth_oidc_scope:
        raise Unauthenticated("OIDC resource profile not configured")
    try:
        jwks_client = _get_jwks_client(settings.auth_arcana_jwks_url)
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            issuer=settings.auth_oidc_issuer,
            audience=settings.auth_oidc_audience,
            options={"require": ["exp", "sub", "iss", "aud"], "strict_aud": True},
        )
    except jwt.PyJWTError as exc:
        raise Unauthenticated(f"JWT verification failed: {type(exc).__name__}") from exc
    except Exception as exc:  # JWKS fetch/network failure — fail closed, not fail-open
        raise Unauthenticated(f"JWKS lookup failed: {type(exc).__name__}") from exc
    principal_id = claims.get("sub")
    if not principal_id:
        raise Unauthenticated("token missing sub claim")
    scopes = claims.get("scope")
    if not isinstance(scopes, str) or settings.auth_oidc_scope not in scopes.split():
        raise Unauthenticated("OIDC token scope mismatch")
    return principal_id, "user"


def _unverified_claims(token: str) -> dict:
    """Parse only enough unsigned JWT data to select a stricter verifier profile."""
    try:
        claims = jwt.decode(
            token,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_iat": False,
                "verify_nbf": False,
                "verify_aud": False,
                "verify_iss": False,
            },
        )
    except jwt.PyJWTError as exc:
        raise Unauthenticated(f"JWT routing failed: {type(exc).__name__}") from exc
    if not isinstance(claims, dict):
        raise Unauthenticated("JWT routing failed: claims must be an object")
    return claims


def _is_ltm_m2m_candidate(token: str, claims: dict) -> bool:
    """Recognize every stable marker of the dedicated principal, including damaged tokens."""
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise Unauthenticated(f"JWT routing failed: {type(exc).__name__}") from exc
    audience = claims.get("aud")
    audiences = audience if isinstance(audience, list) else [audience]
    scope = claims.get("scope")
    scopes = scope.split() if isinstance(scope, str) else []
    client_id = claims.get("client_id")
    subject = claims.get("sub")
    if client_id is not None and not isinstance(client_id, str):
        raise Unauthenticated("JWT routing failed: client_id claim must be a string")
    if subject is not None and not isinstance(subject, str):
        raise Unauthenticated("JWT routing failed: sub claim must be a string")
    return (
        header.get("alg") == "EdDSA"
        or LTM_M2M_AUDIENCE in audiences
        or LTM_M2M_SCOPE in scopes
        or client_id in {LTM_M2M_CLIENT_ID, LTM_M2M_OBSERVER_CLIENT_ID, LTM_M2M_AGENT_CLIENT_ID}
        or subject in {LTM_M2M_CLIENT_ID, LTM_M2M_OBSERVER_CLIENT_ID, LTM_M2M_AGENT_CLIENT_ID}
    )


async def verify_ltm_m2m_token(token: str) -> tuple[str, str]:
    """Verify a dedicated LTM reader JWT under an exact, fail-closed profile."""
    if not settings.auth_arcana_jwks_url:
        raise Unauthenticated("Auth Arcana JWKS endpoint not configured")
    try:
        jwks_client = _get_jwks_client(settings.auth_arcana_jwks_url)
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["EdDSA"],
            issuer=settings.auth_ltm_issuer,
            audience=settings.auth_ltm_audience,
            options={"require": list(_LTM_REQUIRED_CLAIMS), "strict_aud": True},
        )
    except jwt.PyJWTError as exc:
        raise Unauthenticated(f"LTM M2M JWT verification failed: {type(exc).__name__}") from exc
    except Exception as exc:  # JWKS fetch/network failure — fail closed
        raise Unauthenticated(f"JWKS lookup failed: {type(exc).__name__}") from exc

    if claims.get("iss") != settings.auth_ltm_issuer:
        raise Unauthenticated("LTM M2M token issuer mismatch")
    if claims.get("aud") != settings.auth_ltm_audience:
        raise Unauthenticated("LTM M2M token audience mismatch")
    if claims.get("scope") != settings.auth_ltm_scope:
        raise Unauthenticated("LTM M2M token scope mismatch")
    subject = claims.get("sub")
    client_id = claims.get("client_id")
    allowed_clients = {
        settings.auth_ltm_client_id,
        settings.auth_ltm_observer_client_id,
        settings.auth_ltm_agent_client_id,
    }
    if subject != client_id or client_id not in allowed_clients:
        raise Unauthenticated("LTM M2M token client binding mismatch")
    issued_at = claims.get("iat")
    not_before = claims.get("nbf")
    expires_at = claims.get("exp")
    if (
        not isinstance(issued_at, int)
        or isinstance(issued_at, bool)
        or not isinstance(not_before, int)
        or isinstance(not_before, bool)
        or not isinstance(expires_at, int)
        or isinstance(expires_at, bool)
        or not_before != issued_at
        or expires_at - issued_at != settings.auth_ltm_max_token_lifetime_seconds
    ):
        raise Unauthenticated("LTM M2M token lifetime mismatch")
    return client_id, "service"


async def verify_bearer_token(authorization: str | None) -> tuple[str, str]:
    """Verify an `Authorization` header value. Returns (principal_id, principal_type).

    Fail-closed: missing header, malformed scheme, or any verification error raises
    Unauthenticated.
    """
    if not authorization:
        raise Unauthenticated("missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise Unauthenticated("Authorization header must be 'Bearer <token>'")
    if token.startswith("arc_api_"):
        return await verify_service_token(token)
    claims = _unverified_claims(token)
    if _is_ltm_m2m_candidate(token, claims):
        return await verify_ltm_m2m_token(token)
    return await verify_oidc_token(token)
