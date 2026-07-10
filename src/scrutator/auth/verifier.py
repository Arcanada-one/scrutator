"""Bearer credential verification (SRCH-0023 Step 1).

Two credential shapes, per the Auth Arcana mandate (no bespoke Scrutator key system):
- `arc_api_*` service token — verified via Auth Arcana's introspection endpoint.
- OIDC access token — verified via JWKS (`auth_arcana_jwks_url`), short-TTL cached client.

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


def _get_jwks_client(jwks_url: str) -> PyJWKClient:
    """Return a cached PyJWKClient for the given URL, recreating it if the URL changes."""
    global _jwks_client, _jwks_client_url  # noqa: PLW0603
    if _jwks_client is None or _jwks_client_url != jwks_url:
        _jwks_client = PyJWKClient(jwks_url, cache_keys=True, lifespan=300)
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
    data = resp.json()
    if not data.get("active"):
        raise Unauthenticated("service token inactive")
    principal_id = data.get("principal_id")
    if not principal_id:
        raise Unauthenticated("introspection response missing principal_id")
    return principal_id, "service"


async def verify_oidc_token(token: str) -> tuple[str, str]:
    """Verify an OIDC access token via Auth Arcana JWKS. Returns (principal_id, "user").

    Any JWKS lookup failure (host unreachable, network error) or JWT validation failure
    (expired, bad signature, malformed) raises Unauthenticated — fail-closed, never
    fail-open to an unverified principal.
    """
    if not settings.auth_arcana_jwks_url:
        raise Unauthenticated("Auth Arcana JWKS endpoint not configured")
    try:
        jwks_client = _get_jwks_client(settings.auth_arcana_jwks_url)
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            options={"require": ["exp", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise Unauthenticated(f"JWT verification failed: {type(exc).__name__}") from exc
    except Exception as exc:  # JWKS fetch/network failure — fail closed, not fail-open
        raise Unauthenticated(f"JWKS lookup failed: {type(exc).__name__}") from exc
    principal_id = claims.get("sub")
    if not principal_id:
        raise Unauthenticated("token missing sub claim")
    return principal_id, "user"


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
    return await verify_oidc_token(token)
