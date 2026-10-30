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

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict

import httpx
import jwt
from jwt import PyJWKClient

from scrutator.auth.models import VerifiedPrincipal
from scrutator.config import settings

logger = logging.getLogger(__name__)


# Failure classes carried by Unauthenticated and logged as one explicit field (KCP-0017).
FAILURE_DENIED = "denied"  # a definitive answer about the credential (or a missing/invalid credential)
FAILURE_RATE_LIMITED = "429"  # introspection rate-limited (e.g. the edge in front of Auth Arcana)
FAILURE_SERVER = "5xx"  # introspection answered 500/502/503/504
FAILURE_TRANSPORT = "transport"  # timeout, connection, read or protocol error reaching introspection


class Unauthenticated(Exception):
    """Raised when a bearer credential cannot be verified. Always fail-closed.

    `failure_class` says WHY in one field, so a 401 caused by an unreachable or rate-limited
    introspection endpoint is distinguishable in the log from a credential that was refused.
    """

    def __init__(self, message: str, failure_class: str = FAILURE_DENIED) -> None:
        super().__init__(message)
        self.failure_class = failure_class


_jwks_client: PyJWKClient | None = None
_jwks_client_url: str | None = None

LTM_M2M_ISSUER = "https://auth.arcanada.ai"
LTM_M2M_AUDIENCE = "urn:arcanada:scrutator:ltm"
LTM_M2M_SCOPE = "kb:ltm.read"
LTM_M2M_WRITE_SCOPE = "kb:ltm.write"
LTM_M2M_CLIENT_ID = "muneral-kb-sync"
LTM_M2M_OBSERVER_CLIENT_ID = "kb-observer"
LTM_M2M_AGENT_CLIENT_ID = "arcana-agent-kb-reader"
_LTM_REQUIRED_CLAIMS = ("exp", "iat", "nbf", "iss", "aud", "sub", "client_id", "scope")


def parse_scopes(raw: object, *, required: str, optional: frozenset[str]) -> frozenset[str]:
    """Parse an OAuth `scope` claim into the set it grants — fail-closed (A2-308).

    Replaces an exact string equality (`scope == required`). The relaxation is deliberately
    the smallest one that can express "this token may also write": the claim must still be a
    space-separated string, must still carry `required`, and every other value in it must be
    one this service knows. An unrecognized scope denies rather than being ignored, so a
    widened or attacker-shaped claim can never ride along unnoticed.
    """
    if not isinstance(raw, str):
        raise Unauthenticated("scope claim must be a string")
    if not required:
        raise Unauthenticated("required scope not configured")
    granted = frozenset(raw.split())
    if required not in granted:
        raise Unauthenticated("token scope mismatch")
    unknown = granted - {required} - optional
    if unknown:
        raise Unauthenticated("token carries unrecognized scopes")
    return granted


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


# KCP-0017: introspection outcomes that say nothing about the credential itself. Only these are
# retried; every other outcome is a definitive answer and is returned/raised immediately.
_TRANSIENT_HTTP_ERRORS = (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)
_TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})
_RETRY_BACKOFF_SECONDS = (0.2, 0.5)
_RETRY_AFTER_CAP_SECONDS = 1.0
_INTROSPECT_ATTEMPT_TIMEOUT_SECONDS = 5.0
# Indirections so tests can drive the clock and the backoff without touching the global modules.
_monotonic = time.monotonic
_sleep = asyncio.sleep
_INTROSPECT_CACHE_MAX_ENTRIES = 1024


def _status_failure_class(status: int) -> str:
    return FAILURE_RATE_LIMITED if status == 429 else FAILURE_SERVER


def _token_cache_key(token: str) -> str:
    """The only form in which a token is ever held by the cache: its sha256 hex digest."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class _IntrospectionCache:
    """Per-process POSITIVE cache of successful service-token verifications (KCP-0017).

    Holds only principals that passed every check in `verify_service_token`; a failure, a
    denial or an inactive token is never stored. Keyed by sha256(token). Bounded: expired
    entries are dropped first, then the oldest insertion. The TTL is read at insertion time
    and stored per entry; a TTL of 0 stores nothing.
    """

    def __init__(self, max_entries: int = _INTROSPECT_CACHE_MAX_ENTRIES) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[float, VerifiedPrincipal]] = OrderedDict()

    def get(self, key: str, now: float) -> VerifiedPrincipal | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, principal = entry
        if now >= expires_at:
            del self._entries[key]
            return None
        return principal

    def put(self, key: str, principal: VerifiedPrincipal, now: float, ttl: float) -> None:
        if ttl <= 0 or self.max_entries <= 0:
            return
        self._entries.pop(key, None)
        if len(self._entries) >= self.max_entries:
            for stale in [k for k, (exp, _) in self._entries.items() if now >= exp]:
                del self._entries[stale]
        while len(self._entries) >= self.max_entries:
            self._entries.popitem(last=False)
        self._entries[key] = (now + ttl, principal)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> list[str]:
        return list(self._entries)


_introspection_cache = _IntrospectionCache()


def _retry_delay(attempt_index: int, resp: httpx.Response | None) -> float:
    """Backoff before the next attempt: 0.2 s, then 0.5 s; a 429 Retry-After is honoured up to 1 s."""
    delay = _RETRY_BACKOFF_SECONDS[min(attempt_index, len(_RETRY_BACKOFF_SECONDS) - 1)]
    if resp is not None and resp.status_code == 429:
        raw = resp.headers.get("retry-after")
        try:
            requested = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            requested = None
        if requested is not None and requested >= 0:
            delay = min(requested, _RETRY_AFTER_CAP_SECONDS)
    return delay


async def _post_introspection(token: str) -> httpx.Response:
    """POST to the introspection endpoint, retrying only transient failures within a deadline.

    Returns the first non-transient response (any status the caller must judge). Raises
    Unauthenticated when the transient failures outlast the retries or the overall deadline,
    or on a non-transient transport error. Logs every failed attempt by reason class only —
    never the token, never a response body.
    """
    max_attempts = 1 + settings.auth_introspect_retries
    deadline = _monotonic() + settings.auth_introspect_retry_deadline_seconds
    async with httpx.AsyncClient(timeout=_INTROSPECT_ATTEMPT_TIMEOUT_SECONDS) as client:
        attempt = 0
        while True:
            attempt += 1
            remaining = deadline - _monotonic()
            resp: httpx.Response | None = None
            try:
                resp = await client.post(
                    settings.auth_arcana_introspect_url,
                    json={"token": token},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=max(0.001, min(_INTROSPECT_ATTEMPT_TIMEOUT_SECONDS, remaining)),
                )
            except _TRANSIENT_HTTP_ERRORS as exc:
                reason = f"introspection request failed: {type(exc).__name__}"
            except httpx.HTTPError as exc:
                logger.warning(
                    "service-token introspection attempt %d/%d failed (class=%s, %s); not retried",
                    attempt,
                    max_attempts,
                    FAILURE_TRANSPORT,
                    type(exc).__name__,
                )
                raise Unauthenticated(f"introspection request failed: {type(exc).__name__}", FAILURE_TRANSPORT) from exc
            else:
                if resp.status_code not in _TRANSIENT_STATUSES:
                    return resp
                reason = f"introspection returned status {resp.status_code}"
            failure_class = FAILURE_TRANSPORT if resp is None else _status_failure_class(resp.status_code)

            delay = _retry_delay(attempt - 1, resp)
            if attempt >= max_attempts or _monotonic() + delay >= deadline:
                logger.warning(
                    "service-token introspection attempt %d/%d failed (class=%s, %s); giving up",
                    attempt,
                    max_attempts,
                    failure_class,
                    reason,
                )
                raise Unauthenticated(f"{reason} (attempt {attempt}/{max_attempts})", failure_class)
            logger.warning(
                "service-token introspection attempt %d/%d failed (class=%s, %s); retrying in %.1fs",
                attempt,
                max_attempts,
                failure_class,
                reason,
                delay,
            )
            await _sleep(delay)


async def verify_service_token(token: str) -> VerifiedPrincipal:
    """Verify an `arc_api_*` service token via Auth Arcana introspection.

    Raises Unauthenticated on any failure (network error, non-200, inactive token,
    missing principal_id, unrecognized scope) — fail-closed. A successful verification is
    cached per process for `auth_introspect_cache_ttl_seconds` under sha256(token); nothing
    else is cached. Transient introspection failures are retried a bounded number of times
    first; definitive answers (401/403/other 4xx, invalid JSON, inactive token) are never
    retried (KCP-0017).
    """
    if not settings.auth_arcana_introspect_url:
        raise Unauthenticated("Auth Arcana introspection endpoint not configured")
    cache_key = _token_cache_key(token)
    cached = _introspection_cache.get(cache_key, _monotonic())
    if cached is not None:
        return cached
    principal = await _introspect_service_token(token)
    # Only a principal that passed every check in _introspect_service_token reaches this line:
    # failures, denials and inactive tokens raise before it and are never cached.
    _introspection_cache.put(cache_key, principal, _monotonic(), settings.auth_introspect_cache_ttl_seconds)
    return principal


async def _introspect_service_token(token: str) -> VerifiedPrincipal:
    """One uncached verification: introspect (with bounded transient retry) and check the answer."""
    resp = await _post_introspection(token)
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
    scopes = parse_scopes(
        data.get("scope"),
        required=settings.auth_service_scope,
        optional=frozenset({settings.auth_ltm_write_scope}),
    )
    return VerifiedPrincipal(principal_id=principal_id, principal_type="service", scopes=scopes)


async def verify_oidc_token(token: str) -> VerifiedPrincipal:
    """Verify a legacy interactive OIDC token. Returns the principal and its scopes.

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
    # This profile has always tolerated extra scopes, so it is NOT narrowed to an
    # allowlist here (that would revoke live interactive sessions). Only the two scopes
    # this service understands are carried forward; anything else stays inert.
    granted = frozenset(scopes.split()) & {settings.auth_oidc_scope, settings.auth_ltm_write_scope}
    return VerifiedPrincipal(principal_id=principal_id, principal_type="user", scopes=granted)


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
        or LTM_M2M_WRITE_SCOPE in scopes
        or client_id in {LTM_M2M_CLIENT_ID, LTM_M2M_OBSERVER_CLIENT_ID, LTM_M2M_AGENT_CLIENT_ID}
        or subject in {LTM_M2M_CLIENT_ID, LTM_M2M_OBSERVER_CLIENT_ID, LTM_M2M_AGENT_CLIENT_ID}
    )


async def verify_ltm_m2m_token(token: str) -> VerifiedPrincipal:
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
    scopes = parse_scopes(
        claims.get("scope"),
        required=settings.auth_ltm_scope,
        optional=frozenset({settings.auth_ltm_write_scope}),
    )
    subject = claims.get("sub")
    client_id = claims.get("client_id")
    allowed_clients = {
        settings.auth_ltm_client_id,
        settings.auth_ltm_observer_client_id,
        settings.auth_ltm_agent_client_id,
        settings.auth_ltm_argana_client_id,
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
    return VerifiedPrincipal(principal_id=client_id, principal_type="service", scopes=scopes)


async def verify_bearer_token(authorization: str | None) -> VerifiedPrincipal:
    """Verify an `Authorization` header value. Returns the verified principal + its scopes.

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
