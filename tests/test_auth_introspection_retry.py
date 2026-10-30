"""KCP-0017: bounded retry + positive cache of `arc_api_*` introspection; an explained 401.

The project Scrutator instance answered 401 intermittently for a valid service key. These tests
pin the two changes made in response:

- only TRANSIENT introspection failures (timeout, connect/read/protocol error, 429/500/502/503/504)
  are retried — at most `auth_introspect_retries` times, inside an overall deadline, with a
  0.2 s / 0.5 s backoff (a 429 Retry-After is honoured up to 1 s);
- a definitive answer (401/403/other 4xx, invalid JSON, inactive token) is NEVER retried;
- fail-closed is unchanged: spent retries still raise Unauthenticated;
- in enforce mode the 401 is logged with its failure class and reason; the body is unchanged;
- a SUCCESSFUL verification is cached per process for the TTL under sha256(token); failures,
  denials and inactive tokens are never cached (KCP-ROOT-REPLY-17: the public introspection
  route answered about half of a burst with 429 at the edge).

Uses a real httpx.AsyncClient over httpx.MockTransport: no network call ever happens.
"""

from __future__ import annotations

import hashlib
import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import scrutator.auth.verifier as verifier_module
from scrutator.auth.dependency import require_tenant_context
from scrutator.auth.models import VerifiedPrincipal
from scrutator.auth.verifier import Unauthenticated, _IntrospectionCache, verify_service_token
from scrutator.config import Settings

INTROSPECT_URL = "https://auth.example.invalid/api/v1/introspect"
TOKEN = "arc_api_kcp17-fixture-not-a-secret"
GOOD_BODY = {"active": True, "principal_id": "svc-kcp17", "audience": "urn:test:scrutator", "scope": "kb:read"}

_RealAsyncClient = httpx.AsyncClient


def _settings(**overrides) -> Settings:
    values = {
        "auth_arcana_introspect_url": INTROSPECT_URL,
        "auth_service_audience": "urn:test:scrutator",
        "auth_service_scope": "kb:read",
        "auth_introspect_retries": 2,
        "auth_introspect_retry_deadline_seconds": 6.0,
        "auth_introspect_cache_ttl_seconds": 60.0,
    }
    values.update(overrides)
    return Settings(**values)


class _Script:
    """Answers each introspection attempt from a list of steps (a Response or an exception)."""

    def __init__(self, steps, clock=None, attempt_cost=0.0):
        self.steps = list(steps)
        self.calls = 0
        self.timeouts: list[float | None] = []
        self.clock = clock
        self.attempt_cost = attempt_cost

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.timeouts.append(request.extensions.get("timeout", {}).get("read"))
        if self.clock is not None:
            self.clock.now += self.attempt_cost
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def _client_factory(script: _Script):
    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(script.handler)
        return _RealAsyncClient(*args, **kwargs)

    return factory


async def _run(script: _Script, settings_obj: Settings | None = None, clock: _Clock | None = None):
    clock = clock or _Clock()
    with (
        patch("scrutator.auth.verifier.settings", settings_obj or _settings()),
        patch("scrutator.auth.verifier.httpx.AsyncClient", side_effect=_client_factory(script)),
        patch("scrutator.auth.verifier._monotonic", clock.monotonic),
        patch("scrutator.auth.verifier._sleep", new=AsyncMock(side_effect=clock.sleep)) as sleep,
    ):
        try:
            return await verify_service_token(TOKEN), sleep
        except Unauthenticated as exc:
            return exc, sleep


async def _run_many(
    tokens, script: _Script, settings_obj: Settings | None = None, clock: _Clock | None = None, advance=None
):
    """Verify several tokens in sequence under ONE patched environment (the cache persists).

    `advance[i]` seconds are added to the clock before verification i.
    """
    clock = clock or _Clock()
    results = []
    with (
        patch("scrutator.auth.verifier.settings", settings_obj or _settings()),
        patch("scrutator.auth.verifier.httpx.AsyncClient", side_effect=_client_factory(script)),
        patch("scrutator.auth.verifier._monotonic", clock.monotonic),
        patch("scrutator.auth.verifier._sleep", new=AsyncMock(side_effect=clock.sleep)),
    ):
        for i, token in enumerate(tokens):
            if advance:
                clock.now += advance[i]
            try:
                results.append(await verify_service_token(token))
            except Unauthenticated as exc:
                results.append(exc)
    return results


def _ok() -> httpx.Response:
    return httpx.Response(200, json=GOOD_BODY)


def _timeout() -> httpx.ReadTimeout:
    return httpx.ReadTimeout("timed out")


class TestTransientFailuresAreRetried:
    @pytest.mark.asyncio
    async def test_success_on_first_try_makes_one_call_and_no_sleep(self):
        script = _Script([_ok()])
        principal, sleep = await _run(script)
        assert principal.principal_id == "svc-kcp17"
        assert script.calls == 1
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_503_then_200_succeeds(self):
        script = _Script([httpx.Response(503), _ok()])
        principal, sleep = await _run(script)
        assert principal.principal_id == "svc-kcp17"
        assert script.calls == 2
        assert [c.args[0] for c in sleep.await_args_list] == [0.2]

    @pytest.mark.asyncio
    async def test_timeout_then_200_succeeds(self):
        script = _Script([_timeout(), _ok()])
        principal, _ = await _run(script)
        assert principal.principal_id == "svc-kcp17"
        assert script.calls == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure",
        [
            httpx.ConnectError("refused"),
            httpx.ReadError("reset"),
            httpx.RemoteProtocolError("closed"),
            httpx.ConnectTimeout("connect timed out"),
        ],
    )
    async def test_each_transient_transport_error_is_retried(self, failure):
        script = _Script([failure, _ok()])
        principal, _ = await _run(script)
        assert principal.principal_id == "svc-kcp17"
        assert script.calls == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    async def test_each_transient_status_is_retried(self, status):
        script = _Script([httpx.Response(status), _ok()])
        principal, _ = await _run(script)
        assert principal.principal_id == "svc-kcp17"
        assert script.calls == 2

    @pytest.mark.asyncio
    async def test_429_retry_after_is_honoured(self):
        script = _Script([httpx.Response(429, headers={"Retry-After": "0.7"}), _ok()])
        principal, sleep = await _run(script)
        assert principal.principal_id == "svc-kcp17"
        assert [c.args[0] for c in sleep.await_args_list] == [0.7]

    @pytest.mark.asyncio
    async def test_429_retry_after_is_capped_at_one_second(self):
        script = _Script([httpx.Response(429, headers={"Retry-After": "30"}), _ok()])
        principal, sleep = await _run(script)
        assert principal.principal_id == "svc-kcp17"
        assert [c.args[0] for c in sleep.await_args_list] == [1.0]

    @pytest.mark.asyncio
    async def test_backoff_is_point_two_then_point_five(self):
        script = _Script([httpx.Response(502), _timeout(), _ok()])
        principal, sleep = await _run(script)
        assert principal.principal_id == "svc-kcp17"
        assert script.calls == 3
        assert [c.args[0] for c in sleep.await_args_list] == [0.2, 0.5]

    @pytest.mark.asyncio
    async def test_three_transient_failures_stay_unauthenticated(self, caplog):
        script = _Script([httpx.Response(503), _timeout(), httpx.Response(503)])
        with caplog.at_level(logging.WARNING, logger="scrutator.auth.verifier"):
            result, _ = await _run(script)
        assert isinstance(result, Unauthenticated)
        assert "status 503" in str(result)
        assert "attempt 3/3" in str(result)
        assert script.calls == 3
        attempts = [r.getMessage() for r in caplog.records if "introspection attempt" in r.getMessage()]
        assert len(attempts) == 3
        assert "1/3" in attempts[0] and "ReadTimeout" in attempts[1] and "giving up" in attempts[2]
        assert all(TOKEN not in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_retries_setting_zero_disables_retry(self):
        script = _Script([httpx.Response(503), _ok()])
        result, _ = await _run(script, _settings(auth_introspect_retries=0))
        assert isinstance(result, Unauthenticated)
        assert script.calls == 1


class TestDefinitiveAnswersAreNotRetried:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    async def test_definitive_status_is_one_call(self, status):
        script = _Script([httpx.Response(status), _ok()])
        result, sleep = await _run(script)
        assert isinstance(result, Unauthenticated)
        assert f"status {status}" in str(result)
        assert script.calls == 1
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_json_is_one_call(self):
        script = _Script([httpx.Response(200, content=b"<html>not json"), _ok()])
        result, _ = await _run(script)
        assert isinstance(result, Unauthenticated)
        assert "invalid JSON" in str(result)
        assert script.calls == 1

    @pytest.mark.asyncio
    async def test_inactive_token_is_one_call(self):
        script = _Script([httpx.Response(200, json={"active": False}), _ok()])
        result, _ = await _run(script)
        assert isinstance(result, Unauthenticated)
        assert script.calls == 1

    @pytest.mark.asyncio
    async def test_missing_principal_is_one_call(self):
        script = _Script([httpx.Response(200, json={"active": True}), _ok()])
        result, _ = await _run(script)
        assert isinstance(result, Unauthenticated)
        assert "principal_id" in str(result)
        assert script.calls == 1

    @pytest.mark.asyncio
    async def test_non_transient_transport_error_is_one_call(self):
        script = _Script([httpx.UnsupportedProtocol("bad scheme"), _ok()])
        result, _ = await _run(script)
        assert isinstance(result, Unauthenticated)
        assert script.calls == 1


class TestDeadline:
    @pytest.mark.asyncio
    async def test_deadline_stops_retries_before_the_attempt_budget(self):
        # Each attempt burns 3 s of the 6 s deadline: attempt 1 ends at +3.0, the 0.2 s backoff
        # brings +3.2, attempt 2 ends at +6.2 — past the deadline, so attempt 3 never starts.
        clock = _Clock()
        script = _Script([_timeout(), _timeout(), _ok()], clock=clock, attempt_cost=3.0)
        result, _ = await _run(script, clock=clock)
        assert isinstance(result, Unauthenticated)
        assert script.calls == 2

    @pytest.mark.asyncio
    async def test_each_attempt_timeout_is_bounded_by_the_remaining_deadline(self):
        clock = _Clock()
        script = _Script([_timeout(), _ok()], clock=clock, attempt_cost=4.0)
        principal, _ = await _run(script, clock=clock)
        assert principal.principal_id == "svc-kcp17"
        # First attempt: min(5, 6) = 5. Second: 6 - 4 - 0.2 = 1.8 left.
        assert script.timeouts[0] == pytest.approx(5.0)
        assert script.timeouts[1] == pytest.approx(1.8)

    @pytest.mark.asyncio
    async def test_no_sleep_that_would_cross_the_deadline(self):
        clock = _Clock()
        script = _Script([_timeout(), _ok()], clock=clock, attempt_cost=5.9)
        result, sleep = await _run(script, clock=clock)
        assert isinstance(result, Unauthenticated)
        assert script.calls == 1
        sleep.assert_not_awaited()


def _app() -> FastAPI:
    # Registered without a decorator on purpose: this test-only endpoint is not a served route of
    # Scrutator and must not appear as one in the repository's route graph.
    async def probe(ctx=Depends(require_tenant_context)):  # noqa: B008
        return {"principal": ctx.principal_id}

    app = FastAPI()
    app.add_api_route("/kcp17-test/probe", probe, methods=["GET"])
    return app


class TestEnforceModeDenyIsExplained:
    def test_enforce_401_logs_reason_and_keeps_the_body(self, caplog):
        script = _Script([httpx.Response(401)])
        enforced = _settings(auth_enforce=True)
        with (
            patch("scrutator.auth.dependency.settings", enforced),
            patch("scrutator.auth.verifier.settings", enforced),
            patch("scrutator.auth.verifier.httpx.AsyncClient", side_effect=_client_factory(script)),
            caplog.at_level(logging.WARNING, logger="scrutator.auth.dependency"),
        ):
            resp = TestClient(_app()).get("/kcp17-test/probe", headers={"Authorization": f"Bearer {TOKEN}"})

        assert resp.status_code == 401
        assert resp.json() == {"detail": "unauthenticated"}
        denies = [r.getMessage() for r in caplog.records if r.name == "scrutator.auth.dependency"]
        assert len(denies) == 1
        assert "GET /kcp17-test/probe" in denies[0]
        assert "class=denied introspection returned status 401" in denies[0]
        assert all(TOKEN not in r.getMessage() for r in caplog.records)
        assert script.calls == 1

    def test_enforce_401_after_spent_retries_logs_the_transient_reason(self, caplog):
        script = _Script([httpx.Response(503), httpx.Response(503), httpx.Response(503)])
        enforced = _settings(auth_enforce=True)
        with (
            patch("scrutator.auth.dependency.settings", enforced),
            patch("scrutator.auth.verifier.settings", enforced),
            patch("scrutator.auth.verifier.httpx.AsyncClient", side_effect=_client_factory(script)),
            patch("scrutator.auth.verifier._sleep", new_callable=AsyncMock),
            caplog.at_level(logging.WARNING, logger="scrutator.auth.dependency"),
        ):
            resp = TestClient(_app()).get("/kcp17-test/probe", headers={"Authorization": f"Bearer {TOKEN}"})

        assert resp.status_code == 401
        assert resp.json() == {"detail": "unauthenticated"}
        denies = [r.getMessage() for r in caplog.records if r.name == "scrutator.auth.dependency"]
        assert len(denies) == 1
        assert "class=5xx introspection returned status 503 (attempt 3/3)" in denies[0]


class TestFailureClassIsExplicit:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("steps", "failure_class"),
        [
            ([httpx.Response(429), httpx.Response(429), httpx.Response(429)], "429"),
            ([httpx.Response(502), httpx.Response(503), httpx.Response(504)], "5xx"),
            ([httpx.ConnectError("x"), httpx.ReadError("x"), _timeout()], "transport"),
            ([httpx.UnsupportedProtocol("x")], "transport"),
            ([httpx.Response(401)], "denied"),
            ([httpx.Response(403)], "denied"),
            ([httpx.Response(200, json={"active": False})], "denied"),
        ],
    )
    async def test_unauthenticated_carries_the_failure_class(self, steps, failure_class, caplog):
        script = _Script(steps)
        with caplog.at_level(logging.WARNING, logger="scrutator.auth.verifier"):
            result, _ = await _run(script)
        assert isinstance(result, Unauthenticated)
        assert result.failure_class == failure_class
        if failure_class != "denied":
            assert any(f"class={failure_class}" in r.getMessage() for r in caplog.records)
        assert all(TOKEN not in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_429_then_success_retries_and_logs_the_class(self, caplog):
        script = _Script([httpx.Response(429), _ok()])
        with caplog.at_level(logging.WARNING, logger="scrutator.auth.verifier"):
            principal, _ = await _run(script)
        assert principal.principal_id == "svc-kcp17"
        assert script.calls == 2
        assert any("class=429" in r.getMessage() for r in caplog.records)


class TestPositiveCache:
    @pytest.mark.asyncio
    async def test_cache_hit_makes_no_second_http_call(self):
        script = _Script([_ok()])
        first, second = await _run_many([TOKEN, TOKEN], script)
        assert script.calls == 1
        assert first.principal_id == second.principal_id == "svc-kcp17"

    @pytest.mark.asyncio
    async def test_cached_principal_equals_a_fresh_verification(self):
        cached = (await _run_many([TOKEN, TOKEN], _Script([_ok()])))[1]
        verifier_module._introspection_cache.clear()
        fresh = (await _run_many([TOKEN], _Script([_ok()])))[0]
        assert isinstance(cached, VerifiedPrincipal)
        assert cached == fresh
        assert cached.scopes == frozenset({"kb:read"})

    @pytest.mark.asyncio
    async def test_a_different_token_is_not_served_from_the_cache(self):
        script = _Script([_ok(), httpx.Response(401)])
        first, second = await _run_many([TOKEN, TOKEN + "-other"], script)
        assert first.principal_id == "svc-kcp17"
        assert isinstance(second, Unauthenticated)
        assert script.calls == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "denial",
        [
            lambda: httpx.Response(401),
            lambda: httpx.Response(403),
            lambda: httpx.Response(200, json={"active": False}),
            lambda: httpx.Response(200, json={**GOOD_BODY, "audience": "urn:other"}),
        ],
    )
    async def test_denial_stays_denied_and_is_not_cached(self, denial):
        script = _Script([denial(), denial()])
        first, second = await _run_many([TOKEN, TOKEN], script)
        assert isinstance(first, Unauthenticated)
        assert isinstance(second, Unauthenticated)
        assert script.calls == 2
        assert len(verifier_module._introspection_cache) == 0

    @pytest.mark.asyncio
    async def test_spent_transient_failure_is_not_cached(self):
        script = _Script([httpx.Response(429), httpx.Response(429), httpx.Response(429), _ok()])
        first, second = await _run_many([TOKEN, TOKEN], script)
        assert isinstance(first, Unauthenticated)
        assert second.principal_id == "svc-kcp17"
        assert script.calls == 4

    @pytest.mark.asyncio
    async def test_entry_is_served_until_the_ttl_and_expires_after_it(self):
        script = _Script([_ok(), _ok()])
        results = await _run_many([TOKEN, TOKEN, TOKEN], script, advance=[0.0, 59.9, 0.2])
        assert all(r.principal_id == "svc-kcp17" for r in results)
        # 59.9 s later: still cached (1 call); 60.1 s after insertion: expired -> second call.
        assert script.calls == 2

    @pytest.mark.asyncio
    async def test_ttl_zero_disables_the_cache(self):
        script = _Script([_ok(), _ok()])
        await _run_many([TOKEN, TOKEN], script, _settings(auth_introspect_cache_ttl_seconds=0))
        assert script.calls == 2
        assert len(verifier_module._introspection_cache) == 0

    @pytest.mark.asyncio
    async def test_cache_size_is_bounded_and_evicts_the_oldest(self):
        small = _IntrospectionCache(max_entries=3)
        tokens = [f"{TOKEN}-{i}" for i in range(4)]
        script = _Script([_ok() for _ in range(5)])
        with patch.object(verifier_module, "_introspection_cache", small):
            await _run_many([*tokens, tokens[0]], script)
            assert len(small) == 3
        # tokens[0] was evicted by tokens[3], so its second verification went to the network.
        assert script.calls == 5

    def test_expired_entries_are_evicted_before_live_ones(self):
        principal = VerifiedPrincipal(principal_id="p", principal_type="service", scopes=frozenset())
        cache = _IntrospectionCache(max_entries=2)
        cache.put("old-expired", principal, now=0.0, ttl=1.0)
        cache.put("live", principal, now=0.5, ttl=60.0)
        cache.put("new", principal, now=10.0, ttl=60.0)
        assert cache.keys() == ["live", "new"]
        assert cache.get("live", 10.0) is principal

    @pytest.mark.asyncio
    async def test_token_is_never_a_cache_key_nor_logged(self, caplog):
        script = _Script([httpx.Response(429), _ok()])
        with caplog.at_level(logging.DEBUG):
            await _run_many([TOKEN, TOKEN], script)
        keys = verifier_module._introspection_cache.keys()
        assert keys == [hashlib.sha256(TOKEN.encode("utf-8")).hexdigest()]
        assert all(TOKEN not in k and "arc_api_" not in k for k in keys)
        assert all(TOKEN not in r.getMessage() for r in caplog.records)
