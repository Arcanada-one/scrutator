"""KCP-0017: bounded retry of `arc_api_*` introspection + an explained enforce-mode 401.

The project Scrutator instance answered 401 intermittently for a valid service key. These tests
pin the two changes made in response:

- only TRANSIENT introspection failures (timeout, connect/read/protocol error, 429/500/502/503/504)
  are retried — at most `auth_introspect_retries` times, inside an overall deadline, with a
  0.2 s / 0.5 s backoff (a 429 Retry-After is honoured up to 1 s);
- a definitive answer (401/403/other 4xx, invalid JSON, inactive token) is NEVER retried;
- fail-closed is unchanged: spent retries still raise Unauthenticated;
- in enforce mode the 401 is logged with its reason, and the response body is unchanged.

Uses a real httpx.AsyncClient over httpx.MockTransport: no network call ever happens.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from scrutator.auth.dependency import require_tenant_context
from scrutator.auth.verifier import Unauthenticated, verify_service_token
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
    app = FastAPI()

    @app.get("/v1/probe")
    async def probe(ctx=Depends(require_tenant_context)):  # noqa: B008
        return {"principal": ctx.principal_id}

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
            resp = TestClient(_app()).get("/v1/probe", headers={"Authorization": f"Bearer {TOKEN}"})

        assert resp.status_code == 401
        assert resp.json() == {"detail": "unauthenticated"}
        denies = [r.getMessage() for r in caplog.records if r.name == "scrutator.auth.dependency"]
        assert len(denies) == 1
        assert "GET /v1/probe" in denies[0]
        assert "introspection returned status 401" in denies[0]
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
            resp = TestClient(_app()).get("/v1/probe", headers={"Authorization": f"Bearer {TOKEN}"})

        assert resp.status_code == 401
        assert resp.json() == {"detail": "unauthenticated"}
        denies = [r.getMessage() for r in caplog.records if r.name == "scrutator.auth.dependency"]
        assert len(denies) == 1
        assert "introspection returned status 503 (attempt 3/3)" in denies[0]
