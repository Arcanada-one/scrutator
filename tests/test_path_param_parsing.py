"""A path parameter the caller writes is parsed by the ROUTE, never by the database driver (A2-324).

Production answered `GET /v1/ltm/jobs/canary-probe` with 500 and `GET /v1/edges/canary-probe` with a
503 that carried asyncpg's error text: both parameters were declared `str` and handed straight to a
`$1::uuid` cast, so the first thing that parsed the caller's input was the driver. The repository
doubles here refuse a non-UUID the way asyncpg does — they raise `asyncpg.DataError` — so a route
that forwards unparsed input fails these tests the way it failed on production.

The set of templated routes is read from the application's own OpenAPI document, not listed by
hand: a new `{param}` route that is not measured here is a red test, not a silent gap.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest
from fastapi.testclient import TestClient

from tests.conftest import override_tenant_context

ABSENT = "00000000-0000-0000-0000-000000000000"
MALFORMED = "canary-probe"


def _driver_like(result):
    """What asyncpg does with `$1::uuid` and a string that is not one: DataError, before any query."""

    async def lookup(value, *args, **kwargs):
        try:
            uuid.UUID(str(value))
        except ValueError as error:
            raise asyncpg.DataError(f"invalid input for query argument $1: {value!r} ({error})") from error
        return result

    return AsyncMock(side_effect=lookup)


# route template -> (repository symbol the handler calls, value it returns for an absent id,
#                    status for a well-formed absent id)
MEASURED = {
    "/v1/ltm/jobs/{job_id}": ("scrutator.ltm.router.repository.get_ltm_job", None, 404),
    "/v1/edges/{chunk_id}": ("scrutator.health.get_edges_for_chunk", [], 200),
}


def _app():
    from scrutator.health import app

    return app


def test_every_templated_route_is_measured_here():
    templated = {path for path in _app().openapi()["paths"] if "{" in path}
    assert templated, "no templated routes at all — the comparison would be vacuous"
    assert templated == set(MEASURED), f"templated routes not measured here: {sorted(templated ^ set(MEASURED))}"


@pytest.mark.parametrize("template", sorted(MEASURED))
def test_a_well_formed_absent_id_is_not_a_fault(template):
    symbol, absent, expected = MEASURED[template]
    mock = _driver_like(absent)
    app = _app()
    with patch(symbol, mock), override_tenant_context(app):
        response = TestClient(app, raise_server_exceptions=False).get(template.split("{")[0] + ABSENT)
    assert response.status_code == expected, response.text
    assert mock.await_args.args[0] == ABSENT


@pytest.mark.parametrize("template", sorted(MEASURED))
def test_a_malformed_id_is_the_callers_422_and_never_reaches_the_driver(template):
    symbol, absent, _ = MEASURED[template]
    mock = _driver_like(absent)
    app = _app()
    with patch(symbol, mock), override_tenant_context(app):
        response = TestClient(app, raise_server_exceptions=False).get(template.split("{")[0] + MALFORMED)
    assert response.status_code == 422, response.text
    assert mock.await_count == 0
    # The driver's words must not reach the caller either (the 503 on production quoted them).
    assert "asyncpg" not in response.text and "query argument" not in response.text
