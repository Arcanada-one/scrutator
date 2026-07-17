"""ASGI-level request-body limits for the machine batch index route."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from scrutator.config import settings
from scrutator.db.models import BatchIndexSucceeded
from scrutator.health import app


def _scope(headers: list[tuple[bytes, bytes]] | None = None) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/index/batch",
        "raw_path": b"/v1/index/batch",
        "query_string": b"",
        "root_path": "",
        "headers": headers or [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }


@pytest.mark.asyncio
async def test_absent_content_length_stops_receiving_when_stream_crosses_cap():
    chunks = [
        {"type": "http.request", "body": b"x" * 700_000, "more_body": True},
        {"type": "http.request", "body": b"y" * 400_000, "more_body": True},
        {"type": "http.request", "body": b"must-not-be-read", "more_body": False},
    ]
    receive_calls = 0
    sent: list[dict] = []

    async def receive() -> dict:
        nonlocal receive_calls
        message = chunks[receive_calls]
        receive_calls += 1
        return message

    async def send(message: dict) -> None:
        sent.append(message)

    await app(_scope(), receive, send)

    start = next(message for message in sent if message["type"] == "http.response.start")
    assert start["status"] == 413
    assert receive_calls == 2


@pytest.mark.parametrize(
    ("content_length", "expected_status"),
    [
        (b"not-a-number", 400),
        (b"-1", 400),
        (b"1048577", 413),
        (b"9" * 5_000, 413),
    ],
)
@pytest.mark.asyncio
async def test_invalid_or_oversized_content_length_rejected_without_reading_body(content_length, expected_status):
    sent: list[dict] = []
    receive_called = False

    async def receive() -> dict:
        nonlocal receive_called
        receive_called = True
        raise AssertionError("body must not be read after Content-Length rejection")

    async def send(message: dict) -> None:
        sent.append(message)

    await app(
        _scope([(b"content-type", b"application/json"), (b"content-length", content_length)]),
        receive,
        send,
    )

    start = next(message for message in sent if message["type"] == "http.response.start")
    assert start["status"] == expected_status
    assert receive_called is False


def test_normal_sized_body_is_replayed_for_fastapi_parsing():
    original = (settings.feeder_token, settings.feeder_namespaces)
    settings.feeder_token = "feeder-secret"
    settings.feeder_namespaces = "self-improvement"
    body = {
        "documents": [
            {
                "content": "normal lesson",
                "source_path": "normal.md",
                "namespace": "self-improvement",
            }
        ]
    }
    try:
        with (
            patch(
                "scrutator.health.index_documents",
                new_callable=AsyncMock,
                return_value=[BatchIndexSucceeded(source_path="normal.md", chunks_indexed=1)],
            ),
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/index/batch",
                content=json.dumps(body),
                headers={
                    "Content-Type": "application/json",
                    "X-KB-Feeder-Token": "feeder-secret",
                },
            )
    finally:
        settings.feeder_token, settings.feeder_namespaces = original

    assert response.status_code == 200
    assert response.json()["results"][0]["source_path"] == "normal.md"
