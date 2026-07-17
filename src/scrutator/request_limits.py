"""Bounded ASGI request-body handling for high-cost machine routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from starlette.responses import JSONResponse

ASGIReceive = Callable[[], Awaitable[dict[str, Any]]]
ASGISend = Callable[[dict[str, Any]], Awaitable[None]]


class BoundedRequestBodyMiddleware:
    """Read at most ``max_bytes`` and replay the bounded body downstream."""

    def __init__(self, app, *, path: str, max_bytes: int):
        self.app = app
        self.path = path
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: ASGIReceive, send: ASGISend) -> None:
        if scope["type"] != "http" or scope.get("path") != self.path:
            await self.app(scope, receive, send)
            return

        declared_size, rejection = _declared_size(scope.get("headers", []), self.max_bytes)
        if rejection is not None:
            await _reject(scope, receive, send, rejection)
            return

        body, rejection = await _read_bounded(receive, self.max_bytes)
        if rejection is not None:
            await _reject(scope, receive, send, rejection)
            return
        if declared_size is not None and declared_size != len(body):
            await _reject(scope, receive, send, 400)
            return

        await self.app(scope, _replay_body(body, receive), send)


def _declared_size(headers: list[tuple[bytes, bytes]], max_bytes: int) -> tuple[int | None, int | None]:
    values = [value for name, value in headers if name.lower() == b"content-length"]
    if not values:
        return None, None
    if len(values) != 1:
        return None, 400
    try:
        text = values[0].decode("ascii")
    except UnicodeDecodeError:
        return None, 400
    if not text.isdecimal():
        return None, 400
    size = int(text)
    return (size, 413) if size > max_bytes else (size, None)


async def _read_bounded(receive: ASGIReceive, max_bytes: int) -> tuple[bytes, int | None]:
    chunks: list[bytes] = []
    total = 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return b"", 400
        chunk = message.get("body", b"")
        total += len(chunk)
        if total > max_bytes:
            return b"", 413
        chunks.append(chunk)
        if not message.get("more_body", False):
            return b"".join(chunks), None


def _replay_body(body: bytes, receive: ASGIReceive) -> ASGIReceive:
    replayed = False

    async def replay() -> dict[str, Any]:
        nonlocal replayed
        if not replayed:
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return replay


async def _reject(scope: dict[str, Any], receive: ASGIReceive, send: ASGISend, status: int) -> None:
    detail = "batch request body too large" if status == 413 else "invalid request body framing"
    await JSONResponse(status_code=status, content={"detail": detail})(scope, receive, send)
