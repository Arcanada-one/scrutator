"""Durable access log: one structured record per request (who uses the knowledge base, which
namespaces, how).

Design (Muneral 4e0054f0):

* A pure-ASGI middleware writes ONE JSON line per request to ``<dir>/access.jsonl`` through a
  ``TimedRotatingFileHandler`` (midnight UTC, ``backups`` daily files kept, so at least 30 days).
  The default directory lies on the ``scrutator-logs`` named volume, which survives a container
  recreate and a Deploy.
* The middleware NEVER reads the ``Authorization`` header, any capability token header or the body.
  The identity in a record is the PRINCIPAL ID that the verifier returned or a fixed capability kind
  (``capability:feeder`` ...), reported by the dependency that authorised the request through
  ``note_identity``. A credential value therefore cannot reach the file by construction.
* Handlers and dependencies add what only they know (identity, mode, namespace, query) through the
  ``note_*`` helpers, which write into a ``ContextVar`` holding the current record and are no-ops
  outside a request.
* The log can never fail a request: every file error is swallowed and reported to the process log at
  most once a minute.
* ``GET /health`` is not recorded (the container health probe, 5,760 a day, carries no information).

Not attributable by design: a shared secret (one feeder token, one reader principal used by several
agents) is one identity here. Routes that act on the whole allowed set (fetch by id, stats, namespace
list, edges) name no namespace: they log ``namespaces: []`` and ``allowed_n``.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import logging.handlers
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ACCESS_FILE = "access.jsonl"
DEFAULT_BACKUPS = 35  # daily files; stated retention is at least 30 days
SKIPPED_PATHS = frozenset({"/health"})
_WARN_EVERY_SECONDS = 60.0

#: Fixed machine-capability identities (the header decides which one; the value is never recorded).
CAPABILITY_KINDS = (
    "capability:feeder",
    "capability:ltm-writer",
    "capability:rollback",
    "capability:operator-rollback",
    "capability:capability-projection",
)

_current: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "scrutator_access_record", default=None
)


# --- helpers called from dependencies and handlers (no-ops outside a request) ----------------------


def note_identity(identity: str, *, auth: str = "ok", mode: str | None = None) -> None:
    """Record who the request is. ``auth`` is ``ok`` | ``denied`` (a credential was presented and
    refused) | ``none`` (no credential). ``identity`` is a principal id or a capability kind, never a value."""
    record = _current.get()
    if record is None:
        return
    record["identity"] = identity
    record["auth"] = auth
    if mode is not None:
        record["mode"] = mode


def note_mode(mode: str) -> None:
    record = _current.get()
    if record is not None:
        record["mode"] = mode


def note_namespace(name: str | None) -> None:
    record = _current.get()
    if record is None or not name:
        return
    names = record["namespaces"]
    if name not in names and len(names) < 50:
        names.append(name)


def note_allowed(count: int) -> None:
    record = _current.get()
    if record is not None:
        record["allowed_n"] = count


def note_query(text: str | None) -> None:
    """Record the length and a short hash of a search query. The text itself is never stored."""
    record = _current.get()
    if record is None or text is None:
        return
    record["q_len"] = len(text)
    record["q_sha16"] = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


# --- writer ----------------------------------------------------------------------------------------


class _Writer:
    def __init__(self, directory: str, backups: int) -> None:
        self._logger: logging.Logger | None = None
        self._last_warning = 0.0
        try:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.TimedRotatingFileHandler(
                path / ACCESS_FILE, when="midnight", utc=True, backupCount=backups, encoding="utf-8", delay=True
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            writer_logger = logging.Logger(f"scrutator.access.{directory}")
            writer_logger.propagate = False
            writer_logger.addHandler(handler)
            self._logger = writer_logger
        except Exception as exc:  # the log must never stop the service
            self._warn(f"access log disabled: cannot open {directory}: {exc.__class__.__name__}")

    def _warn(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warning >= _WARN_EVERY_SECONDS:
            self._last_warning = now
            logger.warning(message)

    def write(self, record: dict[str, Any]) -> None:
        if self._logger is None:
            return
        try:
            self._logger.info(json.dumps(record, separators=(",", ":"), ensure_ascii=True))
        except Exception as exc:
            self._warn(f"access log write failed: {exc.__class__.__name__}")


class AccessLogMiddleware:
    """Pure ASGI middleware: see the module docstring."""

    def __init__(self, app: Any, *, directory: str | None, backups: int = DEFAULT_BACKUPS) -> None:
        self.app = app
        self._writer = _Writer(directory, backups) if directory else None

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        # A nested instance (an outer one already owns the record) passes through, so one request is one record.
        if (
            self._writer is None
            or scope.get("type") != "http"
            or scope.get("path") in SKIPPED_PATHS
            or _current.get() is not None
        ):
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "identity": "none",
            "auth": "none",
            "mode": "unknown",
            "src": client[0] if client else None,
            "method": scope.get("method"),
            "route": None,
            "namespaces": [],
            "allowed_n": None,
            "status": 500,
            "ms": 0,
            "qs_len": len(scope.get("query_string") or b""),
        }
        status = {"value": 500}

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                status["value"] = int(message["status"])
            await send(message)

        started = time.monotonic()
        token = _current.set(record)
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            _current.reset(token)
            try:
                route = scope.get("route")
                record["route"] = getattr(route, "path", None) or "unmatched"
                record["status"] = status["value"]
                record["ms"] = int((time.monotonic() - started) * 1000)
                if record["mode"] == "unknown" and record["method"] in {"GET", "HEAD", "OPTIONS"}:
                    record["mode"] = "read"
                self._writer.write(record)
            except Exception as exc:
                self._writer._warn(f"access log record failed: {exc.__class__.__name__}")
