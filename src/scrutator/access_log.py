"""Durable access log: one structured record per request (who uses the knowledge base, which
namespaces, how).

Design (Muneral 4e0054f0):

* A pure-ASGI middleware writes ONE JSON line per request to ``<dir>/access.jsonl`` through a
  ``TimedRotatingFileHandler`` (midnight UTC, ``backups`` daily files kept, so at least 30 days).
  The default directory lies on the ``scrutator-logs`` named volume, which survives a container
  recreate and a Deploy.
* The middleware reads exactly THREE request headers and no others: ``User-Agent`` (``ua``, 80
  characters, control characters and non-ASCII replaced), and ``CF-Connecting-IP`` / ``X-Forwarded-For``
  (``fwd``). It NEVER reads ``Authorization``, any capability token header (``X-*-Token``) or the body.
  The identity in a record is the PRINCIPAL ID that the verifier returned or a fixed capability kind
  (``capability:feeder`` ...), reported by the dependency that authorised the request through
  ``note_identity``. A credential value therefore cannot reach the file by construction.
* ``fwd`` is the client address a proxy REPORTS, not a verified one: it is recorded only when the direct
  peer (``src``) is a loopback or private address (the proxy), validated with ``ipaddress``, otherwise null.
  ``src`` alone is 127.0.0.1 or the proxy for almost every request, so ``ua`` and ``fwd`` are what tells one
  agent or host from another when they share a token.
* SINGLE-PROCESS ASSUMPTION: the service runs one uvicorn process. ``TimedRotatingFileHandler`` rotation is
  not safe across processes, so the middleware logs this assumption at start and warns when
  ``WEB_CONCURRENCY`` / ``UVICORN_WORKERS`` is above 1.
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
import ipaddress
import json
import logging
import logging.handlers
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ACCESS_FILE = "access.jsonl"
DEFAULT_BACKUPS = 35  # daily files; stated retention is at least 30 days
SKIPPED_PATHS = frozenset({"/health"})
_WARN_EVERY_SECONDS = 60.0
UA_MAX = 80
_NOT_PRINTABLE_ASCII = re.compile(r"[^\x20-\x7e]")

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


def _header(scope: dict[str, Any], name: bytes) -> str | None:
    """One named request header, decoded leniently. Only ``user-agent``, ``cf-connecting-ip`` and
    ``x-forwarded-for`` are ever requested (see the module docstring); never a credential header."""
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def client_agent(scope: dict[str, Any]) -> str | None:
    """User-Agent, first 80 characters, anything but printable ASCII replaced; None when absent or empty."""
    raw = _header(scope, b"user-agent")
    if not raw:
        return None
    return _NOT_PRINTABLE_ASCII.sub("?", raw)[:UA_MAX]


def reported_client(scope: dict[str, Any]) -> str | None:
    """The client address the proxy reports (CF-Connecting-IP, else the first hop of X-Forwarded-For).

    Recorded ONLY when the direct peer is loopback or a private address, i.e. the proxy itself, so that a
    remote caller cannot forge it from outside; validated with ``ipaddress``, otherwise None. Reported by
    the proxy, not verified.
    """
    client = scope.get("client")
    try:
        peer = ipaddress.ip_address(client[0]) if client else None
    except ValueError:
        return None
    if peer is None or not (peer.is_loopback or peer.is_private):
        return None
    candidate = _header(scope, b"cf-connecting-ip")
    if not candidate:
        forwarded = _header(scope, b"x-forwarded-for")
        candidate = forwarded.split(",")[0] if forwarded else None
    if not candidate:
        return None
    try:
        return str(ipaddress.ip_address(candidate.strip()))
    except ValueError:
        return None


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
        if self._writer is not None:
            workers = max(
                (
                    int(os.environ.get(name, "1"))
                    for name in ("WEB_CONCURRENCY", "UVICORN_WORKERS")
                    if os.environ.get(name, "").isdigit()
                ),
                default=1,
            )
            logger.info(
                "access log on: %s, single-process assumption (daily rotation is not multi-process safe)", directory
            )
            if workers > 1:
                logger.warning(
                    "access log: %d workers configured, rotation assumes ONE process; records may be lost",
                    workers,
                )

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
            "fwd": reported_client(scope),
            "ua": client_agent(scope),
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
                # `mode` is stamped by the dependency that authorised the request: a machine capability stamps
                # "write" (it is a write credential), but a GET/HEAD/OPTIONS made with it changes nothing, and a
                # report that counted it as a write would overstate the writers. Only the LOGGED field follows the
                # verb here; what the capability may do is untouched.
                if record["method"] in {"GET", "HEAD", "OPTIONS"} and record["mode"] in {"unknown", "write"}:
                    record["mode"] = "read"
                self._writer.write(record)
            except Exception as exc:
                self._writer._warn(f"access log record failed: {exc.__class__.__name__}")
