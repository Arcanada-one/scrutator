"""SRCH-0026 / LTM-0022 — Scrutator adapter for the Datarim known-fix recall seam.

`/dr-do` Step 7.4 (Datarim framework) runs `dev-tools/known-fix-memory.py query`, which
searches project-local `known_fix` records and — when `DATARIM_KNOWN_FIX_RETRIEVER` names an
absolute, regular, executable file — shells out to that retriever for *remote* evidence. The
framework command deliberately owns no adapter ("a project-specific remote adapter obtains its
own read credential through that project's documented mechanism; this framework command neither
creates nor broadens credentials"), so the Scrutator-backed adapter lives here.

This module is that adapter. It is the READ half of the self-learning loop: a prior task's
distilled conclusion, indexed into the KB, reaches the next task's context automatically.

Caller contract (measured against `known-fix-memory.py::run_bounded`, not assumed):

- argv is exactly ``[<exe>, "--query", <q>, "--limit", <n>]``
- **the child environment is stripped to ``PATH`` only** — measured child env on arcana-devs was
  ``['LC_CTYPE', 'PATH']``. No ``SCRUTATOR_*`` variable, no ``HOME``, no ``PYTHONPATH`` survives,
  so configuration is read from a FILE and this module imports **stdlib only**. (``Path.home()``
  still resolves via the passwd database with ``HOME`` unset — verified.)
- stdout must be a JSON **list**; every element is kept only if it has string ``citation`` and
  ``excerpt`` keys. The caller truncates them to 500 / 2000 characters.
- hard budget: exit within **3 seconds**, at most **64 KiB** on stdout, exit status **0**.
  Any breach is read by the caller as ``remote_status="unavailable"``.
- stderr is discarded by the caller (``subprocess.DEVNULL``).

Fail-soft is therefore total: every error path prints ``[]`` and exits 0. A KB that is
unreachable, unauthorized, empty, or misconfigured degrades the loop to local-only recall — it
never fails a task.

Security posture (the consilium's preconditions for closing the loop, ARAS-0058 §3):

- **Untrusted data, never instructions.** Retrieved KB text may be attacker-authored (external
  issue bodies and support prose are quoted verbatim into archives, which this lane indexes).
  Excerpts are dropped when the server's ingest-time injection signal is set (ARAS-0055
  ``metadata.injection``) AND re-scanned locally, because a missing/legacy stamp must not read
  as "clean". Surviving text is neutralised: control characters stripped, fence runs defanged so
  an excerpt cannot break out of the consumer's data fence.
- **A forgetting primitive.** A quarantine file lists ``content_hash`` / ``chunk_id`` values that
  must never be recalled again. It is consulted on every hit, so a poisoned or secret-bearing
  chunk can be retired without a re-index.
- **Credential shapes are dropped, not redacted.** `.gitignore` is a KB *inclusion* path, so a
  credential literal can reach the index despite every git-shaped scanner; a hit that looks like
  one is discarded rather than partially masked.

Usage::

    python -m scrutator.tools.known_fix_retriever --query "embedding times out" --limit 5

Configuration (JSON), resolved in order: ``--config`` → ``SCRUTATOR_KNOWN_FIX_CONFIG`` →
``~/.config/scrutator/known-fix-retriever.json``::

    {
      "base_url": "http://100.70.137.104:8310",
      "namespace": "self-improvement",
      "token_file": "/run/credentials/known-fix-retriever/token",
      "timeout_seconds": 2.0,
      "quarantine_file": "~/.config/scrutator/known-fix-quarantine.txt"
    }
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = "~/.config/scrutator/known-fix-retriever.json"
DEFAULT_NAMESPACE = "self-improvement"
DEFAULT_TIMEOUT_SECONDS = 2.0

# Stay clear of the caller's 3 s kill and 64 KiB read cap with room for TLS/DNS jitter.
MAX_TIMEOUT_SECONDS = 2.5
MAX_STDOUT_BYTES = 48 * 1024
MAX_LIMIT = 5
MAX_QUERY_CHARS = 500
MAX_CITATION_CHARS = 500
MAX_EXCERPT_CHARS = 2000
MAX_CONFIG_BYTES = 64 * 1024
MAX_TOKEN_BYTES = 8 * 1024
MAX_QUARANTINE_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024

# Chat-template role markers used to smuggle a fake turn into indexed content. Mirrors
# `search.ingest_safety._ROLE_MARKERS`; duplicated (not imported) because this module must stay
# importable with a stripped environment and no package dependencies.
_ROLE_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "[inst]",
    "[/inst]",
    "<<sys>>",
    "<</sys>>",
    "<start_of_turn>",
    "<end_of_turn>",
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
)

_OVERRIDE_RE = re.compile(
    r"ignore\s+(?:all\s+|the\s+|any\s+)?(?:previous|prior|above|earlier)\s+instructions"
    r"|disregard\s+(?:all\s+|the\s+)?(?:previous|prior|above)\s+(?:instructions|context|text)"
    r"|forget\s+(?:all\s+|everything\s+)?(?:previous|prior|above|you\s+were\s+told)"
    r"|you\s+are\s+now\s+(?:a|an|the)\b"
    r"|new\s+(?:system\s+)?(?:instructions|prompt)\s*:",
    re.IGNORECASE,
)

# Credential shapes. Mirrors `known-fix-memory.py::SECRET_RES` so a hit rejected by the framework
# validator on the write side is also rejected on the read side.
_SECRET_RES = (
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
    re.compile(r"hvs\.[A-Za-z0-9_-]{20,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9_./+=-]{16,}"),
    re.compile(r"[a-z][a-z0-9+.-]*://[^\s/:]+:[^\s/@]+@"),
    re.compile(r"(?i)(?:api[_-]?key|password|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,}"),
)

_FENCE_RE = re.compile(r"[`~]{3,}")
_HASH_RE = re.compile(r"[0-9a-f]{16,64}")


class RetrieverError(Exception):
    """Any condition that must degrade to an empty, exit-0 result."""


# ── configuration ────────────────────────────────────────────────────


def _read_small_file(path: Path, max_bytes: int) -> str:
    """Read a regular, non-symlink, bounded file. Raises RetrieverError otherwise."""
    try:
        info = path.lstat()
    except OSError as exc:
        raise RetrieverError(f"unreadable path: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RetrieverError(f"not a regular non-symlink file: {path}")
    if info.st_size > max_bytes:
        raise RetrieverError(f"file exceeds {max_bytes} bytes: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RetrieverError(f"file is not readable UTF-8: {path}") from exc


def resolve_config_path(explicit: str | None) -> Path:
    """`--config` wins, then SCRUTATOR_KNOWN_FIX_CONFIG, then the fixed default.

    The environment variable is a convenience for tests and interactive use only: under the real
    `/dr-do` invocation the environment carries PATH alone, so the default path is what resolves.
    """
    candidate = explicit or os.environ.get("SCRUTATOR_KNOWN_FIX_CONFIG") or DEFAULT_CONFIG_PATH
    return Path(candidate).expanduser()


def load_config(path: Path) -> dict[str, Any]:
    """Load and bound-check the adapter configuration."""
    try:
        value = json.loads(_read_small_file(path, MAX_CONFIG_BYTES))
    except json.JSONDecodeError as exc:
        raise RetrieverError(f"config is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RetrieverError("config must be a JSON object")
    base_url = value.get("base_url")
    if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
        raise RetrieverError("config.base_url must be an http(s) URL")
    timeout = value.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        timeout = DEFAULT_TIMEOUT_SECONDS
    namespace = value.get("namespace", DEFAULT_NAMESPACE)
    if not isinstance(namespace, str) or not namespace:
        namespace = DEFAULT_NAMESPACE
    return {
        "base_url": base_url.rstrip("/"),
        "namespace": namespace,
        "timeout_seconds": min(float(timeout), MAX_TIMEOUT_SECONDS),
        "token_file": value.get("token_file"),
        "quarantine_file": value.get("quarantine_file"),
    }


def load_token(config: dict[str, Any]) -> str | None:
    """Read the bearer token from its file. Absent or unreadable ⇒ unauthenticated attempt."""
    token_file = config.get("token_file")
    if not isinstance(token_file, str) or not token_file:
        return None
    try:
        token = _read_small_file(Path(token_file).expanduser(), MAX_TOKEN_BYTES).strip()
    except RetrieverError:
        return None
    # A token containing header-injection bytes is treated as absent, never as a header value.
    if not token or any(char in token for char in "\r\n\0"):
        return None
    return token


def load_quarantine(config: dict[str, Any]) -> frozenset[str]:
    """Load the forgetting primitive: content hashes / chunk ids never to be recalled.

    Line-oriented; `#` comments and blank lines ignored. An unreadable or absent file is an
    EMPTY quarantine, not an error — the loop stays available. Operators who need
    fail-closed quarantine should gate the credential instead.
    """
    path = config.get("quarantine_file")
    if not isinstance(path, str) or not path:
        return frozenset()
    try:
        text = _read_small_file(Path(path).expanduser(), MAX_QUARANTINE_BYTES)
    except RetrieverError:
        return frozenset()
    entries = set()
    for line in text.splitlines():
        entry = line.split("#", 1)[0].strip().lower()
        if entry:
            entries.add(entry)
    return frozenset(entries)


# ── transport ────────────────────────────────────────────────────────


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect.

    The scheme allowlist in `load_config` only covers the URL we construct; urllib would happily
    follow a 3xx onward, and its default handler permits `ftp:` as a redirect target. The KB
    never legitimately redirects a `/v1/search` POST, so a redirect is an error, not a hop.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ARG002
        return None


def _no_redirect_opener() -> urllib.request.OpenerDirector:
    """A fresh opener per call — no global state, no cookie or auth handler installed."""
    return urllib.request.build_opener(_NoRedirect)


def search(config: dict[str, Any], token: str | None, query: str, limit: int) -> list[dict[str, Any]]:
    """POST /v1/search and return the raw hit list. Any failure raises RetrieverError."""
    payload = json.dumps(
        {
            "query": query,
            "namespace": config["namespace"],
            "limit": limit,
            "include_content": True,
        }
    ).encode("utf-8")
    # `base_url` is validated to http/https in load_config, and `_no_redirect_opener()` refuses
    # every redirect, so neither a crafted config nor a 3xx can steer this at file:/ or ftp:.
    request = urllib.request.Request(
        f"{config['base_url']}/v1/search",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with _no_redirect_opener().open(request, timeout=config["timeout_seconds"]) as response:  # nosec B310
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # Covers HTTPError (401/403/5xx), connection refused, DNS failure and timeout alike.
        raise RetrieverError(f"search transport failed: {type(exc).__name__}") from exc
    if len(body) > MAX_RESPONSE_BYTES:
        raise RetrieverError("search response exceeds the response cap")
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetrieverError("search response is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise RetrieverError("search response is not an object")
    results = parsed.get("results")
    if not isinstance(results, list):
        raise RetrieverError("search response has no results list")
    return [item for item in results if isinstance(item, dict)]


# ── safety pipeline ──────────────────────────────────────────────────


def looks_injected(text: str) -> bool:
    """Local re-scan for instruction-shaped content.

    Deliberately independent of the server's `metadata.injection` stamp: a legacy or
    un-backfilled chunk carries no stamp, and "unstamped" must not read as "clean".
    """
    lowered = text.lower()
    if any(marker in lowered for marker in _ROLE_MARKERS):
        return True
    return bool(_OVERRIDE_RE.search(text))


def looks_secret(text: str) -> bool:
    """Whether the text carries a credential shape. Such hits are dropped, never masked."""
    return any(pattern.search(text) for pattern in _SECRET_RES)


def neutralise(text: str, max_chars: int) -> str:
    """Render untrusted KB text safe to embed in a delimited data block.

    Strips control characters, defangs fence runs so the excerpt cannot terminate the
    consumer's code/data fence, collapses whitespace runs, and truncates.
    """
    cleaned = "".join(char if char >= " " or char in "\t\n" else " " for char in text)
    cleaned = _FENCE_RE.sub("<fence>", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 1].rstrip() + "…"
    return cleaned


def hit_identifiers(hit: dict[str, Any]) -> set[str]:
    """Lowercased identity keys a quarantine entry may name for this hit."""
    identifiers = set()
    for key in ("content_hash", "chunk_id", "source_id"):
        value = hit.get(key)
        if isinstance(value, str) and value.strip():
            identifiers.add(value.strip().lower())
    return identifiers


def is_quarantined(hit: dict[str, Any], quarantine: frozenset[str]) -> bool:
    """The forgetting primitive: a listed content_hash / chunk_id is never recalled."""
    return bool(quarantine & hit_identifiers(hit))


def build_citation(hit: dict[str, Any], namespace: str) -> str:
    """A stable, auditable pointer back to the source chunk."""
    source_path = hit.get("source_path")
    source_path = source_path if isinstance(source_path, str) else "unknown"
    chunk_index = hit.get("chunk_index")
    chunk_index = chunk_index if isinstance(chunk_index, int) else 0
    citation = f"kb://{namespace}/{source_path}#chunk{chunk_index}"
    content_hash = hit.get("content_hash")
    if isinstance(content_hash, str) and _HASH_RE.fullmatch(content_hash):
        citation = f"{citation}@{content_hash[:16]}"
    return neutralise(citation, MAX_CITATION_CHARS)


def project_hit(hit: dict[str, Any], namespace: str, quarantine: frozenset[str]) -> dict[str, str] | None:
    """Turn one search hit into a contract item, or None if any gate rejects it."""
    if is_quarantined(hit, quarantine):
        return None
    injection = hit.get("injection")
    if isinstance(injection, dict) and injection.get("flag") is True:
        return None
    content = hit.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    if looks_injected(content) or looks_secret(content):
        return None
    excerpt = neutralise(content, MAX_EXCERPT_CHARS)
    if not excerpt:
        return None
    citation = build_citation(hit, namespace)
    if looks_secret(citation):
        return None
    return {"citation": citation, "excerpt": excerpt}


def bound_output(items: list[dict[str, str]]) -> str:
    """Serialize, dropping trailing items until the payload fits the caller's read cap."""
    kept = list(items)
    while kept:
        encoded = json.dumps(kept, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) <= MAX_STDOUT_BYTES:
            return encoded
        kept.pop()
    return "[]"


# ── entry point ──────────────────────────────────────────────────────


def collect(query: str, limit: int, config_path: Path) -> list[dict[str, str]]:
    """Run the full retrieval + safety pipeline. Raises RetrieverError on any failure."""
    config = load_config(config_path)
    token = load_token(config)
    quarantine = load_quarantine(config)
    hits = search(config, token, query, limit)
    items = []
    for hit in hits:
        item = project_hit(hit, config["namespace"], quarantine)
        if item is not None:
            items.append(item)
        if len(items) == limit:
            break
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrutator adapter for Datarim known-fix recall.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=MAX_LIMIT)
    parser.add_argument("--config", default=None, help="path to the adapter config JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Always prints a JSON list and always returns 0 — the caller's fail-soft contract."""
    try:
        args = build_parser().parse_args(argv)
    except SystemExit:
        # argparse would exit 2 and print usage; the contract wants an empty, quiet success.
        print("[]")
        return 0
    query = args.query
    if any(ord(char) < 32 for char in query) or not 1 <= len(query) <= MAX_QUERY_CHARS:
        print("[]")
        return 0
    limit = min(max(args.limit, 1), MAX_LIMIT) if isinstance(args.limit, int) else MAX_LIMIT
    try:
        items = collect(query, limit, resolve_config_path(args.config))
    except RetrieverError as exc:
        print(f"known-fix retriever unavailable: {exc}", file=sys.stderr)
        print("[]")
        return 0
    print(bound_output(items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
