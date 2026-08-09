"""Loopback-only receiver used by Capability Registry cross-repository CI."""

from __future__ import annotations

import json
import os
from pathlib import Path

import uvicorn

import scrutator.health as health
from scrutator.db.models import IndexResponse


def _readback_path() -> Path:
    value = os.environ.get("ARCA_0199_TEST_SCRUTATOR_READBACK_PATH", "")
    path = Path(value)
    if not path.is_absolute() or not path.parent.is_dir():
        raise RuntimeError("ARCA_0199_TEST_SCRUTATOR_READBACK_PATH must have an existing absolute parent")
    return path


def _port() -> int:
    value = os.environ.get("ARCA_0199_TEST_SCRUTATOR_PORT", "")
    if not value.isascii() or not value.isdecimal():
        raise RuntimeError("ARCA_0199_TEST_SCRUTATOR_PORT must be a decimal port")
    port = int(value)
    if not 1 <= port <= 65_535:
        raise RuntimeError("ARCA_0199_TEST_SCRUTATOR_PORT is outside the valid port range")
    return port


async def capture_index_document(**kwargs) -> IndexResponse:
    """Capture the receiver-derived index call without an external embedding service."""
    path = _readback_path()
    payload = {
        "content": kwargs["content"],
        "max_tokens": kwargs["max_tokens"],
        "namespace": kwargs["namespace"],
        "overlap_tokens": kwargs["overlap_tokens"],
        "project": kwargs["project"],
        "source_path": kwargs["source_path"],
        "source_type": kwargs["source_type"],
    }
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)
    return IndexResponse(
        chunks_indexed=1,
        source_path=kwargs["source_path"],
        namespace=kwargs["namespace"],
        strategy_used="markdown",
    )


health.index_document = capture_index_document


if __name__ == "__main__":
    uvicorn.run(health.app, host="127.0.0.1", port=_port(), log_level="warning")
