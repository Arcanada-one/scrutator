"""Safety-contract tests for the SRCH-0031 loopback measurement wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import granted_context_app
import pytest


class _Acquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, *_args):
        return None


class _Pool:
    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        return _Acquire(self.connection)


@pytest.mark.asyncio
async def test_build_context_uses_live_grants_and_requires_arcanada(monkeypatch):
    connection = AsyncMock()
    connection.fetch.return_value = [{"id": 7, "name": "arcanada"}]
    monkeypatch.setenv("SCRUTATOR_BENCHMARK_PRINCIPAL", "kb-observer")
    monkeypatch.setattr(
        granted_context_app,
        "resolve_allowed_namespaces",
        AsyncMock(return_value=frozenset({7})),
    )
    monkeypatch.setattr(granted_context_app, "get_pool", AsyncMock(return_value=_Pool(connection)))

    context = await granted_context_app.build_benchmark_context()

    assert context.principal_id == "kb-observer"
    assert context.allowed_namespace_ids == frozenset({7})
    assert context.allowed_namespace_names == frozenset({"arcanada"})
    connection.fetch.assert_awaited_once()

    connection.fetch.return_value = [{"id": 8, "name": "other"}]
    with pytest.raises(RuntimeError, match="not granted"):
        await granted_context_app.build_benchmark_context()


def test_shell_runner_is_loopback_lifespan_off_and_fail_closed():
    script = (Path(__file__).resolve().parent.parent / "live" / "run_rerank_gate.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "trap cleanup EXIT" in script
    assert "--network host" in script
    assert "--host 127.0.0.1" in script
    assert "--lifespan off" in script
    assert '"production_image": production_image' in script
    assert "requested image does not match production image" not in script
    assert "SCRUTATOR_RERANK_ENABLED=false" in script
    assert "SCRUTATOR_RERANK_ENABLED=true" in script
    assert "SCRUTATOR_DATABASE_POOL_MIN=1" in script
    assert "SCRUTATOR_DATABASE_POOL_MAX=2" in script
    for secret_name in (
        "SCRUTATOR_LTM_WRITER_TOKEN",
        "SCRUTATOR_FEEDER_TOKEN",
        "SCRUTATOR_ROLLBACK_TOKEN",
        "SCRUTATOR_OPERATOR_ROLLBACK_TOKEN",
    ):
        assert f"{secret_name}=" in script
    assert "0.0.0.0" not in script
    assert "StrictHostKeyChecking=no" not in script
