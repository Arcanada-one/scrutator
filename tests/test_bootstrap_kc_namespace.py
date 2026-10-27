"""Fixed project namespace bootstrap refuses shared or misconfigured instances."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "bootstrap_kc_namespace", Path(__file__).resolve().parent.parent / "scripts" / "bootstrap_kc_namespace.py"
)
bootstrap = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bootstrap)


def profile(**changes):
    values = {
        "database_url": "postgresql://kc_scrutator_app:private@100.97.136.74:5432/knowledge_contract",
        "auth_arcana_introspect_url": bootstrap.INTROSPECT_URL,
        "auth_service_audience": bootstrap.AUDIENCE,
        "auth_service_scope": "kb:knowledge-contract.read",
        "feeder_namespaces": "knowledge-contract",
        "auth_enforce": True,
    }
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "change",
    [
        {"database_url": "postgresql://scrutator:private@localhost:5432/arcanada_scrutator"},
        {"auth_arcana_introspect_url": ""},
        {"auth_service_audience": "urn:arcanada:scrutator:shared"},
        {"feeder_namespaces": "knowledge-contract,shared"},
        {"auth_enforce": False},
    ],
)
@pytest.mark.asyncio
async def test_refuses_any_non_project_profile_before_database_access(change):
    repo = SimpleNamespace(get_namespace_id=AsyncMock(), upsert_namespace=AsyncMock())
    with pytest.raises(bootstrap.Refusal):
        await bootstrap.run(True, profile(**change), repo)
    repo.get_namespace_id.assert_not_awaited()
    repo.upsert_namespace.assert_not_awaited()


@pytest.mark.asyncio
async def test_preview_is_read_only_when_project_namespace_is_absent(capsys):
    repo = SimpleNamespace(get_namespace_id=AsyncMock(return_value=None), upsert_namespace=AsyncMock())
    assert await bootstrap.run(False, profile(), repo) == 0
    repo.upsert_namespace.assert_not_awaited()
    assert "PREVIEW state=absent namespace=knowledge-contract" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_live_uses_repository_api_and_requires_exact_readback(capsys):
    repo = SimpleNamespace(
        get_namespace_id=AsyncMock(side_effect=[None, 7]),
        upsert_namespace=AsyncMock(return_value=7),
    )
    assert await bootstrap.run(True, profile(), repo) == 0
    repo.upsert_namespace.assert_awaited_once_with("knowledge-contract")
    assert "PASS state=created namespace=knowledge-contract id=7" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_existing_namespace_is_idempotent_without_mutation(capsys):
    repo = SimpleNamespace(get_namespace_id=AsyncMock(return_value=7), upsert_namespace=AsyncMock())
    assert await bootstrap.run(True, profile(), repo) == 0
    repo.upsert_namespace.assert_not_awaited()
    assert "PASS state=present namespace=knowledge-contract id=7" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_missing_readback_is_not_reported_as_success():
    repo = SimpleNamespace(
        get_namespace_id=AsyncMock(side_effect=[None, None]),
        upsert_namespace=AsyncMock(return_value=7),
    )
    with pytest.raises(RuntimeError, match="readback mismatch"):
        await bootstrap.run(True, profile(), repo)
