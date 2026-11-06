"""The kc-corpus namespace bootstrap refuses every process outside the dedicated profile."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "bootstrap_kc_corpus_namespace",
    Path(__file__).resolve().parent.parent / "scripts" / "bootstrap_kc_corpus_namespace.py",
)
bootstrap = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bootstrap)


def profile(**changes):
    values = {
        "database_url": "postgresql://kc_corpus_scrutator_app:private@100.97.136.74:5432/kc_corpus",
        "feeder_namespaces": "kc-corpus",
        "auth_enforce": True,
        "evidence_exact_bytes": True,
        "evidence_exact_namespaces": ["kc-corpus"],
        "evidence_exact_required_namespaces": ["kc-corpus"],
        "auth_arcana_introspect_url": "",
        "project_write_namespace": "",
    }
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "change",
    [
        {"database_url": "postgresql://scrutator:private@localhost:5432/arcanada_scrutator"},
        {"database_url": "postgresql://kc_scrutator_app:private@100.97.136.74:5432/knowledge_contract"},
        {"feeder_namespaces": "kc-corpus,shared"},
        {"auth_enforce": False},
        {"evidence_exact_bytes": False},
        {"evidence_exact_namespaces": None},
        {"evidence_exact_required_namespaces": ["kc2-store"]},
        {"auth_arcana_introspect_url": "http://x"},
        {"project_write_namespace": "kc-corpus"},
    ],
)
def test_foreign_or_loosened_profile_is_refused(change):
    with pytest.raises(bootstrap.Refusal):
        bootstrap.validate_profile(profile(**change))


@pytest.mark.asyncio
async def test_dry_run_creates_nothing(capsys):
    repository = SimpleNamespace(get_namespace_id=AsyncMock(return_value=None), upsert_namespace=AsyncMock())
    assert await bootstrap.run(False, profile(), repository) == 0
    repository.upsert_namespace.assert_not_awaited()
    assert "PREVIEW" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_live_creates_exactly_the_fixed_namespace_and_reads_it_back(capsys):
    repository = SimpleNamespace(
        get_namespace_id=AsyncMock(side_effect=[None, 7]), upsert_namespace=AsyncMock(return_value=7)
    )
    assert await bootstrap.run(True, profile(), repository) == 0
    repository.upsert_namespace.assert_awaited_once_with("kc-corpus")
    assert "state=created namespace=kc-corpus id=7" in capsys.readouterr().out
