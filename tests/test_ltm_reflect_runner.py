"""Tests for the LTM-0026 bounded reflect runner."""

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from scrutator.ltm.models import ReflectRunSummary
from scrutator.ltm.reflect_runner import ReflectCursor, ReflectRunnerError, run_reflect_once

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PREFLIGHT = REPO_ROOT / "deploy" / "ltm-reflect-state-preflight.sh"


class FakeReflectJob:
    summary = ReflectRunSummary(
        run_id="run-1",
        status="done",
        chunks_scanned=2,
        meta_facts_created=1,
        cost_usd=0.0,
        req_count=1,
        duration_ms=12.0,
    )
    calls: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def run(self, *, since=None, max_chunks=None, dry_run=False):
        self.__class__.calls.append({"since": since, "max_chunks": max_chunks, "dry_run": dry_run})
        return self.__class__.summary, [object()]


@pytest.fixture(autouse=True)
def reset_fake_job():
    FakeReflectJob.calls = []
    FakeReflectJob.summary = ReflectRunSummary(
        run_id="run-1",
        status="done",
        chunks_scanned=2,
        meta_facts_created=1,
        cost_usd=0.0,
        req_count=1,
        duration_ms=12.0,
    )


def test_compose_persists_reflect_cursor_in_exact_fail_closed_bind_mount():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    volumes = compose["services"]["scrutator"]["volumes"]
    reflect_mount = next(
        volume
        for volume in volumes
        if isinstance(volume, dict) and volume.get("target") == "/var/lib/scrutator/ltm-reflect"
    )

    assert reflect_mount == {
        "type": "bind",
        "source": "/var/lib/scrutator/ltm-reflect",
        "target": "/var/lib/scrutator/ltm-reflect",
        "read_only": False,
        "bind": {"create_host_path": False},
    }


def test_state_preflight_accepts_exact_directory_and_rejects_symlink(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o700)
    env = {
        **os.environ,
        "LTM_REFLECT_STATE_EXPECTED_UID": str(os.getuid()),
        "LTM_REFLECT_STATE_EXPECTED_GID": str(os.getgid()),
    }

    accepted = subprocess.run([STATE_PREFLIGHT, state_dir], env=env, capture_output=True, text=True)
    assert accepted.returncode == 0, accepted.stderr

    symlink = tmp_path / "state-link"
    symlink.symlink_to(state_dir, target_is_directory=True)
    rejected = subprocess.run([STATE_PREFLIGHT, symlink], env=env, capture_output=True, text=True)
    assert rejected.returncode != 0


def test_deploy_runs_state_preflight_before_compose():
    workflow = (REPO_ROOT / ".github" / "workflows" / "deploy.yml").read_text()
    deploy = workflow[workflow.index("      - name: Deploy") :]
    transaction = (REPO_ROOT / "deploy" / "scrutator-deploy-transaction.sh").read_text()
    candidate = transaction[transaction.index("deploy_candidate()") : transaction.index("rollback_transaction()")]
    script = (REPO_ROOT / "scripts" / "deploy.sh").read_text()

    assert "deploy/scrutator-deploy-transaction.sh" in deploy
    assert candidate.index("deploy/ltm-reflect-state-preflight.sh") < candidate.index("compose_project")
    assert script.index("deploy/ltm-reflect-state-preflight.sh") < script.index("docker compose up")


def test_cursor_round_trips_utc_datetime(tmp_path):
    path = tmp_path / "cursor.json"
    moment = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)

    ReflectCursor(last_completed_at=moment).save(path)

    assert ReflectCursor.load(path).last_completed_at == moment


async def test_successful_runner_updates_cursor(tmp_path):
    state = tmp_path / "cursor.json"
    with (
        patch("scrutator.ltm.reflect_runner.repository.get_namespace_id", AsyncMock(return_value=348)),
        patch("scrutator.ltm.reflect_runner.LtmLlmClient", lambda **_: object()),
        patch("scrutator.ltm.reflect_runner.ReflectJob", FakeReflectJob),
    ):
        result = await run_reflect_once(namespace="wiki", state_file=state, max_chunks=7)

    assert result["cursor_updated"] is True
    assert result["summary"]["status"] == "done"
    assert FakeReflectJob.calls == [{"since": None, "max_chunks": 7, "dry_run": False}]
    assert ReflectCursor.load(state).last_completed_at is not None


async def test_runner_uses_existing_cursor_as_since(tmp_path):
    state = tmp_path / "cursor.json"
    since = datetime(2026, 7, 16, 10, 30, tzinfo=UTC)
    ReflectCursor(last_completed_at=since).save(state)
    with (
        patch("scrutator.ltm.reflect_runner.repository.get_namespace_id", AsyncMock(return_value=348)),
        patch("scrutator.ltm.reflect_runner.LtmLlmClient", lambda **_: object()),
        patch("scrutator.ltm.reflect_runner.ReflectJob", FakeReflectJob),
    ):
        await run_reflect_once(namespace="wiki", state_file=state)

    assert FakeReflectJob.calls[0]["since"] == since


async def test_dry_run_does_not_update_cursor(tmp_path):
    state = tmp_path / "cursor.json"
    with (
        patch("scrutator.ltm.reflect_runner.repository.get_namespace_id", AsyncMock(return_value=348)),
        patch("scrutator.ltm.reflect_runner.LtmLlmClient", lambda **_: object()),
        patch("scrutator.ltm.reflect_runner.ReflectJob", FakeReflectJob),
    ):
        result = await run_reflect_once(namespace="wiki", state_file=state, dry_run=True)

    assert result["cursor_updated"] is False
    assert result["preview_count"] == 1
    assert not state.exists()


async def test_failed_summary_does_not_update_cursor(tmp_path):
    state = tmp_path / "cursor.json"
    FakeReflectJob.summary = ReflectRunSummary(
        run_id="run-1",
        status="failed",
        chunks_scanned=2,
        meta_facts_created=0,
        cost_usd=0.0,
        req_count=1,
        abort_reason="MC down",
        duration_ms=12.0,
    )
    with (
        patch("scrutator.ltm.reflect_runner.repository.get_namespace_id", AsyncMock(return_value=348)),
        patch("scrutator.ltm.reflect_runner.LtmLlmClient", lambda **_: object()),
        patch("scrutator.ltm.reflect_runner.ReflectJob", FakeReflectJob),
    ):
        result = await run_reflect_once(namespace="wiki", state_file=state)

    assert result["cursor_updated"] is False
    assert not state.exists()


async def test_unknown_namespace_fails_before_llm(tmp_path):
    with (
        patch("scrutator.ltm.reflect_runner.repository.get_namespace_id", AsyncMock(return_value=None)),
        pytest.raises(ReflectRunnerError, match="namespace not found"),
    ):
        await run_reflect_once(namespace="missing", state_file=tmp_path / "cursor.json")
