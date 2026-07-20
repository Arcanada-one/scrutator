"""Safety-contract tests for the bounded LTM-0014 provenance repair."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scrutator.config import settings
from scrutator.tools.ltm_provenance_repair import (
    ApprovalError,
    ChunkRecord,
    EntityRecord,
    EntitySourceRecord,
    ExtractedEntity,
    PlanError,
    Repair,
    RepairPlan,
    Snapshot,
    apply_plan,
    atomic_write_json,
    build_repair_decisions,
    build_snapshot,
    contains_whole_phrase,
    load_plan,
    prepare_plan,
    require_approval,
    rollback_plan,
    validate_plan,
    validated_run_dir,
)

CHUNK_A = "00000000-0000-0000-0000-00000000000a"
CHUNK_B = "00000000-0000-0000-0000-00000000000b"
ENTITY_A = "10000000-0000-0000-0000-00000000000a"
ENTITY_B = "10000000-0000-0000-0000-00000000000b"
ENTITY_C = "10000000-0000-0000-0000-00000000000c"


def _snapshot() -> Snapshot:
    return build_snapshot(
        namespace="ltm-bench-datarim-kb",
        namespace_id=7,
        chunks=[
            ChunkRecord(CHUNK_A, "a.md", 0, "a" * 64, "Alpha   Project ships SRCH-0025."),
            ChunkRecord(CHUNK_B, "b.md", 0, "b" * 64, "Alpha Project is referenced again."),
        ],
        entities=[
            EntityRecord(ENTITY_A, "Alpha Project", "project", None),
            EntityRecord(ENTITY_B, "SRCH-0025", "task", None),
            EntityRecord(ENTITY_C, "Healthy", "project", CHUNK_B),
        ],
        entity_sources=[],
    )


@pytest.mark.parametrize(
    ("content", "phrase", "expected"),
    [
        ("Alpha\n  PROJECT ships", "Alpha Project", True),
        ("Tracking SRCH-0025, now.", "SRCH-0025", True),
        ("Arcana", "Arc", False),
        ("preSRCH-0025post", "SRCH-0025", False),
    ],
)
def test_contains_whole_phrase_uses_normalized_unicode_boundaries(content, phrase, expected):
    assert contains_whole_phrase(content, phrase) is expected


def test_build_repair_decisions_requires_exact_name_type_and_one_evidence_chunk():
    snapshot = _snapshot()
    extracted = {
        CHUNK_A: [
            ExtractedEntity("Alpha Project", "project"),
            ExtractedEntity("SRCH-0025", "task"),
            ExtractedEntity("Healthy", "project"),
        ],
        CHUNK_B: [
            ExtractedEntity("Alpha Project", "project"),
            ExtractedEntity("SRCH-0025", "wrong-type"),
        ],
    }

    decisions = build_repair_decisions(snapshot, extracted)

    assert [item.entity_id for item in decisions.repairs] == [ENTITY_B]
    assert decisions.repairs[0].chunk_id == CHUNK_A
    assert decisions.ambiguous_entity_ids == [ENTITY_A]
    assert ENTITY_C not in decisions.absent_entity_ids


def test_build_repair_decisions_keeps_extracted_name_absent_without_literal_evidence():
    snapshot = _snapshot()
    extracted = {CHUNK_A: [ExtractedEntity("Ghost", "project")]}
    ghost = EntityRecord("10000000-0000-0000-0000-00000000000d", "Ghost", "project", None)
    snapshot = build_snapshot(
        namespace=snapshot.namespace,
        namespace_id=snapshot.namespace_id,
        chunks=snapshot.chunks,
        entities=[*snapshot.entities, ghost],
        entity_sources=[],
    )

    decisions = build_repair_decisions(snapshot, extracted)

    assert ghost.id in decisions.absent_entity_ids


def test_approval_is_environment_only_and_exact(monkeypatch):
    monkeypatch.delenv("LTM0014_APPLY_GO", raising=False)
    with pytest.raises(ApprovalError, match="LTM-0014"):
        require_approval("LTM0014_APPLY_GO", "expected-digest")

    monkeypatch.setenv("LTM0014_APPLY_GO", "wrong")
    with pytest.raises(ApprovalError, match="LTM-0014"):
        require_approval("LTM0014_APPLY_GO", "expected-digest")

    monkeypatch.setenv("LTM0014_APPLY_GO", "expected-digest")
    require_approval("LTM0014_APPLY_GO", "expected-digest")


def test_atomic_write_json_is_private_and_replaces_existing_file(tmp_path: Path):
    target = tmp_path / "plan.json"
    atomic_write_json(target, {"value": 1})
    atomic_write_json(target, {"value": 2})

    assert json.loads(target.read_text()) == {"value": 2}
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("*.tmp"))


def test_validate_plan_detects_any_digest_bound_plan_edit():
    plan = RepairPlan.from_snapshot(_snapshot(), run_id="run-001", repairs=[])
    encoded = plan.to_dict()
    encoded["namespace"] = "other"

    with pytest.raises(PlanError, match="digest"):
        validate_plan(encoded)


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", "", " leading"])
def test_validated_run_dir_rejects_path_traversal(tmp_path: Path, run_id: str):
    with pytest.raises(PlanError, match="run-id"):
        validated_run_dir(tmp_path, run_id)


class _Transaction:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        self.conn.events.append("transaction-enter")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.conn.events.append("transaction-rollback" if exc else "transaction-commit")


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        self.conn.events.append("acquire")
        return self.conn

    async def __aexit__(self, *_args):
        self.conn.events.append("release")


class _Pool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _Acquire(self.conn)


class _Connection:
    def __init__(self, snapshot: Snapshot):
        self.snapshot = snapshot
        self.events: list[str] = []
        self.sql: list[str] = []

    def transaction(self, **kwargs):
        assert kwargs == {"isolation": "serializable"}
        return _Transaction(self)

    async def execute(self, sql, *args):
        compact = " ".join(sql.split())
        self.sql.append(compact)
        if "pg_advisory_lock" in compact:
            self.events.append("session-lock")
        if "pg_advisory_unlock" in compact:
            self.events.append("session-unlock")
        if "SET source_chunk_id = NULL" in compact:
            assert "source_chunk_id = $3::uuid" in compact
            return "UPDATE 1"
        if "UPDATE entities" in compact:
            assert "source_chunk_id IS NULL" in compact
            return "UPDATE 1"
        if "INSERT INTO entity_sources" in compact:
            assert "ON CONFLICT" not in compact
            return "INSERT 0 1"
        if "DELETE FROM entity_sources" in compact:
            assert "content_hash" in compact and "source_chunk_id" in compact
            return "DELETE 1"
        return "SELECT 1"

    async def fetchrow(self, sql, *args):
        compact = " ".join(sql.split())
        self.sql.append(compact)
        if "FROM namespaces" in compact:
            return {"id": self.snapshot.namespace_id}
        raise AssertionError(f"unexpected fetchrow: {compact}")

    async def fetch(self, sql, *args):
        compact = " ".join(sql.split())
        self.sql.append(compact)
        if "FROM chunks" in compact:
            return [item.to_row() for item in self.snapshot.chunks]
        if "FROM entities" in compact:
            return [item.to_row() for item in self.snapshot.entities]
        if "FROM entity_sources" in compact:
            return [item.to_row() for item in self.snapshot.entity_sources]
        raise AssertionError(f"unexpected fetch: {compact}")


class _PrepareConnection(_Connection):
    def transaction(self, **kwargs):
        assert kwargs == {"isolation": "repeatable_read", "readonly": True}
        return _Transaction(self)


def _repair_plan() -> RepairPlan:
    snapshot = _snapshot()
    decisions = build_repair_decisions(
        snapshot,
        {CHUNK_A: [ExtractedEntity("SRCH-0025", "task")]},
    )
    return RepairPlan.from_snapshot(snapshot, run_id="run-001", repairs=decisions.repairs)


@pytest.mark.asyncio
async def test_prepare_is_extraction_only_records_skips_and_replay_does_not_rebill(tmp_path: Path, monkeypatch):
    snapshot = _snapshot()
    conn = _PrepareConnection(snapshot)
    pool = _Pool(conn)
    monkeypatch.setenv("LTM0014_PREPARE_GO", "run-001")
    pipeline = MagicMock()
    pipeline.extract_entities = AsyncMock(
        side_effect=[
            [ExtractedEntity("Alpha Project", "project"), ExtractedEntity("SRCH-0025", "task")],
            [ExtractedEntity("Alpha Project", "project")],
        ]
    )

    with (
        patch("scrutator.tools.ltm_provenance_repair.LtmLlmClient"),
        patch("scrutator.tools.ltm_provenance_repair.IngestPipeline", return_value=pipeline),
    ):
        first = await prepare_plan(pool, snapshot.namespace, "run-001", tmp_path)
        second = await prepare_plan(pool, snapshot.namespace, "run-001", tmp_path)

    plan = load_plan(tmp_path / "run-001" / "plan.json")
    assert first == second
    assert plan.ambiguous_entity_ids == [ENTITY_A]
    assert plan.absent_entity_ids == []
    assert pipeline.extract_entities.await_count == 2
    assert not any(sql.startswith(("UPDATE ", "INSERT ", "DELETE ")) for sql in conn.sql)


@pytest.mark.asyncio
async def test_prepare_refuses_to_rebill_uncertain_inflight_chunk(tmp_path: Path, monkeypatch):
    snapshot = _snapshot()
    conn = _PrepareConnection(snapshot)
    pool = _Pool(conn)
    monkeypatch.setenv("LTM0014_PREPARE_GO", "run-crash")
    pipeline = MagicMock()
    pipeline.extract_entities = AsyncMock(side_effect=RuntimeError("connection lost after send"))

    with (
        patch("scrutator.tools.ltm_provenance_repair.LtmLlmClient"),
        patch("scrutator.tools.ltm_provenance_repair.IngestPipeline", return_value=pipeline),
    ):
        with pytest.raises(RuntimeError, match="connection lost"):
            await prepare_plan(pool, snapshot.namespace, "run-crash", tmp_path, max_chunks=1)
        with pytest.raises(PlanError, match="uncertain paid request"):
            await prepare_plan(pool, snapshot.namespace, "run-crash", tmp_path, max_chunks=1)

    assert pipeline.extract_entities.await_count == 1


@pytest.mark.asyncio
async def test_prepare_resume_rejects_extraction_configuration_drift(tmp_path: Path, monkeypatch):
    snapshot = _snapshot()
    conn = _PrepareConnection(snapshot)
    pool = _Pool(conn)
    monkeypatch.setenv("LTM0014_PREPARE_GO", "run-config")
    pipeline = MagicMock()
    pipeline.extract_entities = AsyncMock(return_value=[])

    with (
        patch("scrutator.tools.ltm_provenance_repair.LtmLlmClient"),
        patch("scrutator.tools.ltm_provenance_repair.IngestPipeline", return_value=pipeline),
    ):
        await prepare_plan(pool, snapshot.namespace, "run-config", tmp_path, max_chunks=1)
        monkeypatch.setattr(settings, "ltm_model", "different/model")
        with pytest.raises(PlanError, match="extraction configuration"):
            await prepare_plan(pool, snapshot.namespace, "run-config", tmp_path, max_chunks=1)

    assert pipeline.extract_entities.await_count == 1


@pytest.mark.asyncio
async def test_zero_chunk_prepare_still_writes_zero_cost_receipt(tmp_path: Path, monkeypatch):
    empty = build_snapshot("empty", 9, [], [], [])
    conn = _PrepareConnection(empty)
    monkeypatch.setenv("LTM0014_PREPARE_GO", "run-empty")

    with (
        patch("scrutator.tools.ltm_provenance_repair.LtmLlmClient"),
        patch("scrutator.tools.ltm_provenance_repair.IngestPipeline"),
    ):
        await prepare_plan(_Pool(conn), "empty", "run-empty", tmp_path)

    receipt = json.loads((tmp_path / "run-empty" / "cost-receipt.json").read_text())
    assert receipt["request_count"] == 0
    assert receipt["cost_usd"] == 0
    assert "request_ids" not in receipt


@pytest.mark.asyncio
async def test_apply_is_one_serializable_transaction_with_lock_and_null_only_cas(monkeypatch):
    plan = _repair_plan()
    conn = _Connection(_snapshot())
    monkeypatch.setenv("LTM0014_APPLY_GO", plan.plan_sha256)

    result = await apply_plan(_Pool(conn), plan)

    assert result == {"applied": 1, "already_applied": 0}
    assert conn.events == [
        "acquire",
        "session-lock",
        "transaction-enter",
        "transaction-commit",
        "session-unlock",
        "release",
    ]
    joined = "\n".join(conn.sql)
    assert "pg_advisory_lock" in joined
    assert "pg_advisory_unlock" in joined
    assert "UPDATE entities" in joined
    assert "INSERT INTO entity_sources" in joined
    assert "entity_edges" not in joined
    assert "entity_events" not in joined


@pytest.mark.asyncio
async def test_apply_rejects_snapshot_drift_before_mutation(monkeypatch):
    plan = _repair_plan()
    monkeypatch.setenv("LTM0014_APPLY_GO", plan.plan_sha256)
    changed = _snapshot()
    changed = build_snapshot(
        namespace=changed.namespace,
        namespace_id=changed.namespace_id,
        chunks=[*changed.chunks, ChunkRecord("00000000-0000-0000-0000-00000000000c", "c.md", 0, "c" * 64, "")],
        entities=changed.entities,
        entity_sources=[],
    )
    conn = _Connection(changed)

    with pytest.raises(PlanError, match="snapshot"):
        await apply_plan(_Pool(conn), plan)

    assert not any("UPDATE entities" in sql for sql in conn.sql)
    assert "transaction-rollback" in conn.events


@pytest.mark.asyncio
async def test_apply_rejects_repair_not_bound_to_exact_chunk_metadata(monkeypatch):
    snapshot = _snapshot()
    plan = RepairPlan.from_snapshot(
        snapshot,
        run_id="run-001",
        repairs=[Repair(ENTITY_B, "SRCH-0025", "task", CHUNK_A, "wrong.md", "a" * 64)],
    )
    conn = _Connection(snapshot)
    monkeypatch.setenv("LTM0014_APPLY_GO", plan.plan_sha256)

    with pytest.raises(PlanError, match="chunk metadata"):
        await apply_plan(_Pool(conn), plan)

    assert not any("UPDATE entities" in sql for sql in conn.sql)


@pytest.mark.asyncio
async def test_apply_rejects_duplicate_entity_repairs_before_mutation(monkeypatch):
    plan = _repair_plan()
    duplicated = RepairPlan.from_snapshot(_snapshot(), "run-001", [*plan.repairs, *plan.repairs])
    conn = _Connection(_snapshot())
    monkeypatch.setenv("LTM0014_APPLY_GO", duplicated.plan_sha256)

    with pytest.raises(PlanError, match="duplicate"):
        await apply_plan(_Pool(conn), duplicated)

    assert not any("UPDATE entities" in sql for sql in conn.sql)


@pytest.mark.asyncio
async def test_apply_api_rejects_missing_approval_before_pool_acquire(monkeypatch):
    plan = _repair_plan()
    conn = _Connection(_snapshot())
    monkeypatch.delenv("LTM0014_APPLY_GO", raising=False)

    with pytest.raises(ApprovalError, match="operator-gated"):
        await apply_plan(_Pool(conn), plan)

    assert conn.events == []


@pytest.mark.asyncio
async def test_apply_api_rejects_in_memory_plan_digest_drift(monkeypatch):
    plan = _repair_plan()
    plan.repairs.append(Repair(ENTITY_A, "Alpha Project", "project", CHUNK_A, "a.md", "a" * 64))
    conn = _Connection(_snapshot())
    monkeypatch.setenv("LTM0014_APPLY_GO", plan.plan_sha256)

    with pytest.raises(PlanError, match="plan digest"):
        await apply_plan(_Pool(conn), plan)

    assert conn.events == []


@pytest.mark.asyncio
async def test_rollback_is_cas_bound_to_exact_inserted_provenance(monkeypatch):
    plan = _repair_plan()
    monkeypatch.setenv("LTM0014_ROLLBACK_GO", plan.plan_sha256)
    applied_snapshot = _snapshot()
    entities = [
        EntityRecord(item.id, item.name, item.entity_type, CHUNK_A if item.id == ENTITY_B else item.source_chunk_id)
        for item in applied_snapshot.entities
    ]
    applied_snapshot = build_snapshot(
        namespace=applied_snapshot.namespace,
        namespace_id=applied_snapshot.namespace_id,
        chunks=applied_snapshot.chunks,
        entities=entities,
        entity_sources=[
            EntitySourceRecord(
                entity_id=ENTITY_B,
                namespace_id=7,
                source_path="a.md",
                content_hash="a" * 64,
                source_chunk_id=CHUNK_A,
            )
        ],
    )
    conn = _Connection(applied_snapshot)

    result = await rollback_plan(_Pool(conn), plan)

    assert result == {"rolled_back": 1, "already_rolled_back": 0}
    joined = "\n".join(conn.sql)
    assert "DELETE FROM entity_sources" in joined
    assert "SET source_chunk_id = NULL" in joined
    assert "entity_edges" not in joined
    assert "entity_events" not in joined


@pytest.mark.asyncio
async def test_rollback_api_rejects_missing_approval_before_pool_acquire(monkeypatch):
    plan = _repair_plan()
    conn = _Connection(_snapshot())
    monkeypatch.delenv("LTM0014_ROLLBACK_GO", raising=False)

    with pytest.raises(ApprovalError, match="operator-gated"):
        await rollback_plan(_Pool(conn), plan)

    assert conn.events == []
