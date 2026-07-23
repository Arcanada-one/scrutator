"""LTM-0019 deterministic historical entity-provenance repair tests."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import scrutator.tools.ltm_entity_source_backfill as backfill
from scrutator.tools.ltm_entity_source_backfill import (
    ChunkRecord,
    DatabaseIdentity,
    EntityRecord,
    PlanError,
    RepairPlan,
    Snapshot,
    atomic_write_json,
    classify_snapshot,
    count_whole_phrase,
    derive_canary_plan,
    load_plan,
    validate_plan,
    validated_run_dir,
)


def _snapshot() -> Snapshot:
    chunks = [
        ChunkRecord(
            id="00000000-0000-0000-0000-000000000001",
            source_path="a.md",
            content_hash="a" * 64,
            content="Auth Arcana is live. Shared appears here and Shared appears twice.",
        ),
        ChunkRecord(
            id="00000000-0000-0000-0000-000000000002",
            source_path="b.md",
            content_hash="b" * 64,
            content="Shared also appears here. Other evidence.",
        ),
    ]
    entities = [
        EntityRecord(
            "10000000-0000-0000-0000-000000000001", "AUTH ARCANA", "project", None, "2026-01-01T00:00:00+00:00"
        ),
        EntityRecord(
            "10000000-0000-0000-0000-000000000002", "Other evidence", "fact", None, "2026-01-01T00:00:01+00:00"
        ),
        EntityRecord(
            "10000000-0000-0000-0000-000000000003", "Missing literal", "fact", None, "2026-01-01T00:00:02+00:00"
        ),
        EntityRecord("10000000-0000-0000-0000-000000000004", "Shared", "fact", None, "2026-01-01T00:00:03+00:00"),
        EntityRecord("10000000-0000-0000-0000-000000000005", "No evidence", "fact", None, "2026-01-01T00:00:04+00:00"),
        EntityRecord("10000000-0000-0000-0000-000000000006", "Preowned", "fact", None, "2026-01-01T00:00:05+00:00"),
    ]
    evidence = {
        entities[0].id: frozenset({chunks[0].id}),
        entities[1].id: frozenset({chunks[0].id, chunks[1].id}),
        entities[2].id: frozenset({chunks[0].id}),
        entities[3].id: frozenset({chunks[0].id}),
    }
    return Snapshot.build(
        database_identity=DatabaseIdentity(
            database="arcanada_scrutator",
            database_oid="46732",
            system_identifier="7648390441241305131",
            in_recovery=False,
        ),
        namespace="ltm-bench-datarim-kb",
        namespace_id=825,
        chunks=chunks,
        entities=entities,
        preowned_entity_ids=frozenset({entities[5].id}),
        evidence_by_entity=evidence,
    )


def test_nfkc_casefold_whole_phrase_count_is_boundary_aware():
    assert count_whole_phrase("Ａuth   Arcana and auth arcana", "Auth Arcana") == 2
    assert count_whole_phrase("NotAuth Arcana or Auth Arcanada", "Auth Arcana") == 0
    assert count_whole_phrase("Straße STRASSE", "strasse") == 2


def test_classifier_requires_structural_and_unique_literal_evidence():
    result = classify_snapshot(_snapshot())

    assert result.counts == {
        "eligible": 1,
        "ambiguous_structural": 1,
        "literal_absent": 1,
        "literal_nonunique": 1,
        "no_structural": 1,
        "preowned": 1,
        "invalid_metadata": 0,
    }
    assert len(result.repairs) == 1
    assert result.repairs[0].entity_name == "AUTH ARCANA"
    assert result.repairs[0].chunk_id == "00000000-0000-0000-0000-000000000001"


def test_classifier_rejects_foreign_or_dangling_structural_evidence():
    snapshot = _snapshot()
    evidence = dict(snapshot.evidence_by_entity)
    evidence[snapshot.entities[0].id] = frozenset(
        {*evidence[snapshot.entities[0].id], "00000000-0000-0000-0000-000000000099"}
    )
    unsafe = Snapshot.build(
        database_identity=snapshot.database_identity,
        namespace=snapshot.namespace,
        namespace_id=snapshot.namespace_id,
        chunks=snapshot.chunks,
        entities=snapshot.entities,
        preowned_entity_ids=snapshot.preowned_entity_ids,
        evidence_by_entity=evidence,
    )

    result = classify_snapshot(unsafe)

    assert result.counts["eligible"] == 0
    assert result.counts["ambiguous_structural"] == 2


def test_structural_evidence_query_preserves_foreign_and_dangling_chunk_ids():
    assert "JOIN chunks" not in backfill._STRUCTURAL_EVIDENCE_SQL
    assert "JOIN entities e" in backfill._STRUCTURAL_EVIDENCE_SQL
    assert "FROM entity_events ev\nJOIN entities e ON e.id = ev.entity_id" in backfill._STRUCTURAL_EVIDENCE_SQL


def test_plan_is_ltm0019_digest_bound_and_tamper_evident():
    snapshot = _snapshot()
    classification = classify_snapshot(snapshot)
    plan = RepairPlan.from_snapshot(snapshot, "run-001", classification)

    assert plan.task_id == "LTM-0019"
    assert plan.method == "structural_plus_unique_literal_v1"
    assert plan.database_identity == snapshot.database_identity
    assert plan.repairs[0].source_updated_at == plan.created_at
    assert validate_plan(plan.to_dict()) == plan

    encoded = plan.to_dict()
    encoded["repairs"][0]["chunk_id"] = "00000000-0000-0000-0000-000000000099"
    with pytest.raises(PlanError, match="digest"):
        validate_plan(encoded)


def test_plan_rejects_snapshot_or_original_timestamp_drift():
    snapshot = _snapshot()
    classification = classify_snapshot(snapshot)
    plan = RepairPlan.from_snapshot(snapshot, "run-001", classification)

    with pytest.raises(PlanError, match="snapshot"):
        plan.validate_against(replace(snapshot, digest="0" * 64))
    with pytest.raises(PlanError, match="snapshot"):
        plan.validate_against(
            replace(
                snapshot,
                database_identity=replace(snapshot.database_identity, database_oid="99999"),
            )
        )

    drifted = list(snapshot.entities)
    drifted[0] = replace(drifted[0], updated_at="2026-02-01T00:00:00+00:00")
    drifted_snapshot = Snapshot.build(
        database_identity=snapshot.database_identity,
        namespace=snapshot.namespace,
        namespace_id=snapshot.namespace_id,
        chunks=snapshot.chunks,
        entities=drifted,
        preowned_entity_ids=snapshot.preowned_entity_ids,
        evidence_by_entity=snapshot.evidence_by_entity,
    )
    with pytest.raises(PlanError, match="snapshot"):
        plan.validate_against(drifted_snapshot)


def test_plan_recomputes_exact_eligibility_and_rejects_other_namespace():
    snapshot = _snapshot()
    plan = RepairPlan.from_snapshot(snapshot, "run-001", classify_snapshot(snapshot))
    unsafe_repair = replace(
        plan.repairs[0],
        chunk_id=snapshot.chunks[1].id,
        source_path=snapshot.chunks[1].source_path,
        content_hash=snapshot.chunks[1].content_hash,
    )
    unsafe = replace(plan, repairs=[unsafe_repair], plan_sha256="")
    unsafe = replace(unsafe, plan_sha256=backfill._digest(unsafe.body()))

    assert validate_plan(unsafe.to_dict()) == unsafe
    with pytest.raises(PlanError, match="selection"):
        unsafe.validate_against(snapshot)

    other = replace(plan, namespace="other-namespace", namespace_id=999, plan_sha256="")
    other = replace(other, plan_sha256=backfill._digest(other.body()))
    with pytest.raises(PlanError, match="restricted"):
        validate_plan(other.to_dict())


def test_private_plan_files_and_run_ids_fail_closed(tmp_path):
    snapshot = _snapshot()
    plan = RepairPlan.from_snapshot(snapshot, "run-001", classify_snapshot(snapshot))
    plan_path = tmp_path / "private" / "plan.json"

    atomic_write_json(plan_path, plan.to_dict())
    assert plan_path.stat().st_mode & 0o777 == 0o600
    assert load_plan(plan_path) == plan

    plan_path.chmod(0o644)
    with pytest.raises(PlanError, match="non-private"):
        load_plan(plan_path)
    with pytest.raises(PlanError, match="safe path"):
        validated_run_dir(tmp_path, "../escape")

    symlink_parent = tmp_path / "symlink-parent"
    symlink_parent.symlink_to(plan_path.parent, target_is_directory=True)
    with pytest.raises(PlanError, match="symlink"):
        atomic_write_json(symlink_parent / "other.json", plan.to_dict())

    plan_path.chmod(0o600)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(os, "geteuid", lambda: plan_path.stat().st_uid + 1)
        with pytest.raises(PlanError, match="owned"):
            load_plan(plan_path)


def test_canary_plan_is_deterministic_and_bound_to_full_plan():
    snapshot = _snapshot()
    full_plan = RepairPlan.from_snapshot(snapshot, "run-001", classify_snapshot(snapshot))

    canary = derive_canary_plan(full_plan)

    assert canary.selection == "canary"
    assert canary.parent_plan_sha256 == full_plan.plan_sha256
    assert canary.run_id == "run-001-canary"
    assert len(canary.repairs) == 1
    assert canary == derive_canary_plan(full_plan)


def test_exact_applied_state_requires_plan_owned_timestamp():
    snapshot = _snapshot()
    plan = RepairPlan.from_snapshot(snapshot, "run-001", classify_snapshot(snapshot))
    repair = plan.repairs[0]
    entity = {
        "name": repair.entity_name,
        "entity_type": repair.entity_type,
        "source_chunk_id": repair.chunk_id,
        "updated_at": datetime.fromisoformat(repair.original_updated_at),
    }
    source = {
        "namespace_id": plan.namespace_id,
        "source_path": repair.source_path,
        "content_hash": repair.content_hash,
        "source_chunk_id": repair.chunk_id,
        "updated_at": datetime.fromisoformat(repair.source_updated_at),
    }

    assert backfill._state_is_exact_applied(entity, [source], plan, repair)
    source["updated_at"] = datetime.fromisoformat("2026-07-23T13:00:00+00:00")
    assert not backfill._state_is_exact_applied(entity, [source], plan, repair)


@pytest.mark.asyncio
async def test_postwrite_readback_rejects_multiple_source_rows():
    snapshot = _snapshot()
    plan = RepairPlan.from_snapshot(snapshot, "run-001", classify_snapshot(snapshot))
    repair = plan.repairs[0]
    exact = {
        "name": repair.entity_name,
        "entity_type": repair.entity_type,
        "entity_chunk_id": repair.chunk_id,
        "entity_updated_at": datetime.fromisoformat(repair.original_updated_at),
        "source_path": repair.source_path,
        "content_hash": repair.content_hash,
        "source_chunk_id": repair.chunk_id,
        "source_updated_at": datetime.fromisoformat(repair.source_updated_at),
        "chunk_source_path": repair.source_path,
        "chunk_content_hash": repair.content_hash,
    }
    connection = MagicMock()
    connection.fetch = AsyncMock(return_value=[exact, exact])

    with pytest.raises(PlanError, match="cardinality"):
        await backfill._assert_exact_postwrite_readback(connection, plan)


class _AsyncContext:
    def __init__(self, value=None):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *_args):
        return None


def _pool_and_connection():
    connection = MagicMock()
    connection.transaction.return_value = _AsyncContext()

    async def execute(sql, *_args):
        compact = " ".join(sql.split())
        if compact.startswith("UPDATE entities"):
            return "UPDATE 1"
        if compact.startswith("DELETE FROM entity_sources"):
            return "DELETE 1"
        return "SELECT 1"

    connection.execute = AsyncMock(side_effect=execute)
    connection.fetchval = AsyncMock(return_value=datetime.fromisoformat("2026-07-23T12:00:00+00:00"))
    pool = MagicMock()
    pool.acquire.return_value = _AsyncContext(connection)
    return pool, connection


@pytest.mark.asyncio
async def test_apply_uses_null_only_cas_and_preserves_entity_timestamp(monkeypatch):
    snapshot = _snapshot()
    plan = RepairPlan.from_snapshot(snapshot, "run-001", classify_snapshot(snapshot))
    repair = plan.repairs[0]
    pool, connection = _pool_and_connection()
    entity = {
        "name": repair.entity_name,
        "entity_type": repair.entity_type,
        "source_chunk_id": None,
        "updated_at": datetime.fromisoformat(repair.original_updated_at),
    }
    monkeypatch.setenv("LTM0019_APPLY_GO", plan.plan_sha256)
    monkeypatch.setattr(backfill, "_acquire_session_lock", AsyncMock())
    monkeypatch.setattr(backfill, "_release_session_lock", AsyncMock())
    monkeypatch.setattr(backfill, "_configure_mutation_transaction", AsyncMock())
    monkeypatch.setattr(backfill, "_lock_plan_rows", AsyncMock())
    monkeypatch.setattr(backfill, "_current_repair_state", AsyncMock(return_value=(entity, [])))
    monkeypatch.setattr(backfill, "load_snapshot", AsyncMock(return_value=snapshot))
    monkeypatch.setattr(backfill, "_assert_exact_postwrite_readback", AsyncMock())
    connection.fetchval.return_value = datetime.fromisoformat(repair.source_updated_at)

    result = await backfill.apply_plan(pool, plan)

    assert result == {
        "applied": 1,
        "already_applied": 0,
        "source_updated_at": {repair.entity_id: repair.source_updated_at},
    }
    sql = "\n".join(call.args[0] for call in connection.execute.await_args_list)
    assert "source_chunk_id IS NULL" in sql
    assert "updated_at =" in sql
    assert "SET source_chunk_id =" in sql
    assert "SET source_chunk_id =" in sql and "SET source_chunk_id = $2::uuid, updated_at" not in sql
    insert_sql = connection.fetchval.await_args.args[0]
    assert "INSERT INTO entity_sources" in insert_sql
    assert "$6::timestamptz" in insert_sql
    assert "ON CONFLICT" not in insert_sql
    backfill._assert_exact_postwrite_readback.assert_awaited_once_with(connection, plan)


@pytest.mark.asyncio
async def test_rollback_requires_receipt_timestamp_and_restores_updated_at(monkeypatch):
    snapshot = _snapshot()
    plan = RepairPlan.from_snapshot(snapshot, "run-001", classify_snapshot(snapshot))
    repair = plan.repairs[0]
    pool, connection = _pool_and_connection()
    applied_timestamp = datetime.fromisoformat(repair.source_updated_at)
    entity = {
        "name": repair.entity_name,
        "entity_type": repair.entity_type,
        "source_chunk_id": repair.chunk_id,
        "updated_at": datetime.fromisoformat(repair.original_updated_at),
    }
    sources = [
        {
            "namespace_id": plan.namespace_id,
            "source_path": repair.source_path,
            "content_hash": repair.content_hash,
            "source_chunk_id": repair.chunk_id,
            "updated_at": applied_timestamp,
        }
    ]
    receipt = {
        "task_id": "LTM-0019",
        "plan_sha256": plan.plan_sha256,
        "source_updated_at": {repair.entity_id: applied_timestamp.isoformat()},
    }
    monkeypatch.setenv("LTM0019_ROLLBACK_GO", plan.plan_sha256)
    monkeypatch.setattr(backfill, "_acquire_session_lock", AsyncMock())
    monkeypatch.setattr(backfill, "_release_session_lock", AsyncMock())
    monkeypatch.setattr(backfill, "_configure_mutation_transaction", AsyncMock())
    monkeypatch.setattr(backfill, "_lock_plan_rows", AsyncMock())
    monkeypatch.setattr(backfill, "_current_repair_state", AsyncMock(return_value=(entity, sources)))
    monkeypatch.setattr(backfill, "load_snapshot", AsyncMock(return_value=snapshot))

    assert await backfill.rollback_plan(pool, plan, receipt) == {"rolled_back": 1}

    sql = "\n".join(call.args[0] for call in connection.execute.await_args_list)
    assert "DELETE FROM entity_sources" in sql
    assert "AND updated_at = $6::timestamptz" in sql
    assert "SET source_chunk_id = NULL, updated_at = $6::timestamptz" in sql
