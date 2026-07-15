from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.muneral_sync.cli import FULL_BACKFILL_GO, RunMode, execute, parse_args, read_cursor, write_cursor_atomic
from tools.muneral_sync.client import LtmClient, ProtocolError
from tools.muneral_sync.graph import build_ingest_payload, canonical_hash
from tools.muneral_sync.secretscan import (
    VERDICT_CLEAN,
    VERDICT_CRITICAL,
    VERDICT_INFO,
    ScanError,
    scan_serialized,
    scan_text,
)
from tools.muneral_sync.source import ChangeRow, MuneralSource

FIXTURE = Path(__file__).parent / "fixtures/muneral_task_aggregate.json"
INGEST_RESPONSE_FIXTURE = Path(__file__).parent / "fixtures/scrutator_structured_ingest_response.testclient.json"


@pytest.fixture
def aggregate() -> dict:
    return json.loads(FIXTURE.read_text())


def test_canonical_hash_is_sorted_stable_and_excludes_runtime_fields(aggregate):
    reordered = {key: aggregate[key] for key in reversed(aggregate)}
    reordered["task"]["captured_at"] = "2000-01-01T00:00:00Z"
    reordered["task"]["run_id"] = "other-run"
    assert canonical_hash(reordered) == canonical_hash(aggregate)
    assert len(canonical_hash(aggregate)) == 64


def test_graph_contract_has_stable_nodes_properties_and_relation_direction(aggregate):
    payload = build_ingest_payload(aggregate)
    graph = payload["structured_graph"]
    task_id = aggregate["task"]["id"]
    project_id = aggregate["project"]["id"]
    names = {entity["name"]: entity for entity in graph["entities"]}
    assert payload["source_path"] == f"muneral://task/{task_id}"
    assert payload["namespace"] == "muneral"
    assert names[f"MUN:{task_id}"]["entity_type"] == "task"
    assert names[f"MUN:{task_id}"]["description"] == "LTM-0025 Muneral to KB graph-merge pilot"
    assert names[f"MUN-PROJECT:{project_id}"]["entity_type"] == "project"
    assert names[f"MUN-PROJECT:{project_id}"]["description"] == "Long Term Memory"
    assert "MUN-TAG:graph-merge" in names
    assert "MUN-TAG:pilot" in names
    assert "MUN-ACTOR:agent:44444444-4444-4444-8444-444444444444" in names
    assert "MUN-ACTOR:agent:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa" in names
    edges = {(edge["source"], edge["target"], edge["relation"], edge["weight"]) for edge in graph["edges"]}
    task = f"MUN:{task_id}"
    assert (task, f"MUN:{aggregate['task']['parent_id']}", "subtask-of", 1.0) in edges
    assert (task, f"MUN-PROJECT:{project_id}", "belongs-to-project", 1.0) in edges
    assert (task, "MUN-TAG:graph-merge", "tagged", 1.0) in edges
    assert (task, "MUN-ACTOR:agent:44444444-4444-4444-8444-444444444444", "performed-by", 1.0) in edges
    assert (task, "MUN-ACTOR:agent:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "assigned-implementer", 1.0) in edges
    assert (task, "MUN:55555555-5555-4555-8555-555555555555", "depends-on", 1.0) in edges
    assert ("MUN:66666666-6666-4666-8666-666666666666", task, "blocks", 1.0) in edges
    assert all(edge[3] == 1.0 for edge in edges)


def test_searchable_content_is_deterministic_and_contains_contract_fields(aggregate):
    first = build_ingest_payload(aggregate)["content"]
    second = build_ingest_payload(deepcopy(aggregate))["content"]
    assert first == second
    for expected in (
        "LTM-0025 Muneral to KB graph-merge pilot",
        "11111111-1111-4111-8111-111111111111",
        "in_progress",
        "Long Term Memory",
        "graph-merge, pilot",
        "Project the pilot task into the knowledge graph.",
    ):
        assert expected in first
    assert "captured_at" not in first and "run_id" not in first


def test_tag_normalization_preserves_non_ascii_identity(aggregate):
    aggregate["tags"] = ["  Граф Знаний  "]
    entity_names = {entity["name"] for entity in build_ingest_payload(aggregate)["structured_graph"]["entities"]}
    assert "MUN-TAG:граф-знаний" in entity_names


def test_nullable_checklist_position_is_preserved_and_sorted_last(aggregate):
    aggregate["checklists"][0]["position"] = None
    snapshot = build_ingest_payload(aggregate)["content"]
    assert snapshot.index("Scan payload") < snapshot.index("Prove recall")


def test_shared_entities_are_task_order_independent_and_stubs_do_not_clobber_owned_task(aggregate):
    other = deepcopy(aggregate)
    owner_id = aggregate["task"]["id"]
    other_id = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    other["task"]["id"] = other_id
    other["task"]["title"] = "Other task"
    other["task"]["created_by_id"] = aggregate["task"]["created_by_id"]
    other["task"]["parent_id"] = owner_id
    other["dependencies"] = []
    first = build_ingest_payload(aggregate)["structured_graph"]["entities"]
    second = build_ingest_payload(other)["structured_graph"]["entities"]
    shared_names = {
        f"MUN-PROJECT:{aggregate['project']['id']}",
        "MUN-TAG:graph-merge",
        f"MUN-ACTOR:agent:{aggregate['task']['created_by_id']}",
    }
    by_name_first = {entity["name"]: entity for entity in first}
    by_name_second = {entity["name"]: entity for entity in second}
    assert {name: by_name_first[name] for name in shared_names} == {name: by_name_second[name] for name in shared_names}
    owner = by_name_first[f"MUN:{owner_id}"]
    stub = by_name_second[f"MUN:{owner_id}"]
    merged_owner_then_stub = {**owner["properties"], **stub["properties"]}
    merged_stub_then_owner = {**stub["properties"], **owner["properties"]}
    assert "source_ref" not in stub["properties"] and "content_hash" not in stub["properties"]
    assert merged_owner_then_stub["source_ref"] == merged_stub_then_owner["source_ref"] == f"muneral://task/{owner_id}"
    assert merged_owner_then_stub["content_hash"] == merged_stub_then_owner["content_hash"]


class _Transaction:
    def __init__(self):
        self.entered = False

    async def __aenter__(self):
        self.entered = True

    async def __aexit__(self, *_args):
        return False


@pytest.mark.asyncio
async def test_source_reads_plural_live_tables_in_read_only_repeatable_read_order():
    conn = AsyncMock()
    transaction = _Transaction()
    conn.transaction = MagicMock(return_value=transaction)
    conn.fetchrow.return_value = {
        "id": "task-1",
        "project_id": "project-1",
        "title": "Pilot",
        "project_identity": "project-1",
        "project_name": "Long Term Memory",
        "project_slug": "ltm",
    }
    conn.fetch.side_effect = [[], [], [], [], []]
    source = MuneralSource("postgresql://redacted", activity_limit=25, connect=AsyncMock(return_value=conn))
    aggregate = await source.fetch_task("task-1")
    conn.transaction.assert_called_once_with(isolation="repeatable_read", readonly=True)
    sql = "\n".join(call.args[0] for call in [*conn.fetchrow.await_args_list, *conn.fetch.await_args_list])
    for table in (
        "tasks",
        "projects",
        "task_tags",
        "task_dependencies",
        "task_checklists",
        "task_agents",
        "activity_log",
    ):
        assert table in sql
    assert sql.count("ORDER BY") >= 5
    activity_call = next(call for call in conn.fetch.await_args_list if "activity_log" in call.args[0])
    assert activity_call.args[-1] == 25
    assert aggregate["task"]["project_id"] == "project-1"
    assert aggregate["project"] == {"id": "project-1", "name": "Long Term Memory", "slug": "ltm"}
    await source.close()


@pytest.mark.asyncio
async def test_incremental_source_reads_revision_registry_deterministically_and_filters_mismatches():
    conn = AsyncMock()
    transaction = _Transaction()
    conn.transaction = MagicMock(return_value=transaction)
    conn.fetch.return_value = [
        {"task_id": "task-1", "revision": 2, "changed_at": "later", "deleted": False},
        {"task_id": "task-2", "revision": 7, "changed_at": "later", "deleted": True},
    ]
    source = MuneralSource("postgresql://redacted", connect=AsyncMock(return_value=conn))
    changes = await source.list_incremental_changes({"task-1": 2, "task-2": 6})
    assert changes == [ChangeRow(task_id="task-2", revision=7, changed_at="later", deleted=True)]
    sql = conn.fetch.await_args.args[0]
    assert "muneral_kb_task_changes" in sql and "ORDER BY task_id" in sql


def test_scanner_clean_info_critical_and_never_exposes_matched_text():
    assert scan_text('{"title":"pilot"}').verdict == VERDICT_CLEAN
    info = scan_text("service at 100.70.137.104", info_patterns=[r"\b100\.(?:\d{1,3}\.){2}\d{1,3}\b"])
    assert info.verdict == VERDICT_INFO
    critical = scan_text("PGPASSWORD=do-not-return-this")
    assert critical.verdict == VERDICT_CRITICAL
    serialized = json.dumps(critical.as_dict())
    assert "do-not-return-this" not in serialized
    assert set(critical.as_dict()["findings"][0]) == {"rule", "severity", "line", "span_hash"}


def test_scanner_detects_high_entropy_assignment_in_exact_compact_json():
    result = scan_text('{"api_key":"V7cH2rQ9xN4mK8pL5sT1wZ6yB3dF0gJ2"}')
    assert result.verdict == VERDICT_CRITICAL
    assert result.findings[0].rule == "generic-entropy"


def test_additive_gitleaks_scans_exact_body_and_redacts_secret(monkeypatch):
    body = '{"description":"safe"}'
    seen = []

    def fake_run(command, **_kwargs):
        source = Path(command[command.index("--source") + 1])
        seen.append(source.read_text())
        report = json.dumps([{"RuleID": "generic-api-key", "Secret": "never-return-this", "StartLine": 1}])
        return SimpleNamespace(stdout=report, returncode=1)

    monkeypatch.setattr("tools.muneral_sync.secretscan.shutil.which", lambda _name: "/usr/bin/gitleaks")
    monkeypatch.setattr("tools.muneral_sync.secretscan.subprocess.run", fake_run)
    result = scan_serialized(body)
    assert seen == [body] and result.verdict == VERDICT_CRITICAL
    assert "never-return-this" not in json.dumps(result.as_dict())
    assert result.findings[-1].rule == "gitleaks:generic-api-key"


def test_gitleaks_absent_keeps_python_fallback(monkeypatch):
    monkeypatch.setattr("tools.muneral_sync.secretscan.shutil.which", lambda _name: None)
    assert scan_serialized("PGPASSWORD=still-blocked").verdict == VERDICT_CRITICAL


@pytest.mark.parametrize(
    "runner",
    [
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("gitleaks", 60)),
        lambda *_args, **_kwargs: SimpleNamespace(stdout="not-json-with-secret-output", returncode=0),
        lambda *_args, **_kwargs: SimpleNamespace(stdout="[]", stderr="sensitive operational error", returncode=2),
    ],
    ids=["timeout", "malformed-report", "operational-nonzero"],
)
def test_installed_gitleaks_operational_failures_block_without_output(monkeypatch, runner):
    monkeypatch.setattr("tools.muneral_sync.secretscan.shutil.which", lambda _name: "/usr/bin/gitleaks")
    monkeypatch.setattr("tools.muneral_sync.secretscan.subprocess.run", runner)
    with pytest.raises(ScanError) as raised:
        scan_serialized('{"description":"safe"}')
    assert str(raised.value) == "gitleaks scan failed closed"
    assert "secret" not in str(raised.value).lower()


@pytest.mark.asyncio
async def test_client_scans_exact_serialized_bytes_immediately_before_post(tmp_path):
    credential = tmp_path / "writer"
    credential.write_text("writer-token\n")
    payload = {"namespace": "muneral", "content": "safe", "structured_graph": {"entities": [], "edges": []}}
    response = MagicMock()
    response.raise_for_status.return_value = None
    recorded = json.loads(INGEST_RESPONSE_FIXTURE.read_text())
    assert recorded["fixture_metadata"]["live_deploy_confirmed"] is False
    response.json.return_value = recorded["response"]
    http = AsyncMock()
    http.post.return_value = response
    seen: list[bytes] = []

    def scanner(body: str):
        seen.append(body.encode())
        return SimpleNamespace(verdict=VERDICT_CLEAN, is_critical=False, findings=[])

    client = LtmClient("https://kb.example/v1/ltm/ingest", credential, http=http, scanner=scanner)
    await client.ingest(payload)
    sent = http.post.await_args.kwargs["content"]
    assert seen == [sent]
    assert http.post.await_args.kwargs["headers"]["X-LTM-Writer-Token"] == "writer-token"


@pytest.mark.parametrize(
    "body",
    [
        {"edges_upserted": 0, "idempotent_noop": False},
        {
            "job_id": "job-1",
            "status": "done",
            "entities_upserted": True,
            "edges_upserted": 0,
            "idempotent_noop": False,
        },
        {
            "job_id": "job-1",
            "status": "done",
            "entities_upserted": -1,
            "edges_upserted": 0,
            "idempotent_noop": False,
        },
        {
            "job_id": "job-1",
            "status": "done",
            "entities_upserted": 0,
            "edges_upserted": "1",
            "idempotent_noop": False,
        },
        {
            "job_id": "job-1",
            "status": "done",
            "entities_upserted": 0,
            "edges_upserted": 1,
            "idempotent_noop": 0,
        },
        {
            "job_id": "",
            "status": "done",
            "entities_upserted": 0,
            "edges_upserted": 0,
            "idempotent_noop": False,
        },
        {
            "job_id": "job-1",
            "status": "failed",
            "entities_upserted": 0,
            "edges_upserted": 0,
            "idempotent_noop": False,
        },
        {
            "job_id": "job-1",
            "entities_upserted": 0,
            "edges_upserted": 0,
            "idempotent_noop": False,
        },
    ],
)
@pytest.mark.asyncio
async def test_client_rejects_malformed_ingest_success_without_echoing_body(tmp_path, body):
    credential = tmp_path / "writer"
    credential.write_text("writer-token")
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = body
    http = AsyncMock()
    http.post.return_value = response
    with pytest.raises(ProtocolError) as raised:
        await LtmClient("https://kb.example/v1/ltm/ingest", credential, http=http).ingest({"content": "safe"})
    assert str(raised.value) == "invalid LTM success response"


@pytest.mark.asyncio
async def test_client_fails_closed_on_critical_or_scanner_exception(tmp_path):
    credential = tmp_path / "writer"
    credential.write_text("secret")
    http = AsyncMock()

    def critical(_body):
        return SimpleNamespace(verdict=VERDICT_CRITICAL, is_critical=True, findings=[])

    with pytest.raises(ScanError):
        await LtmClient("https://kb.example", credential, http=http, scanner=critical).ingest({"content": "x"})
    with pytest.raises(ScanError):
        await LtmClient(
            "https://kb.example", credential, http=http, scanner=lambda _body: (_ for _ in ()).throw(RuntimeError())
        ).ingest({"content": "x"})
    http.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_client_tombstone_uses_writer_credential_and_source_path(tmp_path):
    credential = tmp_path / "writer"
    credential.write_text("writer-token")
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "chunks_deleted": 3,
        "entity_sources_deleted": 2,
        "edge_sources_deleted": 1,
        "edges_deleted": 1,
        "entities_deleted": 0,
        "idempotent_noop": False,
    }
    http = AsyncMock()
    http.request.return_value = response
    scanned = []

    def scanner(body):
        scanned.append(body)
        return SimpleNamespace(verdict=VERDICT_CLEAN, is_critical=False, findings=[])

    client = LtmClient("https://kb.example/v1/ltm/ingest", credential, http=http, scanner=scanner)
    result = await client.tombstone("muneral", "muneral://task/task-2")
    assert result == response.json.return_value
    request = http.request.await_args
    assert request.args[:2] == ("DELETE", "https://kb.example/v1/ltm/source")
    assert json.loads(request.kwargs["content"]) == {
        "namespace": "muneral",
        "source_path": "muneral://task/task-2",
    }
    assert scanned == [request.kwargs["content"].decode()]
    assert request.kwargs["headers"]["X-LTM-Writer-Token"] == "writer-token"


@pytest.mark.parametrize(
    "body",
    [
        {"chunks_deleted": 0, "idempotent_noop": True},
        {
            "chunks_deleted": False,
            "entity_sources_deleted": 0,
            "edge_sources_deleted": 0,
            "edges_deleted": 0,
            "entities_deleted": 0,
            "idempotent_noop": True,
        },
        {
            "chunks_deleted": 0,
            "entity_sources_deleted": -1,
            "edge_sources_deleted": 0,
            "edges_deleted": 0,
            "entities_deleted": 0,
            "idempotent_noop": True,
        },
    ],
)
@pytest.mark.asyncio
async def test_client_rejects_malformed_tombstone_success(tmp_path, body):
    credential = tmp_path / "writer"
    credential.write_text("writer-token")
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = body
    http = AsyncMock()
    http.request.return_value = response
    with pytest.raises(ProtocolError, match="^invalid LTM success response$"):
        await LtmClient("https://kb.example/v1/ltm/ingest", credential, http=http).tombstone(
            "muneral", "muneral://task/task-2"
        )


def test_cli_modes_and_full_backfill_gate():
    assert parse_args(["--task-id", "11111111-1111-4111-8111-111111111111"]).mode is RunMode.TASK
    assert parse_args(["--incremental"]).mode is RunMode.INCREMENTAL
    assert parse_args(["--all", "--dry-run"]).mode is RunMode.ALL
    with pytest.raises(SystemExit):
        parse_args(["--all"])
    assert parse_args(["--all", "--operator-go", FULL_BACKFILL_GO]).mode is RunMode.ALL
    with pytest.raises(SystemExit):
        parse_args(["--timer", "--task-id", "11111111-1111-4111-8111-111111111111"])


@pytest.mark.asyncio
async def test_dry_run_never_posts_and_reports_counts_hashes_only(aggregate, tmp_path):
    args = parse_args(["--all", "--dry-run", "--dsn-credential", str(tmp_path / "dsn")])
    source = AsyncMock()
    other = deepcopy(aggregate)
    other["task"]["id"] = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    source.list_all_task_ids.return_value = [aggregate["task"]["id"], other["task"]["id"]]
    source.fetch_task.side_effect = [aggregate, other]
    client = AsyncMock()
    report = await execute(args, source=source, client=client)
    client.ingest.assert_not_awaited()
    graphs = [build_ingest_payload(item)["structured_graph"] for item in (aggregate, other)]
    expected_entities = len({(e["name"], e["entity_type"]) for graph in graphs for e in graph["entities"]})
    expected_edges = len({(e["source"], e["target"], e["relation"]) for graph in graphs for e in graph["edges"]})
    assert report["tasks"] == 2 and report["projects"] == 1
    assert report["entities"] == expected_entities and report["edges"] == expected_edges
    assert report["hashes"] == [canonical_hash(aggregate), canonical_hash(other)]
    assert "content" not in json.dumps(report)


@pytest.mark.asyncio
async def test_incremental_cursor_advances_only_after_whole_batch_success(aggregate, tmp_path):
    cursor = tmp_path / "cursor.json"
    args = parse_args(["--incremental", "--cursor-file", str(cursor), "--dsn-credential", str(tmp_path / "dsn")])
    source = AsyncMock()
    changes = [
        ChangeRow(task_id="task-1", revision=2, changed_at="now", deleted=False),
        ChangeRow(task_id="task-2", revision=9, changed_at="now", deleted=False),
    ]
    source.list_incremental_changes.return_value = changes
    source.fetch_task.return_value = aggregate
    client = AsyncMock()
    client.ingest.side_effect = [{"entities_upserted": 2, "edges_upserted": 1}, RuntimeError("network")]
    with pytest.raises(RuntimeError):
        await execute(args, source=source, client=client)
    assert not cursor.exists()
    client.ingest.side_effect = [{"entities_upserted": 2, "edges_upserted": 1}] * 2
    await execute(args, source=source, client=client)
    assert read_cursor(cursor) == {"task-1": 2, "task-2": 9}


@pytest.mark.asyncio
async def test_malformed_http_success_does_not_advance_incremental_cursor(aggregate, tmp_path):
    cursor = tmp_path / "cursor.json"
    credential = tmp_path / "writer"
    credential.write_text("writer-token")
    args = parse_args(["--incremental", "--cursor-file", str(cursor), "--dsn-credential", str(tmp_path / "dsn")])
    source = AsyncMock()
    source.list_incremental_changes.return_value = [
        ChangeRow(task_id=aggregate["task"]["id"], revision=8, changed_at="now", deleted=False)
    ]
    source.fetch_task.return_value = aggregate
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "job_id": "job-failed",
        "status": "failed",
        "entities_upserted": 1,
        "edges_upserted": 1,
        "idempotent_noop": False,
    }
    http = AsyncMock()
    http.post.return_value = response
    client = LtmClient("https://kb.example/v1/ltm/ingest", credential, http=http)
    with pytest.raises(ProtocolError):
        await execute(args, source=source, client=client)
    assert not cursor.exists()


@pytest.mark.asyncio
async def test_malformed_tombstone_success_does_not_advance_incremental_cursor(tmp_path):
    cursor = tmp_path / "cursor.json"
    credential = tmp_path / "writer"
    credential.write_text("writer-token")
    args = parse_args(["--incremental", "--cursor-file", str(cursor), "--dsn-credential", str(tmp_path / "dsn")])
    source = AsyncMock()
    source.list_incremental_changes.return_value = [
        ChangeRow(task_id="task-deleted", revision=9, changed_at="now", deleted=True)
    ]
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"chunks_deleted": 1, "idempotent_noop": False}
    http = AsyncMock()
    http.request.return_value = response
    client = LtmClient("https://kb.example/v1/ltm/ingest", credential, http=http)
    with pytest.raises(ProtocolError):
        await execute(args, source=source, client=client)
    assert not cursor.exists()


@pytest.mark.asyncio
async def test_incremental_deleted_row_tombstones_without_fetching(aggregate, tmp_path):
    cursor = tmp_path / "cursor.json"
    args = parse_args(["--incremental", "--cursor-file", str(cursor), "--dsn-credential", str(tmp_path / "dsn")])
    source = AsyncMock()
    source.list_incremental_changes.return_value = [
        ChangeRow(task_id="task-deleted", revision=4, changed_at="now", deleted=True)
    ]
    client = AsyncMock()
    client.tombstone.return_value = {"chunks_deleted": 1}
    report = await execute(args, source=source, client=client)
    source.fetch_task.assert_not_awaited()
    client.tombstone.assert_awaited_once_with("muneral", "muneral://task/task-deleted")
    assert report["tombstones"] == 1 and read_cursor(cursor) == {"task-deleted": 4}


@pytest.mark.asyncio
async def test_incremental_dry_run_does_not_advance_cursor(aggregate, tmp_path):
    cursor = tmp_path / "cursor.json"
    args = parse_args(
        [
            "--incremental",
            "--dry-run",
            "--cursor-file",
            str(cursor),
            "--dsn-credential",
            str(tmp_path / "dsn"),
        ]
    )
    source = AsyncMock()
    source.list_incremental_changes.return_value = [
        ChangeRow(task_id=aggregate["task"]["id"], revision=1, changed_at="now", deleted=False)
    ]
    source.fetch_task.return_value = aggregate
    await execute(args, source=source, client=None)
    assert not cursor.exists()


def test_cursor_write_is_atomic_rename(tmp_path):
    target = tmp_path / "state" / "cursor.json"
    with patch("tools.muneral_sync.cli.os.replace") as replace:
        write_cursor_atomic(target, {"task-1": 3})
    replace.assert_called_once()
    assert replace.call_args.args[1] == target
