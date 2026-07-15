"""Command line runner for bounded, deterministic Muneral graph sync."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .client import LtmClient
from .graph import build_ingest_payload
from .source import ChangeRow, MuneralSource

FULL_BACKFILL_GO = "FULL-MUNERAL-BACKFILL"
CURSOR_SCHEMA_VERSION = 1


class RunMode(StrEnum):
    TASK = "task"
    INCREMENTAL = "incremental"
    ALL = "all"


@dataclass
class Arguments:
    mode: RunMode
    task_id: str | None
    dry_run: bool
    timer: bool
    dsn_credential: Path
    writer_credential: Path
    cursor_file: Path
    endpoint: str


def parse_args(argv: list[str] | None = None) -> Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--task-id")
    modes.add_argument("--incremental", action="store_true")
    modes.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--operator-go")
    parser.add_argument("--timer", action="store_true")
    parser.add_argument("--dsn-credential", type=Path, default=Path("/run/credentials/muneral-db-dsn"))
    parser.add_argument("--writer-credential", type=Path, default=Path("/run/credentials/ltm-writer-token"))
    parser.add_argument("--cursor-file", type=Path, default=Path("/var/lib/muneral-kb-sync/cursor.json"))
    parser.add_argument("--endpoint", default="https://kb.arcanada.ai/v1/ltm/ingest")
    parsed = parser.parse_args(argv)
    mode = RunMode.TASK if parsed.task_id else RunMode.INCREMENTAL if parsed.incremental else RunMode.ALL
    if mode is RunMode.ALL and not parsed.dry_run and parsed.operator_go != FULL_BACKFILL_GO:
        parser.error(f"live --all requires --operator-go {FULL_BACKFILL_GO}")
    if parsed.operator_go and mode is not RunMode.ALL:
        parser.error("--operator-go is valid only with --all")
    if parsed.timer and mode is not RunMode.INCREMENTAL:
        parser.error("--timer requires --incremental")
    return Arguments(
        mode=mode,
        task_id=parsed.task_id,
        dry_run=parsed.dry_run,
        timer=parsed.timer,
        dsn_credential=parsed.dsn_credential,
        writer_credential=parsed.writer_credential,
        cursor_file=parsed.cursor_file,
        endpoint=parsed.endpoint,
    )


def read_cursor(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if value.get("schema_version") != CURSOR_SCHEMA_VERSION or not isinstance(value.get("revisions"), dict):
        raise ValueError("cursor file has unsupported schema")
    revisions = value["revisions"]
    if any(not isinstance(task_id, str) or not isinstance(revision, int) for task_id, revision in revisions.items()):
        raise ValueError("cursor revisions must map task IDs to integers")
    return revisions


def write_cursor_atomic(path: Path, revisions: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        json.dumps(
            {"schema_version": CURSOR_SCHEMA_VERSION, "revisions": revisions},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


async def _work_items(args: Arguments, source: Any) -> tuple[list[ChangeRow], dict[str, int]]:
    if args.mode is RunMode.TASK:
        return [ChangeRow(str(args.task_id), 0, None, False)], {}
    if args.mode is RunMode.ALL:
        task_ids = await source.list_all_task_ids()
        return [ChangeRow(task_id, 0, None, False) for task_id in task_ids], {}
    revisions = read_cursor(args.cursor_file)
    return await source.list_incremental_changes(revisions), revisions


def _empty_report(args: Arguments) -> dict[str, Any]:
    return {
        "mode": args.mode.value,
        "dry_run": args.dry_run,
        "tasks": 0,
        "projects": 0,
        "entities": 0,
        "edges": 0,
        "tombstones": 0,
        "hashes": [],
        "entities_upserted": 0,
        "edges_upserted": 0,
        "idempotent_noops": 0,
    }


async def execute(args: Arguments, *, source: Any, client: Any | None) -> dict[str, Any]:
    changes, revisions = await _work_items(args, source)
    next_revisions = dict(revisions)
    entity_keys: set[tuple[str, str]] = set()
    edge_keys: set[tuple[str, str, str]] = set()
    project_ids: set[str] = set()
    report = _empty_report(args)
    for change in changes:
        report["tasks"] += 1
        if change.deleted:
            report["tombstones"] += 1
            if not args.dry_run:
                if client is None:
                    raise RuntimeError("live sync requires an LTM client")
                await client.tombstone("muneral", f"muneral://task/{change.task_id}")
            next_revisions[change.task_id] = change.revision
            continue
        aggregate = await source.fetch_task(change.task_id)
        payload = build_ingest_payload(aggregate)
        graph = payload["structured_graph"]
        entity_keys.update((entity["name"], entity["entity_type"]) for entity in graph["entities"])
        edge_keys.update((edge["source"], edge["target"], edge["relation"]) for edge in graph["edges"])
        if payload.get("project"):
            project_ids.add(str(payload["project"]))
        report["hashes"].append(graph["content_hash"])
        if not args.dry_run:
            if client is None:
                raise RuntimeError("live sync requires an LTM client")
            result = await client.ingest(payload)
            report["entities_upserted"] += result["entities_upserted"]
            report["edges_upserted"] += result["edges_upserted"]
            report["idempotent_noops"] += int(result.get("idempotent_noop", False))
        next_revisions[change.task_id] = change.revision
    report["projects"] = len(project_ids)
    report["entities"] = len(entity_keys)
    report["edges"] = len(edge_keys)
    if args.mode is RunMode.INCREMENTAL and not args.dry_run:
        write_cursor_atomic(args.cursor_file, next_revisions)
    return report


async def _run(args: Arguments) -> dict[str, Any]:
    dsn = args.dsn_credential.read_text().strip()
    if not dsn:
        raise ValueError("Muneral DSN credential is empty")
    source = MuneralSource(dsn)
    client = None if args.dry_run else LtmClient(args.endpoint, args.writer_credential)
    try:
        return await execute(args, source=source, client=client)
    finally:
        await source.close()
        if client is not None:
            await client.close()


def main(argv: list[str] | None = None) -> int:
    report = asyncio.run(_run(parse_args(argv)))
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
