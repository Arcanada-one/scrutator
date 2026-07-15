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
from .source import MuneralSource

FULL_BACKFILL_GO = "FULL-MUNERAL-BACKFILL"
DEFAULT_CURSOR = "1970-01-01T00:00:00+00:00"


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


def read_cursor(path: Path) -> str:
    if not path.exists():
        return DEFAULT_CURSOR
    value = json.loads(path.read_text()).get("updated_at")
    if not isinstance(value, str) or not value:
        raise ValueError("cursor file has no updated_at string")
    return value


def write_cursor_atomic(path: Path, updated_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"updated_at": updated_at}, sort_keys=True, separators=(",", ":")) + "\n"
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


async def execute(args: Arguments, *, source: Any, client: Any | None) -> dict[str, Any]:
    next_cursor: str | None = None
    if args.mode is RunMode.TASK:
        task_ids = [args.task_id]
    elif args.mode is RunMode.ALL:
        task_ids = await source.list_all_task_ids()
    else:
        task_ids, next_cursor = await source.list_incremental_task_ids(read_cursor(args.cursor_file))

    report: dict[str, Any] = {
        "mode": args.mode.value,
        "dry_run": args.dry_run,
        "tasks": 0,
        "entities": 0,
        "edges": 0,
        "hashes": [],
        "entities_upserted": 0,
        "edges_upserted": 0,
        "idempotent_noops": 0,
    }
    for task_id in task_ids:
        aggregate = await source.fetch_task(task_id)
        payload = build_ingest_payload(aggregate)
        graph = payload["structured_graph"]
        report["tasks"] += 1
        report["entities"] += len(graph["entities"])
        report["edges"] += len(graph["edges"])
        report["hashes"].append(graph["content_hash"])
        if not args.dry_run:
            if client is None:
                raise RuntimeError("live sync requires an LTM client")
            result = await client.ingest(payload)
            report["entities_upserted"] += result["entities_upserted"]
            report["edges_upserted"] += result["edges_upserted"]
            report["idempotent_noops"] += int(result.get("idempotent_noop", False))
    if args.mode is RunMode.INCREMENTAL and next_cursor is not None and not args.dry_run:
        write_cursor_atomic(args.cursor_file, next_cursor)
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
