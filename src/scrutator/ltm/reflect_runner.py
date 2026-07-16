"""Bounded incremental runner for the LTM reflect layer (LTM-0026)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from scrutator.config import settings
from scrutator.db import repository
from scrutator.ltm.llm import LtmLlmClient
from scrutator.ltm.reflect import ReflectBudget, ReflectJob


class ReflectRunnerError(Exception):
    """Raised for operator-correctable runner configuration errors."""


@dataclass(frozen=True)
class ReflectCursor:
    """Persisted cursor for periodic reflect runs."""

    last_completed_at: datetime | None = None

    @classmethod
    def load(cls, path: Path) -> ReflectCursor:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get("last_completed_at")
        if raw is None:
            return cls()
        return cls(last_completed_at=_parse_datetime(raw))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "last_completed_at": self.last_completed_at.isoformat() if self.last_completed_at else None,
        }
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _create_llm_client() -> LtmLlmClient:
    return LtmLlmClient(
        mc_url=settings.ltm_mc_url,
        connector=settings.ltm_connector,
        model=settings.ltm_model,
        api_key=settings.ltm_mc_api_key,
    )


async def run_reflect_once(
    *,
    namespace: str,
    state_file: Path,
    since: datetime | None = None,
    max_chunks: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Run one bounded reflect pass and update the cursor after durable success."""
    if not settings.ltm_reflect_enabled:
        raise ReflectRunnerError("reflect disabled by config")
    namespace_id = await repository.get_namespace_id(namespace)
    if namespace_id is None:
        raise ReflectRunnerError(f"namespace not found: {namespace}")

    cursor = ReflectCursor.load(state_file)
    effective_since = since if since is not None else cursor.last_completed_at
    started_at = datetime.now(UTC)
    job = ReflectJob(
        llm=_create_llm_client(),
        namespace=namespace,
        namespace_id=namespace_id,
        budget=ReflectBudget(
            max_usd=settings.ltm_reflect_budget_usd,
            max_req=settings.ltm_reflect_budget_req_count,
        ),
        max_meta_facts_per_group=settings.ltm_reflect_max_meta_facts_per_chunk,
    )
    summary, facts = await job.run(since=effective_since, max_chunks=max_chunks, dry_run=dry_run)
    if summary.status == "done" and not dry_run:
        ReflectCursor(last_completed_at=started_at).save(state_file)
    return {
        "namespace": namespace,
        "since": effective_since.isoformat() if effective_since else None,
        "dry_run": dry_run,
        "summary": summary.model_dump(mode="json"),
        "preview_count": len(facts) if dry_run else 0,
        "cursor_updated": summary.status == "done" and not dry_run,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one bounded LTM reflect pass")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--state-file", required=True, type=Path)
    parser.add_argument("--since", help="ISO-8601 lower bound; overrides state-file cursor")
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


async def _main_async(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = await run_reflect_once(
            namespace=args.namespace,
            state_file=args.state_file,
            since=_parse_datetime(args.since) if args.since else None,
            max_chunks=args.max_chunks,
            dry_run=args.dry_run,
        )
    except ReflectRunnerError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, separators=(",", ":")))
        return 78
    print(json.dumps(result, separators=(",", ":")))
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main_async(argv))


if __name__ == "__main__":
    raise SystemExit(main())
