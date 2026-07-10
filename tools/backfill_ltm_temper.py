"""One-shot, resumable TEMPR backfill for pre-LTM-0012/0013 chunks (LTM-0014).

HARD-GATED: this script performs LIVE LLM calls (entity/edge/temporal-event
extraction, billed via the configured Model Connector route) AND live writes
to `entities` / `entity_edges` / `entity_events`. It defaults to `--dry-run`
(report-only, zero LLM calls, zero writes); a real run requires the explicit
`--live` flag. This script is authored here but MUST NOT be invoked against
any live namespace (including `arcanada`) by CI, by tests, or by any task
automation — it is operator-run only, same convention as
`tools/backfill_sections.py` (SRCH-0021 Fork 4).

Scope: chunks that predate the TEMPR pipeline (LTM-0012 temporal / LTM-0013
reflect) have zero rows in `entities` pointing back at them — the ingest-time
`IngestPipeline.process_chunk()` extraction only ever ran for chunks ingested
through `POST /v1/ltm/ingest` *after* those features landed. This script finds
that gap directly (`entities e ON e.source_chunk_id = c.id WHERE e.id IS
NULL`) rather than relying on document-level re-ingest.

Idempotent / resumable: `upsert_entity` / `upsert_entity_edge` /
`upsert_entity_event` are ON CONFLICT upserts (LTM-0019 COALESCE-repairs
`source_chunk_id` on every re-run), so re-running this script after a partial
live run or a crash only (re)processes chunks still missing entities — no
duplicate rows, no data loss. Per-chunk failures are logged and skipped so one
bad LLM response cannot abort the whole backfill.

Background: LTM-0014 (backlog) — "Scrutator LTM — Production backfill 1148
chunks с TEMPR". Blocked on LTM-0015 (`_resolve_entity` priority fix,
archived 2026-04-29, `Arcanada-one/scrutator@3c393e1`) — now unblocked.
NOT executed against production by this task; see PR description.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from scrutator.config import settings
from scrutator.db.connection import close_pool, get_pool
from scrutator.ltm.llm import LtmLlmClient
from scrutator.ltm.pipeline import IngestPipeline

log = logging.getLogger("scrutator.tools.backfill_ltm_temper")


async def _fetch_candidate_chunks(namespace: str, limit: int | None = None) -> list[dict]:
    """Chunks in `namespace` with zero rows in `entities` pointing back at them."""
    pool = await get_pool()
    query = """
        SELECT c.id::text AS chunk_id, c.source_path, c.content
        FROM chunks c
        JOIN namespaces n ON n.id = c.namespace_id
        LEFT JOIN entities e ON e.source_chunk_id = c.id
        WHERE n.name = $1
          AND e.id IS NULL
        ORDER BY c.source_path, c.chunk_index
    """
    async with pool.acquire() as conn:
        if limit is not None:
            rows = await conn.fetch(query + " LIMIT $2", namespace, limit)
        else:
            rows = await conn.fetch(query, namespace)
    return [dict(r) for r in rows]


async def _resolve_namespace_id(namespace: str) -> int | None:
    """Read-only namespace_id lookup — does NOT auto-provision (unlike upsert_namespace)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id FROM namespaces WHERE name = $1", namespace)
    return row["id"] if row else None


def _create_llm_client() -> LtmLlmClient:
    """Mirrors `scrutator.ltm.router._create_llm_client` — same connector/model/key config."""
    return LtmLlmClient(
        mc_url=settings.ltm_mc_url,
        connector=settings.ltm_connector,
        model=settings.ltm_model,
        api_key=settings.ltm_mc_api_key,
    )


async def run_backfill(namespace: str, dry_run: bool = True, limit: int | None = None) -> dict:
    """Backfill TEMPR entities/edges/events for one namespace's orphan chunks.

    dry_run=True (default): report the candidate count, make zero LLM calls, write nothing.
    dry_run=False: sequentially runs `IngestPipeline.process_chunk()` per candidate chunk
    (extract_entities → extract_edges → LTM-0012 temporal events, each persisted via
    idempotent upsert). Continues past per-chunk failures (logged, counted, not raised).
    """
    rows = await _fetch_candidate_chunks(namespace, limit=limit)
    processed = 0
    failed = 0

    if not dry_run and rows:
        namespace_id = await _resolve_namespace_id(namespace)
        if namespace_id is None:
            return {
                "namespace": namespace,
                "dry_run": dry_run,
                "candidates": len(rows),
                "processed": 0,
                "failed": 0,
                "error": f"namespace {namespace!r} not found — refusing to auto-provision in a backfill",
            }

        llm = _create_llm_client()
        pipeline = IngestPipeline(
            llm=llm,
            namespace=namespace,
            namespace_id=namespace_id,
            max_entities_per_chunk=settings.ltm_max_entities_per_chunk,
        )
        for row in rows:
            try:
                await pipeline.process_chunk(row["chunk_id"], row["content"])
                processed += 1
            except Exception:
                log.exception("TEMPR backfill failed for chunk %s (%s)", row["chunk_id"], row["source_path"])
                failed += 1

    return {
        "namespace": namespace,
        "dry_run": dry_run,
        "candidates": len(rows),
        "processed": processed,
        "failed": failed,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True, help="namespace to backfill (e.g. 'arcanada')")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of chunks processed this invocation (safe batching / resumable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the candidate count without writing (default behaviour — this flag is a no-op alias)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="perform the live LLM extraction + writes — HARD-GATED, operator-run only (see module docstring)",
    )
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    result = await run_backfill(namespace=args.namespace, dry_run=not args.live, limit=args.limit)
    print(json.dumps(result, indent=2))
    await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
