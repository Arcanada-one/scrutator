"""SRCH-0036 — index freshness detection.

Compares the set of `source_path`s currently indexed in Scrutator (per namespace)
against the current corpus (filesystem or an ingest manifest), and reports:

- STALE:   indexed but no longer present in the corpus (deleted/moved on disk).
- MISSING: present in the corpus but not indexed (never ingested).

This module is READ-ONLY. It never deletes stale chunks, never re-ingests
missing sources, and never writes to the live Scrutator database or index.
The dry-run `--plan` mode only *describes* what a re-index run would do — the
actual re-index (deletion of stale chunks / re-ingest of missing sources)
against a live namespace is a separate, hard-gated operator action.

Usage:
    python -m scrutator.tools.index_freshness \\
        --namespace arcanada --corpus-root /path/to/kb --plan
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_NAMESPACE = "arcanada"
DEFAULT_CORPUS_EXTENSIONS: tuple[str, ...] = (".md", ".markdown", ".pdf")
_REPORT_PREVIEW_LIMIT = 20


@dataclass(frozen=True)
class IndexedSource:
    """One distinct `source_path` currently indexed for a namespace."""

    source_path: str
    chunk_count: int


@dataclass(frozen=True)
class FreshnessReport:
    """Result of comparing the indexed source set against the current corpus."""

    namespace: str
    generated_at: str
    indexed_count: int
    corpus_count: int
    stale: list[dict]
    missing: list[dict]

    @property
    def stale_count(self) -> int:
        return len(self.stale)

    @property
    def missing_count(self) -> int:
        return len(self.missing)

    @property
    def is_clean(self) -> bool:
        return not self.stale and not self.missing

    def to_dict(self) -> dict:
        return {
            "namespace": self.namespace,
            "generated_at": self.generated_at,
            "indexed_count": self.indexed_count,
            "corpus_count": self.corpus_count,
            "stale_count": self.stale_count,
            "missing_count": self.missing_count,
            "clean": self.is_clean,
            "stale": self.stale,
            "missing": self.missing,
        }

    def human_summary(self) -> str:
        lines = [
            f"Index freshness — namespace '{self.namespace}' (generated {self.generated_at})",
            f"  indexed: {self.indexed_count}  corpus: {self.corpus_count}",
        ]
        if self.is_clean:
            lines.append("  clean — no stale or missing source paths")
            return "\n".join(lines)
        if self.stale:
            lines.append(f"  STALE ({self.stale_count}) — indexed but no longer in the corpus:")
            lines.extend(_preview_lines(self.stale, lambda e: f"{e['source_path']} ({e['chunk_count']} chunks)"))
        if self.missing:
            lines.append(f"  MISSING ({self.missing_count}) — in the corpus but not indexed:")
            lines.extend(_preview_lines(self.missing, lambda e: str(e["source_path"])))
        return "\n".join(lines)


def _preview_lines(entries: list[dict], fmt) -> list[str]:
    lines = [f"    - {fmt(e)}" for e in entries[:_REPORT_PREVIEW_LIMIT]]
    remainder = len(entries) - _REPORT_PREVIEW_LIMIT
    if remainder > 0:
        lines.append(f"    ... and {remainder} more")
    return lines


def detect_freshness(
    indexed: list[IndexedSource],
    corpus_paths: set[str],
    namespace: str,
    generated_at: str,
) -> FreshnessReport:
    """Pure comparison: indexed source_paths vs the current on-disk corpus."""
    indexed_map = {s.source_path: s for s in indexed}
    indexed_set = set(indexed_map)
    stale_paths = sorted(indexed_set - corpus_paths)
    missing_paths = sorted(corpus_paths - indexed_set)
    return FreshnessReport(
        namespace=namespace,
        generated_at=generated_at,
        indexed_count=len(indexed_set),
        corpus_count=len(corpus_paths),
        stale=[{"source_path": p, "chunk_count": indexed_map[p].chunk_count} for p in stale_paths],
        missing=[{"source_path": p} for p in missing_paths],
    )


def scan_corpus_paths(corpus_root: Path, extensions: tuple[str, ...] = DEFAULT_CORPUS_EXTENSIONS) -> set[str]:
    """Enumerate on-disk source paths under `corpus_root` as POSIX-relative strings."""
    if not corpus_root.is_dir():
        raise FileNotFoundError(f"corpus root not found: {corpus_root}")
    lowered_extensions = tuple(ext.lower() for ext in extensions)
    return {
        path.relative_to(corpus_root).as_posix()
        for path in corpus_root.rglob("*")
        if path.is_file() and path.suffix.lower() in lowered_extensions
    }


def load_manifest_paths(manifest_path: Path) -> set[str]:
    """Load on-disk source paths from a JSON ingest manifest.

    Accepts either a bare JSON array of strings, or `{"paths": [...]}`.
    """
    data = json.loads(manifest_path.read_text())
    if isinstance(data, dict):
        data = data.get("paths", [])
    if not isinstance(data, list):
        raise ValueError(f"manifest {manifest_path} must be a JSON array or an object with a 'paths' array")
    return {str(p) for p in data}


async def fetch_indexed_sources(database_url: str, namespace: str) -> list[IndexedSource]:
    """READ-ONLY: distinct source_paths + chunk counts currently indexed for `namespace`.

    Opens its own short-lived connection rather than reusing
    `scrutator.db.connection`'s app-lifecycle-bound pool, so this tool has no
    runtime coupling to the FastAPI app and can run standalone (e.g. from CI
    or an operator shell).
    """
    import asyncpg

    conn = await asyncpg.connect(dsn=database_url)
    try:
        rows = await conn.fetch(
            """
            SELECT c.source_path, COUNT(*)::int AS chunk_count
            FROM chunks c
            JOIN namespaces n ON n.id = c.namespace_id
            WHERE n.name = $1
            GROUP BY c.source_path
            ORDER BY c.source_path
            """,
            namespace,
        )
    finally:
        await conn.close()
    return [IndexedSource(source_path=row["source_path"], chunk_count=row["chunk_count"]) for row in rows]


async def probe_health(api_url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """READ-ONLY `GET /health` connectivity probe. Never used for enumeration."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{api_url.rstrip('/')}/health")
    except httpx.HTTPError as exc:
        return False, f"/health unreachable: {exc}"
    if resp.status_code == 200:
        return True, "ok"
    return False, f"/health returned {resp.status_code}"


def build_reindex_plan(report: FreshnessReport) -> dict:
    """Build a dry-run re-index PLAN — actions a future run *would* take.

    This function only describes actions; it never executes them. Executing a
    re-index (deleting stale chunks, re-ingesting missing sources) against a
    live Scrutator namespace is a separate, hard-gated operator action.
    """
    actions = [
        {"action": "delete", "source_path": e["source_path"], "reason": "stale-indexed-but-gone"} for e in report.stale
    ] + [
        {"action": "reingest", "source_path": e["source_path"], "reason": "on-disk-but-unindexed"}
        for e in report.missing
    ]
    return {
        "namespace": report.namespace,
        "generated_at": report.generated_at,
        "action_count": len(actions),
        "actions": actions,
        "executed": False,
    }


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def run_detection(
    *,
    database_url: str,
    namespace: str,
    corpus_root: Path | None,
    manifest_path: Path | None,
    extensions: tuple[str, ...] = DEFAULT_CORPUS_EXTENSIONS,
) -> FreshnessReport:
    """Fetch the indexed set + the current corpus, and return the comparison."""
    if manifest_path is not None:
        corpus_paths = load_manifest_paths(manifest_path)
    elif corpus_root is not None:
        corpus_paths = scan_corpus_paths(corpus_root, extensions)
    else:
        raise ValueError("one of corpus_root or manifest_path is required")

    indexed = await fetch_indexed_sources(database_url, namespace)
    return detect_freshness(indexed, corpus_paths, namespace, _now_iso())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SRCH-0036 — detect stale/missing indexed source paths. Read-only, report-only by default."
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="namespace to check (default: %(default)s)")
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres DSN (default: scrutator.config.settings.database_url)",
    )
    parser.add_argument("--corpus-root", type=Path, default=None, help="filesystem root to scan for the current corpus")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSON ingest manifest listing on-disk source paths (alternative to --corpus-root)",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_CORPUS_EXTENSIONS),
        help="file extensions to include when scanning --corpus-root",
    )
    parser.add_argument("--plan", action="store_true", help="also emit a dry-run re-index plan (never executed)")
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report (+ plan) to this path")
    parser.add_argument("--probe-url", default=None, help="optional read-only GET /health probe before detection")
    parser.add_argument(
        "--fail-on-stale",
        action="store_true",
        help="exit 1 if any stale/missing paths are found (for CI use)",
    )
    return parser


async def _amain(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.corpus_root is None and args.manifest is None:
        print("ERROR: one of --corpus-root or --manifest is required", file=sys.stderr)
        return 2

    if args.probe_url:
        reachable, reason = await probe_health(args.probe_url)
        print(f"probe {args.probe_url}: {'OK' if reachable else 'UNREACHABLE'} — {reason}")

    if args.database_url is not None:
        database_url = args.database_url
    else:
        from scrutator.config import settings

        database_url = settings.database_url

    try:
        report = await run_detection(
            database_url=database_url,
            namespace=args.namespace,
            corpus_root=args.corpus_root,
            manifest_path=args.manifest,
            extensions=tuple(args.extensions),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(report.human_summary())

    output: dict = {"report": report.to_dict()}
    if args.plan:
        plan = build_reindex_plan(report)
        output["plan"] = plan
        print(f"\nRe-index PLAN ({plan['action_count']} action(s), NOT executed):")
        print(json.dumps(plan, indent=2))

    if args.output:
        args.output.write_text(json.dumps(output, indent=2))
        print(f"\nReport written to: {args.output}")

    if args.fail_on_stale and not report.is_clean:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_amain(argv))


if __name__ == "__main__":
    sys.exit(main())
