"""Read-only report over the durable access log (Muneral 4e0054f0).

Answers: which identities used which namespaces in the last N days, reads versus writes, last seen,
and which KNOWN identities had zero use. It reads the access-log files only; the optional list of
known reader principals comes from one read-only SELECT on ``principal_namespace_grants`` and is
skipped, with a note, when the database cannot be reached or ``--no-db`` is given.

Usage (inside the service container):
    python -m scrutator.tools.access_report --days 1
    python -m scrutator.tools.access_report --days 30 --json

Limits that the report repeats in its header: a shared secret is one identity, and routes that act on
the whole allowed set name no namespace (they are listed as ``(unscoped)``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from scrutator.access_log import ACCESS_FILE, CAPABILITY_KINDS

UNSCOPED = "(unscoped)"


@dataclass
class IdentityStats:
    requests: int = 0
    reads: int = 0
    writes: int = 0
    denied: int = 0
    first_seen: str = ""
    last_seen: str = ""
    namespaces: Counter = field(default_factory=Counter)
    sources: Counter = field(default_factory=Counter)
    statuses: Counter = field(default_factory=Counter)


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def iter_records(directory: Path, since: datetime):
    """Yield parsed records newer than ``since`` from the current and the rotated files, skipping bad lines."""
    for path in sorted(directory.glob(ACCESS_FILE + "*")):
        try:
            handle = path.open(encoding="utf-8")
        except OSError:
            continue
        with handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                stamp = _parse_ts(record.get("ts", ""))
                if stamp is not None and stamp >= since:
                    yield record


def aggregate(records) -> tuple[dict[str, IdentityStats], int]:
    stats: dict[str, IdentityStats] = defaultdict(IdentityStats)
    total = 0
    for record in records:
        total += 1
        key = record.get("identity") or "none"
        item = stats[key]
        item.requests += 1
        mode = record.get("mode")
        if mode == "write":
            item.writes += 1
        elif mode == "read":
            item.reads += 1
        if record.get("auth") == "denied" or record.get("status") in {401, 403}:
            item.denied += 1
        ts = record.get("ts", "")
        item.first_seen = min(item.first_seen or ts, ts)
        item.last_seen = max(item.last_seen, ts)
        namespaces = record.get("namespaces") or [UNSCOPED]
        for name in namespaces:
            item.namespaces[name] += 1
        if record.get("src"):
            item.sources[record["src"]] += 1
        item.statuses[str(record.get("status"))] += 1
    return dict(stats), total


async def _known_principals() -> list[str]:
    from scrutator.db.connection import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT DISTINCT principal_id FROM principal_namespace_grants ORDER BY 1")
    return [f"reader:{row['principal_id']}" for row in rows]


def known_identities(use_db: bool) -> tuple[list[str], str]:
    known = list(CAPABILITY_KINDS)
    if not use_db:
        return known, "reader principals not listed (--no-db)"
    try:
        known = list(CAPABILITY_KINDS) + asyncio.run(_known_principals())
        return known, "reader principals from principal_namespace_grants"
    except Exception as exc:  # report without them rather than fail
        return known, f"reader principals NOT listed: database unreachable ({exc.__class__.__name__})"


def build_report(directory: Path, days: float, use_db: bool, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    since = now - timedelta(days=days)
    stats, total = aggregate(iter_records(directory, since))
    known, known_note = known_identities(use_db)
    return {
        "window": {
            "from": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "days": days,
        },
        "directory": str(directory),
        "records": total,
        "identities": {
            name: {
                "requests": s.requests,
                "reads": s.reads,
                "writes": s.writes,
                "denied": s.denied,
                "first_seen": s.first_seen,
                "last_seen": s.last_seen,
                "namespaces": dict(s.namespaces.most_common()),
                "sources": dict(s.sources.most_common()),
                "statuses": dict(s.statuses.most_common()),
            }
            for name, s in sorted(stats.items(), key=lambda kv: -kv[1].requests)
        },
        "known_identities_note": known_note,
        "zero_use": sorted(set(known) - set(stats)),
        "limits": [
            "A shared secret is one identity: all feeder lanes share capability:feeder, "
            "all agents on one reader principal share it.",
            f"{UNSCOPED} = the route acts on the whole allowed set (fetch by id, stats, namespace list, edges) "
            "or names no namespace.",
            "No history before the first record; /health is not recorded.",
        ],
    }


def render_text(report: dict[str, Any]) -> str:
    out = [
        "Scrutator access report  "
        f"{report['window']['from']} .. {report['window']['to']}  ({report['window']['days']} d)",
        f"directory: {report['directory']}   records: {report['records']}",
        "",
    ]
    for name, s in report["identities"].items():
        out.append(
            f"{name}\n  requests {s['requests']}  reads {s['reads']}  writes {s['writes']}  denied {s['denied']}"
            f"  first {s['first_seen']}  last {s['last_seen']}"
        )
        out.append("  namespaces: " + ", ".join(f"{k}={v}" for k, v in s["namespaces"].items()))
        out.append("  sources:    " + ", ".join(f"{k}={v}" for k, v in s["sources"].items()))
        out.append("  statuses:   " + ", ".join(f"{k}={v}" for k, v in s["statuses"].items()))
        out.append("")
    out.append(f"Known identities with ZERO use in the window ({report['known_identities_note']}):")
    out.extend(f"  {name}" for name in report["zero_use"]) if report["zero_use"] else out.append("  (none)")
    out.append("")
    out.append("Limits:")
    out.extend(f"  - {line}" for line in report["limits"])
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only report over the Scrutator access log")
    parser.add_argument("--days", type=float, default=1.0, help="window length in days (default 1)")
    parser.add_argument("--dir", default=None, help="access-log directory (default: SCRUTATOR_ACCESS_LOG_DIR)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--no-db", action="store_true", help="do not read reader principals from the database")
    args = parser.parse_args(argv)

    directory = args.dir
    if directory is None:
        from scrutator.config import settings

        directory = settings.access_log_dir
    if not directory or not Path(directory).is_dir():
        print(f"access log directory not found: {directory!r}", file=sys.stderr)
        return 2
    report = build_report(Path(directory), args.days, use_db=not args.no_db)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
