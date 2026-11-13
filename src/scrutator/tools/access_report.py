"""Read-only report over the durable access log (Muneral 4e0054f0).

Answers the operator's question "who uses the knowledge base, which namespaces, how": one row per
(identity, User-Agent, proxy-reported client address) with requests, read/write split, namespaces touched,
denied count and first/last seen; then the known identities with zero use, the namespaces nobody read in
the window and the namespaces that have no reader grant at all.

It reads the access-log files only. The known reader principals and the namespace list come from read-only
SELECTs (``principal_namespace_grants``, ``namespaces``) and are skipped, with a note, when the database
cannot be reached or ``--no-db`` is given.

Usage (inside the service container):
    python -m scrutator.tools.access_report --days 1
    python -m scrutator.tools.access_report --days 30 --json

Limits repeated in the report: a shared secret is one identity (``ua`` and ``fwd`` separate agents and hosts
only as far as they differ; ``fwd`` is reported by the proxy, not verified); routes that act on the whole
allowed set name no namespace (``(unscoped)``).
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
NONE_LABEL = "-"


@dataclass
class GroupStats:
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


def aggregate(records) -> tuple[dict[tuple[str, str, str], GroupStats], int, set[str]]:
    """Group by (identity, ua, fwd). Also return the namespaces that an authenticated READ named."""
    groups: dict[tuple[str, str, str], GroupStats] = defaultdict(GroupStats)
    read_namespaces: set[str] = set()
    total = 0
    for record in records:
        total += 1
        key = (record.get("identity") or "none", record.get("ua") or NONE_LABEL, record.get("fwd") or NONE_LABEL)
        item = groups[key]
        item.requests += 1
        mode = record.get("mode")
        if mode == "write":
            item.writes += 1
        elif mode == "read":
            item.reads += 1
        denied = record.get("auth") == "denied" or record.get("status") in {401, 403}
        if denied:
            item.denied += 1
        ts = record.get("ts", "")
        item.first_seen = min(item.first_seen or ts, ts)
        item.last_seen = max(item.last_seen, ts)
        for name in record.get("namespaces") or [UNSCOPED]:
            item.namespaces[name] += 1
            if mode == "read" and record.get("auth") == "ok" and not denied and name != UNSCOPED:
                read_namespaces.add(name)
        if record.get("src"):
            item.sources[record["src"]] += 1
        item.statuses[str(record.get("status"))] += 1
    return dict(groups), total, read_namespaces


async def _database_lists() -> tuple[list[str], list[tuple[str, bool]]]:
    from scrutator.db.connection import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        principals = await conn.fetch("SELECT DISTINCT principal_id FROM principal_namespace_grants ORDER BY 1")
        namespaces = await conn.fetch(
            "SELECT n.name, EXISTS (SELECT 1 FROM principal_namespace_grants g WHERE g.namespace_id = n.id) AS granted "
            "FROM namespaces n ORDER BY n.name"
        )
    return (
        [f"reader:{row['principal_id']}" for row in principals],
        [(row["name"], bool(row["granted"])) for row in namespaces],
    )


def database_lists(use_db: bool) -> tuple[list[str] | None, list[tuple[str, bool]] | None, str]:
    if not use_db:
        return None, None, "not read (--no-db)"
    try:
        principals, namespaces = asyncio.run(_database_lists())
        return principals, namespaces, "from principal_namespace_grants and namespaces"
    except Exception as exc:  # report without them rather than fail
        return None, None, f"NOT read: database unreachable ({exc.__class__.__name__})"


def build_report(directory: Path, days: float, use_db: bool, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    since = now - timedelta(days=days)
    groups, total, read_namespaces = aggregate(iter_records(directory, since))
    principals, namespaces, db_note = database_lists(use_db)
    seen_identities = {identity for identity, _ua, _fwd in groups}
    known = list(CAPABILITY_KINDS) + (principals or [])
    ordered = sorted(groups.items(), key=lambda kv: (-kv[1].requests, kv[0]))
    identity_totals: dict[str, int] = defaultdict(int)
    for (identity, _ua, _fwd), stats in groups.items():
        identity_totals[identity] += stats.requests
    return {
        "window": {
            "from": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "days": days,
        },
        "directory": str(directory),
        "records": total,
        "groups": [
            {
                "identity": identity,
                "ua": ua,
                "fwd": fwd,
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
            for (identity, ua, fwd), s in ordered
        ],
        "identity_totals": dict(sorted(identity_totals.items(), key=lambda kv: -kv[1])),
        "database": db_note,
        "zero_use_identities": sorted(set(known) - seen_identities),
        "namespaces_without_reads_in_window": (
            sorted(name for name, _granted in namespaces if name not in read_namespaces)
            if namespaces is not None
            else None
        ),
        "namespaces_without_any_reader_grant": (
            sorted(name for name, granted in namespaces if not granted) if namespaces is not None else None
        ),
        "limits": [
            "A shared secret is one identity: all feeder lanes share capability:feeder, all agents on one reader "
            "principal share it; ua and fwd separate them only as far as they differ.",
            "fwd is the client address the proxy reports (CF-Connecting-IP / X-Forwarded-For), recorded only when "
            "the direct peer is loopback or private; reported by the proxy, not verified.",
            f"{UNSCOPED} = the route acts on the whole allowed set (fetch by id, stats, namespace list, edges) "
            "or names no namespace; such reads do not count towards 'read in the window'.",
            "No history before the first record; /health is not recorded.",
        ],
    }


def render_text(report: dict[str, Any]) -> str:
    w = report["window"]
    out = [
        f"Scrutator access report  {w['from']} .. {w['to']}  ({w['days']} d)",
        f"directory: {report['directory']}   records: {report['records']}   database lists: {report['database']}",
        "",
        "identity x user-agent x proxy-reported client",
    ]
    for g in report["groups"]:
        out.append(f"{g['identity']}  ua={g['ua']}  fwd={g['fwd']}")
        out.append(
            f"  requests {g['requests']}  reads {g['reads']}  writes {g['writes']}  denied {g['denied']}"
            f"  first {g['first_seen']}  last {g['last_seen']}"
        )
        out.append("  namespaces: " + ", ".join(f"{k}={v}" for k, v in g["namespaces"].items()))
        out.append(
            "  src: "
            + ", ".join(f"{k}={v}" for k, v in g["sources"].items())
            + "   status: "
            + ", ".join(f"{k}={v}" for k, v in g["statuses"].items())
        )
    out.append("")

    def block(title: str, values: list[str] | None) -> None:
        out.append(title)
        if values is None:
            out.append("  (not available: database lists " + report["database"] + ")")
        elif values:
            out.extend(f"  {v}" for v in values)
        else:
            out.append("  (none)")
        out.append("")

    block("Known identities with ZERO use in the window:", report["zero_use_identities"])
    block("Namespaces nobody READ in the window (named reads only):", report["namespaces_without_reads_in_window"])
    block(
        "Namespaces with NO reader grant at all (nobody can read them):", report["namespaces_without_any_reader_grant"]
    )
    out.append("Limits:")
    out.extend(f"  - {line}" for line in report["limits"])
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only report over the Scrutator access log")
    parser.add_argument("--days", type=float, default=1.0, help="window length in days (default 1)")
    parser.add_argument("--dir", default=None, help="access-log directory (default: SCRUTATOR_ACCESS_LOG_DIR)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--no-db", action="store_true", help="do not read principals and namespaces from the database")
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
