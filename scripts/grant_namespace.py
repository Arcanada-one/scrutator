#!/usr/bin/env python3
"""Operator tool: grant ONE principal read access to ONE namespace.

`db/schema.sql` has referenced this script since SRCH-0023 as the out-of-band seeder for
`principal_namespace_grants`, the FK-only cache that `auth/rebac_client.py` falls back to when
OpenFGA is unconfigured. The reference existed; the script did not. This is it.

The table is the ONLY place a principal's namespace scope is expressed. `verify_ltm_m2m_token`
pins issuer, audience, scope (`kb:ltm.read`) and an exact 300-second token lifetime, but every
M2M reader presents the same scope — so **which namespaces a principal may read is decided here
and nowhere else**. A tool that makes a broad grant easy is a tool that will eventually make one.

Narrow by construction:

- **One namespace per invocation.** No list, no repeat flag, no comma-separated value. Granting
  two namespaces takes two audited commands.
- **No wildcards.** `*`, `%`, `all`, `any` and empty are refused before any query is built, so a
  wildcard cannot arrive via a shell glob or a copied runbook line.
- **The namespace must already exist.** The FK would reject a dangling id anyway; refusing early
  turns a foreign-key traceback into a sentence, and refuses to create a namespace as a
  side effect of granting one.
- **Dry-run is the default.** `--live` is required to write, mirroring `kb-feeder`'s idiom.
- **Read-only by nature.** The grant conveys namespace visibility; it confers no write, admin, or
  cross-namespace capability, because no write path consults this table.
- **Revocable.** `--revoke` removes exactly one grant, so a credential handed out for a bounded
  purpose can be withdrawn without a migration.

Usage::

    grant_namespace.py --principal kb-observer --namespace wiki            # preview (default)
    grant_namespace.py --principal kb-observer --namespace wiki --live     # write
    grant_namespace.py --principal kb-observer --revoke --namespace wiki --live
    grant_namespace.py --list                                              # audit, read-only

Exit codes: 0 success (including a no-op) · 1 refusal · 2 usage/connection error.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys

# A principal id is an OIDC client_id or a service principal: conservative, no separators that
# could be meaningful to a downstream consumer of the audit line.
PRINCIPAL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,126}$")
NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")

# Refused outright, case-insensitively, before any query is constructed. These are the shapes a
# broad grant actually arrives as: a shell glob, a SQL wildcard, or a well-meaning "all".
WILDCARD_TOKENS = frozenset({"*", "%", "_", "all", "any", "everything", "-", "."})


class Refusal(Exception):
    """A request this tool declines to make. Never a traceback."""


def validate_principal(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        raise Refusal("principal is empty")
    if candidate.lower() in WILDCARD_TOKENS:
        raise Refusal(f"principal {value!r} is a wildcard; grants are per-principal")
    if not PRINCIPAL_RE.fullmatch(candidate):
        raise Refusal(f"principal {value!r} is not a valid principal id")
    return candidate


def validate_namespace(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        raise Refusal("namespace is empty")
    if candidate.lower() in WILDCARD_TOKENS:
        raise Refusal(f"namespace {value!r} is a wildcard; grant exactly one namespace")
    if any(sep in candidate for sep in (",", " ", ";", "|")):
        raise Refusal(f"namespace {value!r} looks like a list; grant exactly one namespace per run")
    if not NAMESPACE_RE.fullmatch(candidate):
        raise Refusal(f"namespace {value!r} is not a valid namespace name")
    return candidate


async def resolve_namespace_id(conn, namespace: str) -> int:
    namespace_id = await conn.fetchval("SELECT id FROM namespaces WHERE name = $1", namespace)
    if namespace_id is None:
        raise Refusal(
            f"namespace {namespace!r} does not exist; this tool grants access to a namespace, it never creates one"
        )
    return int(namespace_id)


async def existing_grant(conn, principal: str, namespace_id: int) -> bool:
    found = await conn.fetchval(
        "SELECT 1 FROM principal_namespace_grants WHERE principal_id = $1 AND namespace_id = $2",
        principal,
        namespace_id,
    )
    return found is not None


async def list_grants(conn) -> list[dict]:
    rows = await conn.fetch(
        "SELECT g.principal_id, n.name AS namespace, g.granted_at "
        "FROM principal_namespace_grants g JOIN namespaces n ON n.id = g.namespace_id "
        "ORDER BY g.principal_id, n.name"
    )
    return [dict(row) for row in rows]


async def apply_grant(conn, principal: str, namespace_id: int) -> bool:
    """Insert exactly one grant. Returns whether a row was created."""
    result = await conn.execute(
        "INSERT INTO principal_namespace_grants (principal_id, namespace_id) VALUES ($1, $2) "
        "ON CONFLICT (principal_id, namespace_id) DO NOTHING",
        principal,
        namespace_id,
    )
    return result.endswith("1")


async def apply_revoke(conn, principal: str, namespace_id: int) -> bool:
    """Delete exactly one grant. Returns whether a row was removed."""
    result = await conn.execute(
        "DELETE FROM principal_namespace_grants WHERE principal_id = $1 AND namespace_id = $2",
        principal,
        namespace_id,
    )
    return not result.endswith(" 0")


async def run(args: argparse.Namespace) -> int:
    from scrutator.db.connection import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        if args.list:
            grants = await list_grants(conn)
            if not grants:
                print("no grants; every principal resolves to the empty set and is denied")
                return 0
            for grant in grants:
                print(f"{grant['principal_id']}\t{grant['namespace']}\t{grant['granted_at']}")
            print(f"({len(grants)} grant(s))")
            return 0

        principal = validate_principal(args.principal)
        namespace = validate_namespace(args.namespace)
        namespace_id = await resolve_namespace_id(conn, namespace)
        present = await existing_grant(conn, principal, namespace_id)
        verb = "revoke" if args.revoke else "grant"

        if not args.live:
            state = "present" if present else "absent"
            if args.revoke:
                would = "remove it" if present else "do nothing"
            else:
                would = "do nothing" if present else "create it"
            print(f"DRY RUN: {verb} {principal} -> {namespace} (id {namespace_id}); grant is {state}; would {would}")
            print("re-run with --live to apply")
            return 0

        changed = await (apply_revoke if args.revoke else apply_grant)(conn, principal, namespace_id)
        outcome = "applied" if changed else "no change (already in the requested state)"
        print(f"{verb}: {principal} -> {namespace} (id {namespace_id}): {outcome}")
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--principal", help="exactly one principal id (OIDC client_id)")
    parser.add_argument("--namespace", help="exactly one namespace name")
    parser.add_argument("--revoke", action="store_true", help="remove the grant instead of creating it")
    parser.add_argument("--live", action="store_true", help="apply the change (default is a dry run)")
    parser.add_argument("--list", action="store_true", help="print every current grant and exit (read-only)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.list and not (args.principal and args.namespace):
        print("ERROR: --principal and --namespace are required (or use --list)", file=sys.stderr)
        return 2
    if args.list and (args.principal or args.namespace or args.revoke or args.live):
        print("ERROR: --list is read-only and takes no other arguments", file=sys.stderr)
        return 2
    try:
        return asyncio.run(run(args))
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # connection/driver failure — never a traceback at the operator
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
