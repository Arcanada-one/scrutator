#!/usr/bin/env python3
"""Root-operated, project-only bootstrap for the Knowledge Contract namespace.

Run inside the dedicated project container. Dry-run is the default; --live uses
Scrutator's normal repository API to create exactly one namespace, then reads
it back. This prepares the existing grant_namespace.py interface; it grants no
principal and touches no shared Scrutator tenant.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from urllib.parse import urlsplit

NAMESPACE = "knowledge-contract"
INTROSPECT_URL = "https://auth.arcanada.ai/api/v1/api-keys/knowledge-contract/introspect"
AUDIENCE = "urn:arcanada:scrutator:knowledge-contract"


class Refusal(Exception):
    """The process is outside the dedicated project profile."""


def validate_profile(settings) -> None:
    dsn = urlsplit(settings.database_url)
    if (
        dsn.scheme not in {"postgres", "postgresql"}
        or dsn.username != "kc_scrutator_app"
        or dsn.path != "/knowledge_contract"
        or settings.auth_arcana_introspect_url != INTROSPECT_URL
        or settings.auth_service_audience != AUDIENCE
        or settings.auth_service_scope != "kb:knowledge-contract.read"
        or settings.feeder_namespaces != NAMESPACE
        or not settings.auth_enforce
    ):
        raise Refusal("dedicated Knowledge Contract Scrutator profile is not active")


async def run(live: bool, settings, repository) -> int:
    validate_profile(settings)
    existing = await repository.get_namespace_id(NAMESPACE)
    if existing is not None:
        print(f"KC_NAMESPACE_BOOTSTRAP_PASS state=present namespace={NAMESPACE} id={existing}")
        return 0
    if not live:
        print(f"KC_NAMESPACE_BOOTSTRAP_PREVIEW state=absent namespace={NAMESPACE}; rerun with --live")
        return 0
    created = await repository.upsert_namespace(NAMESPACE)
    readback = await repository.get_namespace_id(NAMESPACE)
    if readback is None or readback != created:
        raise RuntimeError("namespace readback mismatch")
    print(f"KC_NAMESPACE_BOOTSTRAP_PASS state=created namespace={NAMESPACE} id={readback}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="create the fixed namespace if absent")
    args = parser.parse_args(argv)
    from scrutator.config import settings
    from scrutator.db import repository

    try:
        return asyncio.run(run(args.live, settings, repository))
    except Refusal as exc:
        print(f"KC_NAMESPACE_BOOTSTRAP_REFUSED reason={exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"KC_NAMESPACE_BOOTSTRAP_FAIL error_type={type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
