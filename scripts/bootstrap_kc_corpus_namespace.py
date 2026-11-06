#!/usr/bin/env python3
"""Root-operated, instance-only bootstrap for the kc-corpus namespace (KB-CONS-02).

Run inside the dedicated kc-corpus container. Dry-run is the default; --live uses Scrutator's normal
repository API to create exactly one namespace, then reads it back. It grants no principal; the existing
scripts/grant_namespace.py is the grant interface. It refuses to run in any process that is not the
dedicated kc-corpus profile, so it cannot create the namespace in the shared or Knowledge Contract DBs.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from urllib.parse import urlsplit

NAMESPACE = "kc-corpus"
DATABASE = "/kc_corpus"
ROLE = "kc_corpus_scrutator_app"


class Refusal(Exception):
    """The process is outside the dedicated kc-corpus profile."""


def validate_profile(settings) -> None:
    dsn = urlsplit(settings.database_url)
    if (
        dsn.scheme not in {"postgres", "postgresql"}
        or dsn.username != ROLE
        or dsn.path != DATABASE
        or settings.feeder_namespaces != NAMESPACE
        or not settings.auth_enforce
        or not settings.evidence_exact_bytes
        or settings.evidence_exact_namespaces != [NAMESPACE]
        or settings.evidence_exact_required_namespaces != [NAMESPACE]
        or settings.auth_arcana_introspect_url
        or settings.project_write_namespace
    ):
        raise Refusal("dedicated kc-corpus Scrutator profile is not active")


async def run(live: bool, settings, repository) -> int:
    validate_profile(settings)
    existing = await repository.get_namespace_id(NAMESPACE)
    if existing is not None:
        print(f"KC_CORPUS_NAMESPACE_BOOTSTRAP_PASS state=present namespace={NAMESPACE} id={existing}")
        return 0
    if not live:
        print(f"KC_CORPUS_NAMESPACE_BOOTSTRAP_PREVIEW state=absent namespace={NAMESPACE}; rerun with --live")
        return 0
    created = await repository.upsert_namespace(NAMESPACE)
    readback = await repository.get_namespace_id(NAMESPACE)
    if readback is None or readback != created:
        raise RuntimeError("namespace readback mismatch")
    print(f"KC_CORPUS_NAMESPACE_BOOTSTRAP_PASS state=created namespace={NAMESPACE} id={readback}")
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
        print(f"KC_CORPUS_NAMESPACE_BOOTSTRAP_REFUSED reason={exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"KC_CORPUS_NAMESPACE_BOOTSTRAP_FAIL error_type={type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
