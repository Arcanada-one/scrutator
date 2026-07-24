"""Loopback-only benchmark wrapper around the deployed Scrutator FastAPI app.

The wrapper bypasses JWT transport because the benchmark host does not hold the
reader client secret.  It does not bypass authorization: every request receives
a TenantContext built from the live grant rows for the explicitly configured
benchmark principal.
"""

from __future__ import annotations

import hashlib
import os

from fastapi import Depends, HTTPException

from scrutator.auth.dependency import require_tenant_context
from scrutator.auth.models import TenantContext
from scrutator.auth.rebac_client import resolve_allowed_namespaces
from scrutator.db.connection import get_pool
from scrutator.health import app

_BENCHMARK_NAMESPACE = "arcanada"


async def build_benchmark_context() -> TenantContext:
    principal = os.environ.get("SCRUTATOR_BENCHMARK_PRINCIPAL", "")
    if not principal:
        raise RuntimeError("SCRUTATOR_BENCHMARK_PRINCIPAL is required")

    allowed_ids = await resolve_allowed_namespaces(principal)
    pool = await get_pool()
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            "SELECT id, name FROM namespaces WHERE id = ANY($1::int[]) ORDER BY id",
            list(allowed_ids),
        )
    allowed_names = frozenset(row["name"] for row in rows)
    if _BENCHMARK_NAMESPACE not in allowed_names:
        raise RuntimeError(f"principal {principal!r} is not granted namespace {_BENCHMARK_NAMESPACE!r}")

    return TenantContext(
        principal_id=principal,
        principal_type="service",
        allowed_namespace_ids=allowed_ids,
        allowed_namespace_names=allowed_names,
    )


app.dependency_overrides[require_tenant_context] = build_benchmark_context


@app.get("/__benchmark/fingerprint")
async def benchmark_fingerprint(
    context: TenantContext = Depends(build_benchmark_context),
) -> dict[str, str | int]:
    pool = await get_pool()
    async with pool.acquire() as connection:
        namespace = await connection.fetchrow(
            "SELECT id FROM namespaces WHERE name = $1",
            _BENCHMARK_NAMESPACE,
        )
        if namespace is None or namespace["id"] not in context.allowed_namespace_ids:
            raise HTTPException(status_code=403, detail="benchmark namespace is not authorized")
        rows = await connection.fetch(
            """
            SELECT id::text AS id, content_hash, updated_at
            FROM chunks
            WHERE namespace_id = $1
            ORDER BY id
            """,
            namespace["id"],
        )

    digest = hashlib.sha256()
    for row in rows:
        updated_at = row["updated_at"].isoformat() if row["updated_at"] is not None else ""
        digest.update(f"{row['id']}\0{row['content_hash'] or ''}\0{updated_at}\n".encode())
    return {
        "namespace": _BENCHMARK_NAMESPACE,
        "namespace_id": namespace["id"],
        "chunk_count": len(rows),
        "sha256": digest.hexdigest(),
    }
