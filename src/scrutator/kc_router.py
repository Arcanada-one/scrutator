"""KB-CONS-01-N: dedicated private-generation read routes.

Every operation is bound to a server-held lease (never to request project/generation fields) and
re-checks lease state + current authority at start, before any external model sees content, and at the
response/use commit point. With no real authority backend configured every route answers 503
``CURRENT_AUTHORITY_UNAVAILABLE`` — by design. Graph/LTM views answer ``UNSUPPORTED_PRIVATE_OPERATION``.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from scrutator.auth.dependency import require_private_tenant_context, resolve_namespace_selector
from scrutator.auth.models import TenantContext
from scrutator.db import kc_generations as kc
from scrutator.db.connection import get_pool
from scrutator.db.models import SearchResponse
from scrutator.search.searcher import search

# Fully qualified decorator paths on purpose: the graph builder reads the decorator text and ignores
# APIRouter(prefix=...), so a prefix would make these routes collide with same-named LTM routes.
router = APIRouter(tags=["kc-private-generation"])

_STATUS = {
    "CURRENT_AUTHORITY_UNAVAILABLE": 503,
    "LOCK_FAILURE": 503,
    "NOT_FOUND": 404,
    "LEASE_NOT_FOUND": 404,
    "SCOPE_NOT_FOUND": 404,
    "LEASE_EXPIRED": 410,
    "NO_SELECTION": 409,
    "GENERATION_NOT_SERVABLE": 409,
    "CONTENT_MISMATCH": 409,
    "LIMIT_EXCEEDED": 429,
    "TOKEN_INVALID": 422,
    "UNSUPPORTED_PRIVATE_OPERATION": 501,
}


def _http(exc: kc.KcGenerationError) -> HTTPException:
    # Only the stable code is returned: no locators, bodies or existence hints.
    return HTTPException(status_code=_STATUS.get(exc.code, 403), detail=exc.code)


class BeginReadRequest(BaseModel):
    namespace: str
    project: str
    purpose: str
    operation_id: str


class LeaseRequest(BaseModel):
    lease_id: str


class KcSearchRequest(LeaseRequest):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=10, ge=1, le=50)
    include_content: bool = True
    group_by: Literal["document", "section"] | None = None


class KcFetchRequest(LeaseRequest):
    member_id: str


class KcChunkRequest(LeaseRequest):
    chunk_id: str


@router.post("/v1/kc/read/begin")
async def begin(request: BeginReadRequest, ctx: TenantContext = Depends(require_private_tenant_context)) -> dict:
    namespace_id = await resolve_namespace_selector(ctx, request.namespace)
    pool = await get_pool()
    project_id = await pool.fetchval(
        "SELECT id FROM projects WHERE namespace_id = $1 AND name = $2", namespace_id, request.project
    )
    scope = await kc.find_scope(pool, namespace_id, project_id, request.purpose) if project_id else None
    if scope is None:
        raise HTTPException(status_code=404, detail="SCOPE_NOT_FOUND")  # never creates a scope/project
    try:
        rc = await kc.begin_read(pool, scope.scope_id, principal_id=ctx.principal_id, operation_id=request.operation_id)
    except kc.KcGenerationError as exc:
        raise _http(exc) from exc
    return {
        "lease_id": rc.lease_id,
        "generation_id": rc.generation_id,
        "manifest_sha256": rc.manifest_sha256,
        "selection_incarnation": rc.selection_incarnation,
        "selection_revision": str(rc.selection_revision),
    }


async def _context(request: LeaseRequest, ctx: TenantContext) -> kc.GenerationReadContext:
    try:
        return await kc.load_context(await get_pool(), request.lease_id, principal_id=ctx.principal_id)
    except kc.KcGenerationError as exc:
        raise _http(exc) from exc


@router.post("/v1/kc/read/search", response_model=SearchResponse)
async def kc_search(
    request: KcSearchRequest, ctx: TenantContext = Depends(require_private_tenant_context)
) -> SearchResponse:
    rc = await _context(request, ctx)
    pool = await get_pool()

    async def gate() -> None:
        await kc.authorize_use(pool, rc, principal_id=ctx.principal_id)

    try:
        await gate()
        return await search(
            query=request.query,
            namespace_id=rc.namespace_id,
            limit=request.limit,
            include_content=request.include_content,
            group_by=request.group_by,
            generation=rc.eligibility,
            pre_disclosure_check=gate,
        )
    except kc.KcGenerationError as exc:
        raise _http(exc) from exc


@router.post("/v1/kc/read/fetch")
async def kc_fetch(request: KcFetchRequest, ctx: TenantContext = Depends(require_private_tenant_context)) -> dict:
    rc = await _context(request, ctx)
    try:
        return await kc.read_member_exact(
            await get_pool(), rc, principal_id=ctx.principal_id, member_id=request.member_id
        )
    except kc.KcGenerationError as exc:
        raise _http(exc) from exc


@router.post("/v1/kc/read/chunk")
async def kc_chunk(request: KcChunkRequest, ctx: TenantContext = Depends(require_private_tenant_context)) -> dict:
    rc = await _context(request, ctx)
    try:
        return await kc.read_chunk(await get_pool(), rc, principal_id=ctx.principal_id, chunk_id=request.chunk_id)
    except kc.KcGenerationError as exc:
        raise _http(exc) from exc


@router.post("/v1/kc/read/related")
async def kc_related(request: KcChunkRequest, ctx: TenantContext = Depends(require_private_tenant_context)) -> list:
    rc = await _context(request, ctx)
    try:
        return await kc.read_related(await get_pool(), rc, principal_id=ctx.principal_id, chunk_id=request.chunk_id)
    except kc.KcGenerationError as exc:
        raise _http(exc) from exc


@router.post("/v1/kc/read/graph")
async def kc_graph(request: LeaseRequest, ctx: TenantContext = Depends(require_private_tenant_context)) -> dict:
    """Entity graph / LTM views merge sources; refused for private generations (no body leaked)."""
    try:
        kc.refuse_unsupported_private_view("entity graph / LTM")
    except kc.KcGenerationError as exc:
        raise _http(exc) from exc
    return {}
