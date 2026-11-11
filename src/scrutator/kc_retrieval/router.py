"""Read-only FastAPI router: GET /v1/kc/retrieve (KC-PIPE-S3).

No write routes are registered on this router. Auth uses the same Auth Arcana
tenant choke-point as /v1/search and /v1/fetch. Size and rate budgets are
enforced before bytes leave the process.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from scrutator.auth.dependency import require_tenant_context
from scrutator.auth.models import TenantContext
from scrutator.config import settings
from scrutator.kc_retrieval.models import KcRetrieveQuery, KcRetrieveResponse
from scrutator.kc_retrieval.service import RetrievalPolicy, retrieve
from scrutator.request_rate_limit import SlidingWindowRateLimiter

router = APIRouter(prefix="/v1/kc", tags=["knowledge-contract-retrieval"])

_rate_limiter = SlidingWindowRateLimiter(limit=60, window_seconds=60.0)


def reset_rate_limiter_for_tests(
    *,
    limit: int | None = None,
    window_seconds: float | None = None,
) -> SlidingWindowRateLimiter:
    """Replace the process-local limiter (test harness only)."""
    global _rate_limiter
    _rate_limiter = SlidingWindowRateLimiter(
        limit=limit if limit is not None else settings.kc_retrieval_rate_limit,
        window_seconds=(window_seconds if window_seconds is not None else settings.kc_retrieval_rate_window_seconds),
    )
    return _rate_limiter


def _policy() -> RetrievalPolicy:
    registry_path = settings.kc_retrieval_registry_path
    if not registry_path:
        raise HTTPException(status_code=503, detail="location registry unavailable")
    public_raw = settings.kc_retrieval_public_object_ids
    public_ids = frozenset(item.strip() for item in public_raw.split(",") if item.strip())
    return RetrievalPolicy(
        registry_path=Path(registry_path),
        max_bytes=settings.kc_retrieval_max_bytes,
        public_object_ids=public_ids,
        namespace=settings.kc_retrieval_namespace,
    )


@router.get("/retrieve", response_model=KcRetrieveResponse)
async def retrieve_endpoint(
    response: Response,
    id: str = Query(..., min_length=66, max_length=200),
    view: str = Query(default="original"),
    offset_start: int | None = Query(default=None, ge=0),
    offset_end: int | None = Query(default=None, ge=0),
    ctx: TenantContext = Depends(require_tenant_context),
) -> KcRetrieveResponse:
    """Resolve a source/package id to verified bytes via the location registry."""
    if not settings.kc_retrieval_enabled:
        raise HTTPException(status_code=503, detail="kc retrieval disabled")

    retry_after = _rate_limiter.try_acquire(settings.kc_retrieval_namespace)
    if retry_after is not None:
        response.headers["Retry-After"] = str(retry_after)
        raise HTTPException(status_code=429, detail="rate budget exceeded")

    try:
        query = KcRetrieveQuery.model_validate(
            {
                "id": id,
                "view": view,
                "offset_start": offset_start,
                "offset_end": offset_end,
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail="invalid retrieve query") from exc

    return retrieve(
        query,
        allowed_namespaces=ctx.allowed_namespace_names,
        policy=_policy(),
    )
