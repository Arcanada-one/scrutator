"""Shared test fixtures (SRCH-0023): FastAPI auth-dependency override helper.

Pre-SRCH-0023 route tests exercised `/v1/*` and `/v1/ltm/*` endpoints with no credential,
relying on the (now-closed) unauthenticated-default behaviour. Those tests use
`override_tenant_context` to simulate an authenticated caller scoped to a known namespace,
matching the endpoints' new `Depends(require_tenant_context)` requirement.
"""

from __future__ import annotations

from contextlib import contextmanager

from scrutator.auth.dependency import require_tenant_context
from scrutator.auth.models import TenantContext


def make_tenant_context(
    namespace_ids: frozenset[int] = frozenset({1}),
    namespace_names: frozenset[str] = frozenset({"arcanada"}),
    principal_id: str = "test-principal",
) -> TenantContext:
    return TenantContext(
        principal_id=principal_id,
        principal_type="service",
        allowed_namespace_ids=namespace_ids,
        allowed_namespace_names=namespace_names,
    )


@contextmanager
def override_tenant_context(app, ctx: TenantContext | None = None):
    """Override the `require_tenant_context` FastAPI dependency for the duration of the block."""
    ctx = ctx or make_tenant_context()
    app.dependency_overrides[require_tenant_context] = lambda: ctx
    try:
        yield ctx
    finally:
        app.dependency_overrides.pop(require_tenant_context, None)
