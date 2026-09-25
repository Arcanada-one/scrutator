"""Every /v1 route declares exactly one auth policy, and the manifest is exhaustive.

A2-308 extended this from "has a policy" to "has the RIGHT policy": before it, every
non-machine route shared one dependency (`require_tenant_context`) that checked identity
and namespace grants but never whether the credential may write, so `POST /v1/edges`,
`/v1/edges/by-path` and `DELETE /v1/edges` accepted a `kb:ltm.read` token.

An undeclared route now fails this test rather than silently defaulting to read — adding a
route forces the author to state which of the three classes it belongs to.
"""

from fastapi.routing import APIRoute

from scrutator.health import app
from scrutator.ltm.router import router as ltm_router

# Routes gated by a dedicated machine credential (a header token), not by a bearer scope.
MACHINE_ROUTES = {
    ("GET", "/v1/index/capability"): "require_feeder_capability",
    ("GET", "/v1/index/rollback-capability"): "require_rollback_capability",
    ("POST", "/v1/index"): "require_feeder_capability",
    ("POST", "/v1/index/capability-projection"): "require_capability_projection_capability",
    ("POST", "/v1/index/batch"): "require_feeder_capability",
    ("DELETE", "/v1/index"): "require_rollback_capability",
    ("POST", "/v1/ltm/ingest"): "require_ltm_writer_capability",
    ("DELETE", "/v1/ltm/source"): "require_ltm_writer_capability",
}

# Routes that change stored state (or spend a budget) — bearer + `kb:ltm.write`.
WRITE_ROUTES = {
    ("POST", "/v1/namespaces"),
    ("POST", "/v1/edges"),
    ("POST", "/v1/edges/by-path"),
    ("DELETE", "/v1/edges"),
    ("POST", "/v1/memories"),
    ("POST", "/v1/memories/bulk"),
    ("DELETE", "/v1/memories"),
    ("POST", "/v1/ltm/reflect"),
}

# Read-only routes — bearer + namespace grants, unchanged by A2-308. POST here is a
# request-body verb, not a mutation: /v1/chunk is pure computation, /v1/dream/analyze only
# SELECTs (its inserts live behind /v1/edges), and the recall paths only search.
READ_ROUTES = {
    ("POST", "/v1/chunk"),
    ("POST", "/v1/search"),
    ("POST", "/v1/fetch"),
    ("GET", "/v1/navigate/outline"),
    ("GET", "/v1/navigate/section"),
    ("GET", "/v1/chunks"),
    ("GET", "/v1/namespaces"),
    ("GET", "/v1/stats"),
    ("POST", "/v1/dream/analyze"),
    ("GET", "/v1/edges/{chunk_id}"),
    ("POST", "/v1/memories/recall"),
    ("GET", "/v1/memories/stats"),
    ("GET", "/v1/ltm/jobs/{job_id}"),
    ("POST", "/v1/ltm/recall"),
    ("GET", "/v1/ltm/entities"),
    ("GET", "/v1/ltm/graph"),
    ("GET", "/v1/ltm/meta_facts"),
    ("GET", "/v1/ltm/events"),
}


def _dependency_names(route: APIRoute) -> set[str]:
    """Every dependency on the route, including nested ones.

    `require_ltm_write_scope` depends on `require_tenant_context`, so a flat read of
    `route.dependant.dependencies` would miss the identity dependency underneath it.
    """
    names: set[str] = set()
    stack = list(route.dependant.dependencies)
    while stack:
        dependency = stack.pop()
        if dependency.call is not None:
            names.add(dependency.call.__name__)
        stack.extend(dependency.dependencies)
    return names


def _inventory():
    seen = {}
    # This FastAPI version retains included routers as a nested route object,
    # so inventory both the app routes and the included LTM router explicitly.
    for route in [*app.routes, *ltm_router.routes]:
        if not isinstance(route, APIRoute) or not route.path.startswith("/v1/"):
            continue
        for method in route.methods:
            seen[(method, route.path)] = _dependency_names(route)
    return seen


def test_manifest_covers_every_v1_route():
    """No route may exist without a declared class — including one added tomorrow."""
    declared = set(MACHINE_ROUTES) | WRITE_ROUTES | READ_ROUTES
    live = set(_inventory())
    assert live - declared == set(), f"undeclared route(s): {sorted(live - declared)}"
    assert declared - live == set(), f"manifest lists route(s) that do not exist: {sorted(declared - live)}"


def test_machine_routes_use_their_own_credential_and_not_the_bearer_path():
    inventory = _inventory()
    for key, machine_dependency in MACHINE_ROUTES.items():
        names = inventory[key]
        assert machine_dependency in names, key
        assert "require_tenant_context" not in names, key
        assert "require_ltm_write_scope" not in names, key


def test_write_routes_require_the_write_scope():
    inventory = _inventory()
    for key in WRITE_ROUTES:
        names = inventory[key]
        assert "require_ltm_write_scope" in names, key
        # identity still resolved underneath — the scope check does not replace it
        assert "require_tenant_context" in names, key
        assert not names.intersection(MACHINE_ROUTES.values()), key


def test_read_routes_are_not_gated_behind_the_write_scope():
    inventory = _inventory()
    for key in READ_ROUTES:
        names = inventory[key]
        assert "require_tenant_context" in names, key
        assert "require_ltm_write_scope" not in names, key
        assert not names.intersection(MACHINE_ROUTES.values()), key


def test_the_three_classes_are_disjoint():
    assert set(MACHINE_ROUTES).isdisjoint(WRITE_ROUTES)
    assert set(MACHINE_ROUTES).isdisjoint(READ_ROUTES)
    assert WRITE_ROUTES.isdisjoint(READ_ROUTES)
