from fastapi.routing import APIRoute

from scrutator.health import app
from scrutator.ltm.router import router as ltm_router

MACHINE_ROUTES = {
    ("POST", "/v1/index"): "require_feeder_capability",
    ("POST", "/v1/index/batch"): "require_feeder_capability",
    ("DELETE", "/v1/index"): "require_rollback_capability",
    ("POST", "/v1/ltm/ingest"): "require_ltm_writer_capability",
    ("DELETE", "/v1/ltm/source"): "require_ltm_writer_capability",
}


def test_every_business_route_has_exactly_one_declared_auth_policy():
    seen: set[tuple[str, str]] = set()
    # This FastAPI version retains included routers as a nested route object,
    # so inventory both the app routes and the included LTM router explicitly.
    for route in [*app.routes, *ltm_router.routes]:
        if not isinstance(route, APIRoute) or not route.path.startswith("/v1/"):
            continue
        dependency_names = {dependency.call.__name__ for dependency in route.dependant.dependencies}
        for method in route.methods:
            key = (method, route.path)
            machine_dependency = MACHINE_ROUTES.get(key)
            if machine_dependency:
                assert machine_dependency in dependency_names, key
                assert "require_tenant_context" not in dependency_names, key
                seen.add(key)
            else:
                assert "require_tenant_context" in dependency_names, key
                assert not dependency_names.intersection(MACHINE_ROUTES.values()), key

    assert seen == set(MACHINE_ROUTES)
