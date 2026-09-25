#!/usr/bin/env python3
"""Generate the route-presence CanaryPlan/v1 from the application's own routing table.

WHAT THIS MEASURES, AND WHAT IT DOES NOT. One thing only: that the resident version still serves
this path with this verb. Not that the answer is correct, not that the data is right — presence.
That narrow claim is the one the offline verifiers cannot make: AUP-GRAPH-007 could not calibrate
the `provides_route` class at all (13 detected mutants against a minimum of 20, bias-free recall
0.25) and 50 of its 65 route mutants were path or verb rewrites, which no type-checker and no unit
test on a Python tree can observe. A rewritten path answers 404, and `expect.route_present` is red.

Every probe is UNAUTHENTICATED AND CARRIES NO BODY on purpose. Both refusals (401 at the door, 403
at the verb) and FastAPI's 422 for a missing body are reached BEFORE the handler runs, so the probe
observes the routing table without touching the data. Non-safe methods are declared `mutating: true`
with the plan's owner as rule C4 requires — the question the rule asks is what the request COULD do,
not what it is meant to do.

    tools/gen_canary_plan.py [--out deploy/canary/scrutator-route-presence.plan.json] [--check]

The entity ids must be the ones `build_graph` writes, or the canary lists entities the receipt has
never heard of. build_graph reads the DECORATOR path (`@router.get("/graph")`) and does not apply
`APIRouter(prefix=…)`, so a route served at /v1/ltm/graph is `route:GET /graph` in the graph. That
is a defect in the builder, reported as such; until it is fixed the plan has to speak the builder's
language, and `tests/test_canary_plan.py` asserts the two agree by building the graph itself.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path

SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
PATH_PARAM = re.compile(r"\{[^}]+\}")
# A syntactically valid id that indexes nothing; it exists so the URL is well formed. It is NOT
# always refused before use: while SCRUTATOR_AUTH_ENFORCE is off an unauthenticated GET reaches the
# handler, which looks the id up and answers 404 for an absent resource (A2-324) — a read, and the
# reason presence is not read off the status code.
PLACEHOLDER = "00000000-0000-0000-0000-000000000000"

OWNER = "Arcanada (control session) — scrutator deploy transaction"


def served_routes(app):
    """Every APIRoute of the application, including those behind an included router.

    FastAPI 0.141 wraps an included router in an opaque `_IncludedRouter`; its `original_router`
    holds the real routes. Walked rather than special-cased, so a second router added later is
    covered without touching this file.
    """
    out, stack = [], list(app.routes)
    while stack:
        route = stack.pop()
        inner = getattr(route, "original_router", None)
        if inner is not None:
            stack.extend(inner.routes)
            continue
        if getattr(route, "endpoint", None) is not None and getattr(route, "methods", None):
            out.append(route)
    return out


def declaring_file(endpoint) -> str | None:
    """The repository-relative path of the module that declares this endpoint, or None."""
    source = inspect.getsourcefile(endpoint)
    if not source:
        return None
    path = Path(source).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists():
            return path.relative_to(parent).as_posix()
    return None


def decorator_paths() -> dict:
    """served path -> the path as WRITTEN in the decorator, for every prefixed router.

    FastAPI 0.141 keeps included routers as opaque `_IncludedRouter` objects in `app.routes`, so
    the served surface is read from the OpenAPI document instead and the decorator spelling is
    recovered from each router's own routes. Only routers declared here are translated; anything
    else keeps its served path, which is also what `build_graph` recorded for it.
    """
    from scrutator.ltm.router import router as ltm_router

    out = {}
    for router in (ltm_router,):
        prefix = router.prefix
        for route in router.routes:
            # FastAPI stores the FULL path on the route when the prefix is set on the APIRouter
            # constructor; `build_graph` read the decorator's literal argument, which is the
            # remainder. Measured, not assumed: router.routes[0].path is "/v1/ltm/ingest".
            served = route.path if route.path.startswith(prefix) else prefix + route.path
            out[served] = served[len(prefix) :] or "/"
    return out


def build(app) -> dict:
    written = decorator_paths()
    served = app.openapi().get("paths", {})
    declared_by = {}
    for route in served_routes(app):
        source = declaring_file(route.endpoint)
        for method in route.methods:
            declared_by[(method, route.path)] = source
    probes = []
    for path, operations in served.items():
        for verb in operations:
            method = verb.upper()
            if method not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
                continue
            entity_path = written.get(path, path)
            entities = [f"route:{method} {entity_path}"]
            # The controller is an entity too, and it is the one that carries the
            # `inferred_boundary` hold in verify.py: a canary that names only the route leaves
            # src/scrutator/ltm/router.py not_measured while every route it serves is verified.
            source = declared_by.get((method, path))
            if source:
                entities.append(f"code_unit:{source}")
            probe = {
                "id": re.sub(r"[^a-z0-9]+", "-", f"{method} {path}".lower()).strip("-"),
                "method": method,
                "path": PATH_PARAM.sub(PLACEHOLDER, path),
                # The template as the resident OpenAPI spells it: presence is decided against that
                # document, not read off a 404 the handler may legitimately answer (A2-324).
                "route": path,
                "auth": "none",
                "expect": {"route_present": True},
                "entities": entities,
            }
            if method not in SAFE_METHODS:
                probe["mutating"] = True
            probes.append(probe)
    probes.sort(key=lambda p: p["id"])
    return {
        "schema": "CanaryPlan/v1",
        "id": "scrutator-route-presence",
        "environment": "kb-production",
        "base_url": "http://127.0.0.1:8310",
        "openapi_path": "/openapi.json",
        "owner": OWNER,
        "read_only": False,
        "_generated_by": "tools/gen_canary_plan.py — DO NOT EDIT BY HAND",
        "_claim": (
            "Presence only: the resident version still serves this path with this verb. Every probe "
            "is unauthenticated and carries no body. A route is present when the resident OpenAPI "
            "declares it AND the answer is not the router's own no-match (405, or a 404 byte-identical "
            "to the fingerprint of an unrouted sibling path); a handler's 404 for an absent resource "
            "is a served route (A2-324)."
        ),
        "probes": probes,
    }


def render(app) -> str:
    return json.dumps(build(app), indent=2, ensure_ascii=False) + "\n"


def generate() -> str:
    from scrutator.health import app

    return render(app)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="deploy/canary/scrutator-route-presence.plan.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    generated = generate()
    target = Path(args.out)
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != generated:
            print(f"{target}: out of date — run `python3 tools/gen_canary_plan.py`", file=sys.stderr)
            return 1
        print(f"{target}: matches the application")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(generated, encoding="utf-8")
    print(f"{target}: {len(json.loads(generated)['probes'])} probes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
