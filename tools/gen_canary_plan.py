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
import json
import re
import sys
from pathlib import Path

SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
PATH_PARAM = re.compile(r"\{[^}]+\}")
# A syntactically valid id that indexes nothing. Every probe is refused or rejected before the
# value is used; it exists so the URL is well formed.
PLACEHOLDER = "00000000-0000-0000-0000-000000000000"

OWNER = "Arcanada (control session) — scrutator deploy transaction"


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
    probes = []
    document = app.openapi()
    for path, operations in document.get("paths", {}).items():
        for verb in operations:
            method = verb.upper()
            if method not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
                continue
            entity_path = written.get(path, path)
            probe = {
                "id": re.sub(r"[^a-z0-9]+", "-", f"{method} {path}".lower()).strip("-"),
                "method": method,
                "path": PATH_PARAM.sub(PLACEHOLDER, path),
                "auth": "none",
                "expect": {"route_present": True},
                "entities": [f"route:{method} {entity_path}"],
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
        "owner": OWNER,
        "read_only": False,
        "_generated_by": "tools/gen_canary_plan.py — DO NOT EDIT BY HAND",
        "_claim": (
            "Presence only: the resident version still serves this path with this verb. Every probe "
            "is unauthenticated and carries no body, so 401, 403 and 422 are all reached before the "
            "handler; 404 is the mutant this plan exists to catch."
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
