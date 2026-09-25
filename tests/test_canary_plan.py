"""The route-presence plan names exactly the routes the ECOSYSTEM'S BUILDER sees.

A canary is worth nothing if it lists entity ids the receipt has never heard of: `verify.py` matches
`CanaryResult.entity_verdicts[].entity` against the graph's node ids by string equality, so an id
that is one prefix out is silently no coverage at all. The ids are therefore not asserted against a
hand-written list — the vendored `build_graph.py` is RUN over this repository and the two sets are
compared. A plan that drifts from the application, or a builder whose route naming changes, is red
here rather than quietly uncovered in an admission.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / ".github" / "graph-admission" / "tools" / "graph" / "build_graph.py"
PLAN = ROOT / "deploy" / "canary" / "scrutator-route-presence.plan.json"


def _generator():
    spec = importlib.util.spec_from_file_location("gen_canary_plan", ROOT / "tools" / "gen_canary_plan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _graph_routes() -> set[str]:
    with tempfile.TemporaryDirectory(prefix="a2-311-graph-") as work:
        out = Path(work) / "graph.json"
        done = subprocess.run(
            [sys.executable, str(BUILDER), str(ROOT), "--out", str(out), "--rev", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if done.returncode != 0 or not out.exists():
            pytest.skip(f"the vendored builder could not run here: {done.stderr[-300:]}")
        graph = json.loads(out.read_text(encoding="utf-8"))
    return {n["id"] for n in graph["nodes"] if n["type"] == "route" and (n.get("path") or "").startswith("src/")}


def _plan_entities() -> set[str]:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    return {e for probe in plan["probes"] for e in probe["entities"] if e.startswith("route:")}


def _plan_controllers() -> set[str]:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    return {e for probe in plan["probes"] for e in probe["entities"] if e.startswith("code_unit:")}


def test_the_committed_plan_equals_the_application():
    assert PLAN.read_text(encoding="utf-8") == _generator().generate(), (
        "deploy/canary/scrutator-route-presence.plan.json is stale — run `python3 tools/gen_canary_plan.py`"
    )


def test_check_mode_goes_red_when_the_plan_drifts(tmp_path):
    drifted = tmp_path / "drifted.json"
    drifted.write_text(_generator().generate().replace("/health", "/healthz", 1), encoding="utf-8")
    assert _generator().main(["--check", "--out", str(drifted)]) == 1


def test_every_route_the_builder_sees_is_probed():
    graph, plan = _graph_routes(), _plan_entities()
    assert graph, "the builder found no routes at all — the comparison would be vacuous"
    assert graph - plan == set(), f"routes the canary does not probe: {sorted(graph - plan)}"
    assert plan - graph == set(), f"probed entities no graph node carries: {sorted(plan - graph)}"


def test_no_probe_sends_a_body_or_a_credential():
    """Presence is observed before the handler runs — that is what makes this safe on production."""
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    for probe in plan["probes"]:
        assert "body" not in probe, probe["id"]
        assert probe["auth"] == "none", probe["id"]
        assert probe["expect"] == {"route_present": True}, probe["id"]
        assert any(e.startswith("route:") for e in probe["entities"]), probe["id"]
        if probe["method"] not in ("GET", "HEAD", "OPTIONS"):
            assert probe["mutating"] is True, f"rule C4: {probe['id']}"
    assert plan["owner"]


def test_a_prefixed_router_is_probed_at_the_served_path_and_named_at_the_decorator_path():
    """The translation nine routes depend on, asserted on the committed plan.

    `build_graph` records `@router.get("/graph")` as `route:GET /graph` and never applies the
    router's `/v1/ltm` prefix, while the URL that must be probed is `/v1/ltm/graph`. Get this
    backwards either way and the canary covers nothing while looking complete.
    """
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    graph_probe = next(p for p in plan["probes"] if p["path"] == "/v1/ltm/graph")
    assert "route:GET /graph" in graph_probe["entities"]
    assert "code_unit:src/scrutator/ltm/router.py" in graph_probe["entities"]
    assert not any("/v1/ltm" in e for e in graph_probe["entities"])


def test_every_controller_is_named_too():
    """verify.py holds the CONTROLLER on an inferred boundary as well as the route: a canary that
    names only routes leaves src/scrutator/ltm/router.py not_measured while all nine of its routes
    are verified."""
    assert _plan_controllers() == {"code_unit:src/scrutator/health.py", "code_unit:src/scrutator/ltm/router.py"}


def test_the_two_plans_do_not_contradict_each_other():
    """Both plans may name the same route; `verify.py` takes the WORST verdict, so a presence probe
    can never paper over a failed refusal probe."""
    production = json.loads((ROOT / "deploy" / "canary" / "scrutator-production.plan.json").read_text("utf-8"))
    shared = {e for p in production["probes"] for e in p["entities"]} & _plan_entities()
    assert "route:POST /v1/edges" in shared
