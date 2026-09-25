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
import os
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
    return {e for probe in plan["probes"] for e in probe["entities"]}


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
        if probe["method"] not in ("GET", "HEAD", "OPTIONS"):
            assert probe["mutating"] is True, f"rule C4: {probe['id']}"
    assert plan["owner"]


def test_a_renamed_route_is_caught(tmp_path):
    """The mutant class the offline verifiers cannot see: a path rewrite answers 404."""
    module = _generator()

    class _Route:
        def __init__(self, path):
            self.path = path
            self.methods = {"GET"}

    class _App:
        def openapi(self):
            return {"paths": {"/v1/ltm/graph": {"get": {}}, "/health": {"get": {}}}}

    plan = module.build(_App())
    ids = {e for p in plan["probes"] for e in p["entities"]}
    assert ids == {"route:GET /graph", "route:GET /health"}


@pytest.mark.skipif(os.environ.get("SCRUTATOR_SKIP_SLOW"), reason="opt-out for slow graph build")
def test_the_two_plans_do_not_contradict_each_other():
    """Both plans may name the same route; `verify.py` takes the WORST verdict, so a presence probe
    can never paper over a failed refusal probe."""
    production = json.loads((ROOT / "deploy" / "canary" / "scrutator-production.plan.json").read_text("utf-8"))
    shared = {e for p in production["probes"] for e in p["entities"]} & _plan_entities()
    assert "route:POST /v1/edges" in shared
