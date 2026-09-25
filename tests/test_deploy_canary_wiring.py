"""The post-deploy canary is wired into deploy.yml, and its verdict logic is executed, not read.

A workflow step is code nobody runs locally, which is how a gate quietly stops gating. So the
Python embedded in the "fail the deploy on a failed canary entity" step is EXTRACTED FROM THE
WORKFLOW FILE and run here against three results: one with a failed row, one all green, one
not_measured. If someone weakens that step, these go red.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "deploy.yml"
PLAN = ROOT / "deploy" / "canary" / "scrutator-production.plan.json"


def _deploy_steps():
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return document["jobs"]["deploy"]["steps"]


def _check_script() -> str:
    step = next(s for s in _deploy_steps() if "PYCHECK" in (s.get("run") or ""))
    return re.search(r"<<'PYCHECK'\n(.*?)\nPYCHECK", step["run"], re.S).group(1)


def _run_check(rows, tmp_path, plans=2) -> subprocess.CompletedProcess:
    """Run the workflow's own verdict script over a directory of results, as the step does."""
    results = tmp_path / "canary"
    results.mkdir()
    for index in range(plans):
        (results / f"plan{index}-post.json").write_text(
            json.dumps({"entity_verdicts": rows if index == 0 else []}), encoding="utf-8"
        )
    script = tmp_path / "check.py"
    script.write_text(_check_script(), encoding="utf-8")
    return subprocess.run([sys.executable, str(script), str(results)], capture_output=True, text=True)


def test_the_deploy_job_runs_the_committed_plan_after_the_deploy():
    steps = _deploy_steps()
    names = [s.get("name") for s in steps]
    canary = next(s for s in steps if s.get("id") == "canary")
    assert names.index("Deploy") < names.index(canary["name"]), "the canary must observe the deployed version"
    run = canary["run"]
    assert "tools/canary_probe.py" in run
    assert "scrutator-production" in run and "scrutator-route-presence" in run
    assert "--phase post" in run
    # the credential arrives through the environment and is never echoed
    assert "SCRUTATOR_CANARY_TOKEN" in json.dumps(canary.get("env", {}))
    assert "echo" not in run.split("SCRUTATOR_CANARY_TOKEN")[-1].split("\n")[0]


def test_the_plan_the_workflow_names_exists_and_carries_the_a2_308_probe():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["schema"] == "CanaryPlan/v1"
    negative = next(p for p in plan["probes"] if p["path"] == "/v1/edges" and p["method"] == "POST")
    # 401 and 403 are BOTH the A2-308 fix — which one answers depends on the grace-window flag,
    # measured on the real application in A2-311 §4. A 2xx is the hole, and no 2xx is accepted.
    assert negative["expect"]["status_in"] == [401, 403]
    assert not any(200 <= code < 300 for code in negative["expect"]["status_in"])
    assert negative["body"] == []
    assert negative.get("auth") == "none"
    assert negative["mutating"] is True and plan["owner"], "rule C4: a non-safe method needs an owner"
    assert "route:POST /v1/edges" in negative["entities"]
    reads = [p for p in plan["probes"] if p.get("method", "GET").upper() == "GET"]
    assert reads, "the plan must exercise a read route as well as the refusal"


def test_a_failed_entity_fails_the_deploy_step(tmp_path):
    done = _run_check(
        [{"entity": "route:POST /v1/edges", "verdict": "failed", "reason": "status 200 not in [403]"}], tmp_path
    )
    assert done.returncode == 1, done.stdout
    assert "::error" in done.stdout


def test_all_green_passes(tmp_path):
    done = _run_check([{"entity": "route:GET /health", "verdict": "verified", "reason": "status 200"}], tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_missing_plan_result_fails_the_step(tmp_path):
    """Rule C5: a canary that did not run is not a pass. One result where two were planned is red."""
    done = _run_check([{"entity": "route:GET /health", "verdict": "verified", "reason": "ok"}], tmp_path, plans=1)
    assert done.returncode == 1, done.stdout
    assert "expected one per plan" in done.stdout


def test_not_measured_is_reported_but_does_not_fail_the_deploy(tmp_path):
    """Rule C3: absence of measurement is not a regression — and it is never silent either."""
    done = _run_check(
        [{"entity": "route:GET /v1/namespaces", "verdict": "not_measured", "reason": "no credential"}], tmp_path
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "::warning" in done.stdout and "not_measured" in done.stdout


# A2-329. A red canary fails the run AFTER the target is live; from outside the run that red looked
# the same as a deploy that never happened. The outcome step says which red it is. Its shell is
# extracted from the workflow and run here, so a rewording that loses the distinction goes red.
def _outcome_step():
    return next(s for s in _deploy_steps() if s.get("name") == "Deploy outcome (deployed vs canary)")


def _run_outcome(tmp_path, deployed, canary, preflight="pass"):
    summary = tmp_path / "summary.md"
    env = {
        "PATH": "/usr/bin:/bin",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_STEP_SUMMARY": str(summary),
        "PREFLIGHT": preflight,
        "DEPLOYED": deployed,
        "CANARY": canary,
    }
    done = subprocess.run(["bash", "-c", _outcome_step()["run"]], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout, summary.read_text()


def test_the_outcome_step_always_runs_and_reads_the_real_step_ids():
    step, ids = _outcome_step(), {s.get("id") for s in _deploy_steps()}
    assert step["if"] == "always()"
    for ref in ("steps.deploy.outcome", "steps.canary_verdict.outputs.verdict", "steps.preflight.outputs.status"):
        assert ref in json.dumps(step["env"])
        assert ref.split(".")[1] in ids, f"{ref} names a step id that does not exist"


@pytest.mark.parametrize(
    ("deployed", "canary", "annotation"),
    [
        ("success", "failed", "::error title=deploy: DEPLOYED, canary RED::"),
        ("success", "passed", "::notice title=deploy: DEPLOYED, canary green::"),
        ("success", "not_measured", "::warning title=deploy: DEPLOYED, canary partly not_measured::"),
        ("success", "", "::error title=deploy: DEPLOYED, canary produced no verdict::"),
        ("failure", "", "::error title=deploy: target NOT live"),
        ("cancelled", "", "::error title=deploy: INTERRUPTED"),
        ("skipped", "", "::error title=deploy: NOT DEPLOYED (transaction did not run; preflight=fail)"),
    ],
)
def test_deployed_and_canary_red_is_distinguishable_from_not_deployed(tmp_path, deployed, canary, annotation):
    out, summary = _run_outcome(tmp_path, deployed, canary, preflight="fail" if deployed == "skipped" else "pass")
    assert out.startswith(annotation), out
    assert f"transaction={deployed}" in out and "a" * 40 in summary


def test_the_verdict_script_hands_its_verdict_to_the_outcome_step(tmp_path, monkeypatch):
    """The value the outcome step reads is what the REAL verdict script writes, not a fixture of it."""
    for rows, expected in (
        ([{"entity": "route:POST /v1/edges", "verdict": "failed", "reason": "200"}], "failed"),
        ([{"entity": "route:GET /health", "verdict": "verified", "reason": "200"}], "passed"),
        ([{"entity": "route:GET /x", "verdict": "not_measured", "reason": "no token"}], "not_measured"),
    ):
        case = tmp_path / expected
        case.mkdir()
        output = case / "github_output"
        output.write_text("")
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        _run_check(rows, case)
        assert output.read_text() == f"verdict={expected}\n"
