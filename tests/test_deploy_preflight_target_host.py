"""Pre-deploy preflight must name the live runtime host (INFRA-0417).

`target-host` is the label carried into preflight findings and Ops Bot meta.
After INFRA-0417 Scrutator runs on arcana-prd; arcana-kb is decommissioned.
spaces/arcanada/space.yml → projects[scrutator].runtime_host: arcana-prd.
"""

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "deploy.yml"


def test_preflight_target_host_is_arcana_prd():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    deploy = workflow["jobs"]["deploy"]
    preflight = next(
        step
        for step in deploy["steps"]
        if step.get("name") == "Pre-deploy host readiness"
    )
    assert preflight["with"]["target-host"] == "arcana-prd"
    assert "arcana-kb" not in WORKFLOW.read_text()
