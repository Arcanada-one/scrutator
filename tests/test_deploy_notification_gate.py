from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "deploy.yml"


def test_external_deploy_notifications_are_structurally_disabled():
    workflow = WORKFLOW.read_text()

    assert 'ops-bot-emit: "false"' in workflow
    for forbidden in (
        "OPSBOT",
        "ops-bot-key:",
        "ops.arcanada.ai/events",
        "Notify Ops Bot",
    ):
        assert forbidden not in workflow
