from pathlib import Path

WORKFLOW = Path(__file__).parents[2] / ".github/workflows/deploy.yml"


def test_scrutator_deploy_quiesces_timers_and_refuses_active_reconcilers():
    workflow = WORKFLOW.read_text()

    assert "deploy/scrutator-deploy-transaction.sh" in workflow
    assert 'systemctl stop "${kb_timers[@]}"' not in workflow
    assert "docker compose up -d --build" not in workflow
    assert "git pull --ff-only" not in workflow
    assert "systemctl mask" not in workflow
    assert "systemctl unmask" not in workflow
    assert "systemctl is-enabled" not in workflow
