from pathlib import Path

WORKFLOW = Path(__file__).parents[2] / ".github/workflows/deploy.yml"


def test_scrutator_deploy_quiesces_timers_and_refuses_active_reconcilers():
    workflow = WORKFLOW.read_text()

    # MEASURED 2026-09-19. The step used to run the transaction script straight
    # out of the checkout; after the service moved hosts that failed with
    # "detected dubious ownership" — the production checkout is root:root and
    # the runner is ci-runner. The broker runs the ROOT-OWNED copy of the same
    # transaction, so what this assertion protects is unchanged (the deploy
    # goes through the transaction, never straight to compose or systemctl);
    # only the path to it moved.
    assert "scrutator-deploy-broker deploy" in workflow
    assert 'systemctl stop "${kb_timers[@]}"' not in workflow
    assert "docker compose up -d --build" not in workflow
    assert "git pull --ff-only" not in workflow
    assert "systemctl mask" not in workflow
    assert "systemctl unmask" not in workflow
    assert "systemctl is-enabled" not in workflow
