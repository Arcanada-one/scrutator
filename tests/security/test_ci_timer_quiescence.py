from pathlib import Path

WORKFLOW = Path(__file__).parents[2] / ".github/workflows/ci.yml"


def test_scrutator_deploy_masks_timers_and_refuses_active_reconcilers():
    workflow = WORKFLOW.read_text()

    mask_at = workflow.index('systemctl mask --runtime --now "${kb_timers[@]}"')
    idle_gate_at = workflow.index("kb_reconcile_services=(")
    recreate_at = workflow.index("docker compose up -d --build")
    assert mask_at < idle_gate_at < recreate_at
    assert 'systemctl is-active --quiet "$service"' in workflow
    assert "exit 75" in workflow
    assert "trap resume_kb_timers EXIT" in workflow
    assert 'systemctl unmask --runtime "${kb_timers[@]}"' in workflow
    assert "active_kb_timers=()" in workflow
    assert 'active_kb_timers+=("$timer")' in workflow
    assert "if ((${#active_kb_timers[@]})); then" in workflow
    assert 'systemctl start "${active_kb_timers[@]}"' in workflow
    assert 'systemctl start "${kb_timers[@]}"' not in workflow
