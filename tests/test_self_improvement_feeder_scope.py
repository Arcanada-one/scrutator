from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_compose_appends_only_reviewed_skills_proof_scopes():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    environment = compose["services"]["scrutator"]["environment"]

    assert environment["SCRUTATOR_FEEDER_NAMESPACES"] == (
        "${SCRUTATOR_FEEDER_NAMESPACES:-},self-improvement,arcanada-design-system,skills"
    )
    assert environment["SCRUTATOR_ROLLBACK_NAMESPACES"] == ("${SCRUTATOR_ROLLBACK_NAMESPACES:-},skills")
    assert "SCRUTATOR_CAPABILITY_PROJECTION_TOKEN" not in environment
    assert "SCRUTATOR_CAPABILITY_PROJECTION_TENANTS" not in environment
