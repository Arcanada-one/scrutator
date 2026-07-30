from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_compose_appends_only_reviewed_feeder_write_scopes():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    environment = compose["services"]["scrutator"]["environment"]

    assert environment["SCRUTATOR_FEEDER_NAMESPACES"] == (
        "${SCRUTATOR_FEEDER_NAMESPACES:-},self-improvement,arcanada-design-system"
    )
