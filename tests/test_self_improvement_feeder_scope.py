from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_compose_appends_isolated_self_improvement_to_feeder_write_scope():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    environment = compose["services"]["scrutator"]["environment"]

    assert environment["SCRUTATOR_FEEDER_NAMESPACES"] == ("${SCRUTATOR_FEEDER_NAMESPACES:-},self-improvement")
