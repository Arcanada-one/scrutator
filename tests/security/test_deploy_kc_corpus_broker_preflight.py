"""A2-457: deploy-kc-corpus must refuse docker-group runners before compose-broker sudo."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WF = ROOT / ".github" / "workflows" / "deploy-kc-corpus.yml"


def test_deploy_kc_corpus_has_runner_broker_preflight_before_sudo() -> None:
    text = WF.read_text(encoding="utf-8")
    assert "Runner broker preflight (A2-457)" in text
    assert "scripts/ci/runner-broker-preflight.sh" in text
    assert "BROKER_PREFLIGHT_BIN: /usr/local/sbin/arcanada-compose-broker" in text
    preflight_at = text.index("Runner broker preflight (A2-457)")
    first_sudo_at = text.index("sudo -n /usr/local/sbin/arcanada-compose-broker")
    assert preflight_at < first_sudo_at, "preflight must precede any compose-broker sudo"


def test_deploy_kc_corpus_preflight_absence_is_detectable() -> None:
    """Negative: a workflow stripped of preflight fails the presence contract."""
    text = WF.read_text(encoding="utf-8")
    stripped = "\n".join(
        line
        for line in text.splitlines()
        if "Runner broker preflight" not in line
        and "runner-broker-preflight.sh" not in line
        and "BROKER_PREFLIGHT_BIN" not in line
    )
    assert "Runner broker preflight (A2-457)" not in stripped
    assert "sudo -n /usr/local/sbin/arcanada-compose-broker" in stripped
