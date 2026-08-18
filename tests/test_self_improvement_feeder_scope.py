from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every namespace appended to SCRUTATOR_FEEDER_NAMESPACES is a write scope the
# kb-feeder token carries at runtime. kb-feeder is fail-closed on exactly this:
# `feeder.py` aborts the whole batch with `unexpected_runtime_grants=<ns>` when
# the token holds a namespace its reconcile run does not expect.
#
# The two sides live in different repositories on different hosts. Appending
# `skills` here on 2026-08-02 (commit 5270848) stopped KB ingestion the same
# evening — last success 18:08, first halt 18:29 — and it stayed stopped for
# sixteen days. The compose contract test of the day was satisfied by updating
# the expected string, which is why nothing objected.
#
# So the literal list below is not only a review record; each entry must also be
# accounted for on the consumer side, in arcanada-workspace:
#   dev-tools/kb-feeder/deploy/kb-reconcile-run.sh  --runtime-only-namespace <ns>
#   dev-tools/kb-feeder/deploy/authoritative-namespaces.txt  (projected corpus)
# Gate there: dev-tools/check-kb-feeder-grant-drift.sh.
FEEDER_APPENDED_SCOPES = ("self-improvement", "arcanada-design-system", "skills")


def test_appended_feeder_scopes_are_declared_for_the_kb_feeder_consumer():
    """Appending a write scope here is half a change; the consumer must follow.

    This test cannot reach the other repository, so it asserts the reviewed
    inventory instead: adding a namespace to the compose grant forces this tuple
    to change, and the comment above says what else has to change with it.
    """
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    grant = compose["services"]["scrutator"]["environment"]["SCRUTATOR_FEEDER_NAMESPACES"]

    base, _, appended = grant.partition("}")
    assert base + "}" == "${SCRUTATOR_FEEDER_NAMESPACES:-}", "operator-configured base must stay first"

    actual = tuple(ns for ns in appended.lstrip(",").split(",") if ns)
    assert actual == FEEDER_APPENDED_SCOPES, (
        "the feeder write grant changed — each appended namespace must also be declared in "
        "arcanada-workspace (kb-reconcile-run.sh --runtime-only-namespace, or the projected "
        "authoritative-namespaces list), or kb-feeder halts every batch and KB ingestion stops"
    )


def test_compose_appends_only_reviewed_skills_proof_scopes():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    environment = compose["services"]["scrutator"]["environment"]

    assert environment["SCRUTATOR_FEEDER_NAMESPACES"] == (
        "${SCRUTATOR_FEEDER_NAMESPACES:-},self-improvement,arcanada-design-system,skills"
    )
    assert environment["SCRUTATOR_ROLLBACK_NAMESPACES"] == ("${SCRUTATOR_ROLLBACK_NAMESPACES:-},skills")
    assert "SCRUTATOR_CAPABILITY_PROJECTION_TOKEN" not in environment
    assert "SCRUTATOR_CAPABILITY_PROJECTION_TENANTS" not in environment
