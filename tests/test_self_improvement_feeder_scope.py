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
#
# `talomnia` (2026-08-30) is the projected-corpus kind, not the runtime-only
# kind, so it needs no --runtime-only-namespace: arcana-kb's live
# /var/lib/kb-feeder/corpus/.authoritative-namespaces already lists it (line 52
# of 58), which is why the feeder reported it as `missing_namespace_grants`
# rather than halting on `unexpected_runtime_grants`. The accounting it does
# need is the repo-side copy of that manifest, added in the paired
# arcanada-workspace change to
# dev-tools/kb-feeder/deploy/authoritative-namespaces.txt — without it
# check-kb-feeder-grant-drift.sh sees a grant nobody declared.
FEEDER_APPENDED_SCOPES = ("self-improvement", "arcanada-design-system", "skills", "talomnia")


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
        "${SCRUTATOR_FEEDER_NAMESPACES:-},self-improvement,arcanada-design-system,skills,talomnia"
    )
    assert environment["SCRUTATOR_ROLLBACK_NAMESPACES"] == ("${SCRUTATOR_ROLLBACK_NAMESPACES:-},skills")
    assert "SCRUTATOR_CAPABILITY_PROJECTION_TOKEN" not in environment
    assert "SCRUTATOR_CAPABILITY_PROJECTION_TENANTS" not in environment


def test_production_deploy_injects_dense_sparse_flag_without_mutating_dotenv():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "deploy.yml").read_text())

    environment = compose["services"]["scrutator"]["environment"]
    assert environment["SCRUTATOR_EMBEDDING_DENSE_SPARSE_ENABLED"] == (
        "${SCRUTATOR_EMBEDDING_DENSE_SPARSE_ENABLED:-false}"
    )

    deploy_step = next(step for step in workflow["jobs"]["deploy"]["steps"] if step.get("name") == "Deploy")
    assert deploy_step["env"]["SCRUTATOR_EMBEDDING_DENSE_SPARSE_ENABLED"] == (
        "${{ vars.SCRUTATOR_EMBEDDING_DENSE_SPARSE_ENABLED }}"
    )
