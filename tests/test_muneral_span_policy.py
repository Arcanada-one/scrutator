"""Synthetic-only controls; no corpus values or credentials are fixtures."""

import copy
import hashlib
from unittest.mock import patch

import pytest

from tools.muneral_sync.secretscan import SEV_CRITICAL, Finding, ScanError, ScanResult
from tools.muneral_sync.span_policy import policy_digest, scan_task_field

TEXT = 'local_id="SyntheticTaskIdentifierAbCdef0123456789XYZ"'
TASK = "00000000-0000-4000-8000-000000000001"
SPAN = hashlib.sha256(TEXT.split('"')[1].encode()).hexdigest()


def candidate():
    return {
        "schema": "ExactSpanClassificationPolicy/v1",
        "status": "admitted",
        "enabled": True,
        "entries": [
            {
                "task_id": TASK,
                "field": "description",
                "line": 1,
                "current_field_sha256": hashlib.sha256(TEXT.encode()).hexdigest(),
                "span_sha256": SPAN,
                "provenance": {
                    "repo": "Arcanada-one/synthetic",
                    "commit": "a" * 40,
                    "blob": "b" * 40,
                    "file_sha256": "c" * 64,
                    "path": "fixtures/tasks.json",
                    "pointers": ["$.tasks[0].id"],
                    "span_sha256": SPAN,
                },
            }
        ],
    }


def scan(policy, *, text=TEXT, task=TASK, field="description", digest=True):
    with patch("tools.muneral_sync.secretscan._run_gitleaks", return_value=[]):
        return scan_task_field(
            text,
            task_id=task,
            field=field,
            policy=policy,
            admitted_policy_sha256=policy_digest(policy) if digest else None,
        )


def test_exact_admitted_binding_classifies_only_generic_entropy():
    result = scan(candidate())
    assert len(result.findings) == 1
    assert result.findings[0].rule == "generic-entropy"
    assert result.verdict == "info"


def test_disabled_candidate_and_no_policy_do_not_clear():
    policy = candidate()
    policy.update(enabled=False, status="candidate_not_admitted_not_enabled")
    assert scan(policy).is_critical
    with patch("tools.muneral_sync.secretscan._run_gitleaks", return_value=[]):
        assert scan_task_field(TEXT, task_id=TASK, field="description").is_critical


@pytest.mark.parametrize("change", ["task_id", "field", "line", "current_field_sha256", "span_sha256"])
def test_exact_identity_near_misses_block(change):
    policy = candidate()
    entry = policy["entries"][0]
    entry[change] = {
        "task_id": "00000000-0000-4000-8000-000000000002",
        "field": "title",
        "line": 2,
        "current_field_sha256": "0" * 64,
        "span_sha256": "0" * 64,
    }[change]
    if change == "span_sha256":
        entry["provenance"]["span_sha256"] = "0" * 64
    assert scan(policy).is_critical


def test_changed_field_and_adjacent_secret_block():
    assert scan(candidate(), text=TEXT + "\nPGPASSWORD=synthetic-only").is_critical


@pytest.mark.parametrize("kind", ["missing", "digest", "candidate", "proof", "duplicate"])
def test_authority_or_provenance_failure_refuses(kind):
    policy = candidate()
    if kind == "candidate":
        policy["status"] = "candidate_not_admitted_not_enabled"
    if kind == "proof":
        policy["entries"][0]["provenance"]["commit"] = "not-immutable"
    if kind == "duplicate":
        policy["entries"].append(copy.deepcopy(policy["entries"][0]))
    with pytest.raises(ScanError):
        if kind == "digest":
            scan_task_field(TEXT, task_id=TASK, field="description", policy=policy, admitted_policy_sha256="0" * 64)
        else:
            scan(policy, digest=kind != "missing")


@pytest.mark.parametrize("rule", ["vault-token-hvs", "pgpassword", "gitleaks:generic-api-key"])
def test_named_critical_and_gitleaks_cannot_be_classified(rule):
    result = ScanResult([Finding(rule, SEV_CRITICAL, 1, SPAN)], "critical")
    with patch("tools.muneral_sync.span_policy.scan_serialized", return_value=result):
        policy = candidate()
        assert scan_task_field(
            TEXT, task_id=TASK, field="description", policy=policy, admitted_policy_sha256=policy_digest(policy)
        ).is_critical


@pytest.mark.parametrize("key", ["commit", "path", "pointers"])
def test_changed_immutable_source_does_not_reuse_admission(key):
    policy = candidate()
    digest = policy_digest(policy)
    policy["entries"][0]["provenance"][key] = ["$.wrong"] if key == "pointers" else "wrong"
    with pytest.raises(ScanError):
        scan_task_field(TEXT, task_id=TASK, field="description", policy=policy, admitted_policy_sha256=digest)


@pytest.mark.parametrize("key", ["expected_output", "local_id"])
def test_unlisted_synthetic_assignment_stays_blocking(key):
    assert scan(candidate(), text=TEXT.replace("local_id", key) + " ").is_critical


def test_shipped_catalog_is_disabled_and_all_136_entries_have_typed_proof():
    import json
    from pathlib import Path

    from tools.muneral_sync.span_policy import _proven

    path = Path(__file__).parents[1] / "tools/muneral_sync/policies/F2-exact-span-candidate-20261010.json"
    policy = json.loads(path.read_text())
    assert policy["enabled"] is False
    assert policy["status"] == "candidate_not_admitted_not_enabled"
    assert len(policy["entries"]) == 136
    assert all(_proven(e) for e in policy["entries"])
