"""muneral-kb-sync outbound scan: identifier slugs are not credentials, real tokens still block (c5b5df36)."""

from __future__ import annotations

import hashlib
import json

import pytest

from tools.muneral_sync import secretscan
from tools.muneral_sync.secretscan import (
    SEV_CRITICAL,
    SEV_INFO,
    Finding,
    ScanError,
    apply_allowlist,
    is_identifier_slug,
    load_allowlist,
    scan_text,
)

# Shapes measured on the blocked tasks of 2026-11-13 (a kebab slug of 4-7 words plus a date; one has a leading capital).
SLUGS = [
    "prime-short-ui-stage-2026-11-12",
    "Prime-short-ui-stage-bounded-review-2026-11-02",
    "one-more-readable-task-title-for-an-ops-review",
    "release_candidate_review_notes_2026_11_13",
]
# A synthetic random-looking value, assembled at run time so that no secret scanner reads a literal in this file.
RANDOM_LOOKING = "".join(["Zq8vT3nB7x", "Lk2Wm9PdR5", "sYc1HfJ6Ae", "Gu"])
# Things that look like credentials and must still be reported when they are quoted assignments.
TOKENS = [
    RANDOM_LOOKING,  # mixed-case alphanumerics
    "d41d8cd98f00b204e9800998ecf8427e1a2b3c4d5e6f7a8b9c0d",  # long hex
    "aB3dE5fG7hJ9kL1mN3pQ5rS7tU9vW1xY3zA5bC7d",  # random letters+digits
    "abcdefghij-klmnopqrst-uvwxyz0123-456789ab",  # a part mixes letters and digits
    "xqzmk-dpvrw-tbnhj-lcsfy-mgkzt",  # lowercase groups without vowels: a license-key shape
    "x7q-m2k-p9z-w4r-t8v-b6n",  # license-key shape: short mixed parts
]


@pytest.mark.parametrize("slug", SLUGS)
def test_a_readable_slug_is_an_identifier_not_a_secret(slug):
    assert is_identifier_slug(slug)
    assert secretscan.shannon_entropy(slug) > 3.0  # the sample set includes values that the entropy rule would flag
    payload = json.dumps({"entity": {"muneral_id": slug, "display_name": slug, "description": slug}})
    assert scan_text(payload).findings == []


def test_at_least_one_sample_slug_is_above_the_entropy_threshold():
    # the point of the exemption: without it these are findings
    flagged = [s for s in SLUGS if secretscan.shannon_entropy(s) > secretscan._ENTROPY_THRESHOLD]
    assert flagged, "the sample must contain a slug that the raw entropy rule would block"
    for slug in flagged:
        raw = json.dumps({"muneral_id": slug})
        assert is_identifier_slug(slug) and scan_text(raw).findings == []


@pytest.mark.parametrize("token", TOKENS)
def test_a_random_looking_token_is_not_a_slug_and_is_still_blocked(token):
    assert not is_identifier_slug(token)


def test_a_quoted_assignment_of_a_real_looking_secret_still_blocks():
    result = scan_text('{"api_key":"' + RANDOM_LOOKING + '"}')
    assert result.is_critical and result.findings[0].rule == "generic-entropy"
    assert scan_text('password = "' + RANDOM_LOOKING + '"').is_critical
    # pure hex can never exceed 4.0 bits, so the entropy rule does not read it as a secret (named rules and gitleaks do)
    assert secretscan.shannon_entropy("d41d8cd98f00b204e9800998ecf8427e1a2b3c4d5e6f7a8b9c0d") <= 4.0


def test_the_critical_named_rules_are_untouched_by_the_slug_exemption():
    # a Vault token inside a slug-looking JSON value is caught by its own named rule, not by entropy
    result = scan_text('{"display_name":"hvs.CAESIabcdefghijklmnopqrstuvwxyz0123456789"}')
    assert [f.rule for f in result.findings] == ["vault-token-hvs"]


def _entry(rule: str, text: str, reason: str = "reviewed prose") -> dict:
    return {"rule": rule, "span_hash": hashlib.sha256(text.encode()).hexdigest(), "reason": reason}


def test_an_allowlisted_span_is_downgraded_to_info_and_others_are_not(tmp_path):
    path = tmp_path / "allow.json"
    path.write_text(json.dumps({"entries": [_entry("gitleaks:generic-api-key", "the prose span")]}))
    allowed = load_allowlist(path)
    listed = Finding("gitleaks:generic-api-key", SEV_CRITICAL, 1, hashlib.sha256(b"the prose span").hexdigest())
    other_span = Finding("gitleaks:generic-api-key", SEV_CRITICAL, 1, hashlib.sha256(b"another span").hexdigest())
    other_rule = Finding("generic-entropy", SEV_CRITICAL, 1, hashlib.sha256(b"the prose span").hexdigest())
    out = apply_allowlist([listed, other_span, other_rule], allowed)
    assert [f.severity for f in out] == [SEV_INFO, SEV_CRITICAL, SEV_CRITICAL]


def test_a_missing_allowlist_means_no_entries_and_a_malformed_one_fails_closed(tmp_path):
    assert load_allowlist(tmp_path / "absent.json") == {}
    for bad in (
        '{"entries":[{"rule":"r"}]}',
        "not json",
        '{"entries":[{"rule":"r","span_hash":"zz","reason":"x"}]}',
        json.dumps({"entries": [_entry("r", "x", reason="  ")]}),
    ):
        path = tmp_path / "bad.json"
        path.write_text(bad)
        with pytest.raises(ScanError):
            load_allowlist(path)


def test_the_shipped_allowlist_is_valid_and_every_entry_has_a_reason():
    allowed = load_allowlist()
    assert allowed and all(reason.strip() for reason in allowed.values())
    data = json.loads(secretscan.ALLOWLIST_PATH.read_text())
    assert all(set(e) >= {"rule", "span_hash", "reason", "task"} for e in data["entries"])
    assert not any(len(e["span_hash"]) != 64 for e in data["entries"])


def test_scan_serialized_applies_the_allowlist(monkeypatch, tmp_path):
    span = "synthetic prose span"
    path = tmp_path / "allow.json"
    path.write_text(json.dumps({"entries": [_entry("gitleaks:generic-api-key", span)]}))
    monkeypatch.setattr(secretscan, "ALLOWLIST_PATH", path)
    finding = Finding("gitleaks:generic-api-key", SEV_CRITICAL, 1, hashlib.sha256(span.encode()).hexdigest())
    monkeypatch.setattr(secretscan, "_run_gitleaks", lambda text: [finding])
    result = secretscan.scan_serialized('{"k":"v"}')
    assert not result.is_critical and result.findings[0].severity == SEV_INFO
    monkeypatch.setattr(
        secretscan, "_run_gitleaks", lambda text: [Finding("gitleaks:generic-api-key", SEV_CRITICAL, 1, "0" * 64)]
    )
    assert secretscan.scan_serialized('{"k":"v"}').is_critical


# A slug that the entropy rule would flag (entropy above 4.0 because it is long and varied).
HIGH_ENTROPY_SLUG = "Prime-short-ui-stage-bounded-review-2026-11-02"


@pytest.mark.parametrize(
    "key",
    [
        "password",
        "PASSWORD",
        "passwd",
        "client_secret",
        "access_token",
        "api_key",
        "apikey",
        "api-key",
        "credential",
        "private_key",
        "auth",
        "authorization",
        "db_password",
        "X-Auth-Token",
    ],
)
def test_the_same_slug_is_blocked_under_a_credential_named_key(key):
    assert secretscan.shannon_entropy(HIGH_ENTROPY_SLUG) > secretscan._ENTROPY_THRESHOLD
    assert is_identifier_slug(HIGH_ENTROPY_SLUG)
    result = scan_text(json.dumps({key: HIGH_ENTROPY_SLUG}))
    assert result.is_critical and result.findings[0].rule == "generic-entropy"
    assert scan_text(f'{key} = "{HIGH_ENTROPY_SLUG}"').is_critical


@pytest.mark.parametrize("key", ["display_name", "muneral_id", "description", "title", "name", "source_path"])
def test_the_same_slug_passes_under_a_name_field(key):
    assert scan_text(json.dumps({key: HIGH_ENTROPY_SLUG})).findings == []
