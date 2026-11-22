#!/usr/bin/env python3
"""Measure archived receipt DATA; never re-execute or approve its source change.

DEC-AUP-0083. Locators and stored `posted` fields are untrusted. Native GETs,
Git bytes and canonical schema validation are independently required.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
import urllib.error
import urllib.request

import schema_check

SCOPE = "archived_evidence_data"
ENDPOINT = "https://api.muneral.com/api/v1"
POLICY_PATH = "contracts/graph-verified-change/admission-gate.v1.json"
MAX_BLOB = 16 * 1024 * 1024
MAX_NATIVE = 1024 * 1024
UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z", re.I)
OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class Unmeasured(Exception):
    """The measurement cannot be established; no upgrade is allowed."""


class Contradiction(Exception):
    """Measured data contradicts its declared archive binding."""


def strict_json(raw: bytes):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise Contradiction("DUPLICATE_JSON_KEY")
            out[key] = value
        return out
    try:
        return json.loads(raw, object_pairs_hook=pairs)
    except (ValueError, UnicodeError):
        raise Contradiction("MALFORMED_JSON") from None


def safe_path(value):
    if (not isinstance(value, str) or not value or "\\" in value
            or any(ord(c) < 32 for c in value) or value.startswith("/")
            or any(c in ("", ".", "..") for c in value.split("/"))
            or PurePosixPath(value).as_posix() != value):
        raise Contradiction("UNSAFE_GIT_PATH")
    return value


class GitObjects:
    def __init__(self, repo):
        self.repo = Path(repo)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        self.env["GIT_NO_REPLACE_OBJECTS"] = "1"
        self.env["GIT_LITERAL_PATHSPECS"] = "1"
        self.cache = {}
        self.map_cache = {}
        common = Path(self.command("rev-parse", "--git-common-dir").decode().strip())
        if not common.is_absolute():
            common = self.repo / common
        if (common / "info/grafts").exists():
            raise Unmeasured("GIT_GRAFTS_UNSUPPORTED")

    def command(self, *args):
        result = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", *args],
                                cwd=self.repo, env=self.env, capture_output=True, timeout=20)
        if result.returncode:
            raise Unmeasured("GIT_OBJECT_UNAVAILABLE")
        return result.stdout

    def blob(self, rev, path):
        if not isinstance(rev, str) or not OID.fullmatch(rev):
            raise Contradiction("IMMUTABLE_REVISION_REQUIRED")
        safe_path(path)
        key = rev, path
        if key in self.cache:
            return self.cache[key]
        row = self.command("ls-tree", "-z", rev, "--", path)
        if not row:
            return None
        parts = row.rstrip(b"\0").split(b"\t")
        if len(parts) != 2 or parts[1].decode() != path:
            raise Contradiction("AMBIGUOUS_GIT_PATH")
        mode, kind, oid = parts[0].decode().split()
        if mode != "100644" or kind != "blob":
            raise Contradiction("REGULAR_RECEIPT_BLOB_REQUIRED")
        size = int(self.command("cat-file", "-s", oid))
        if size > MAX_BLOB:
            raise Unmeasured("GIT_BLOB_BOUND_EXCEEDED")
        raw = self.command("cat-file", "blob", oid)
        if len(raw) != size:
            raise Contradiction("GIT_BLOB_SIZE_MISMATCH")
        witness = b"blob " + str(len(raw)).encode() + b"\0" + raw
        actual_oid = (hashlib.sha1(witness) if len(oid) == 40 else hashlib.sha256(witness)).hexdigest()
        if actual_oid != oid:
            raise Contradiction("GIT_BLOB_OID_MISMATCH")
        self.cache[key] = raw, oid
        return raw, oid

    def ancestor(self, older, newer):
        if not OID.fullmatch(older or "") or not OID.fullmatch(newer or ""):
            raise Contradiction("IMMUTABLE_REVISION_REQUIRED")
        self.command("cat-file", "-e", older + "^{commit}")
        self.command("cat-file", "-e", newer + "^{commit}")
        result = subprocess.run(["git", "merge-base", "--is-ancestor", older, newer],
                                cwd=self.repo, env=self.env, capture_output=True, timeout=20)
        if result.returncode == 1:
            raise Contradiction("ORIGIN_NOT_ANCESTOR")
        if result.returncode:
            raise Unmeasured("ORIGIN_LINEAGE_UNAVAILABLE")

    def relocations(self, head):
        if head in self.map_cache:
            return self.map_cache[head]
        # Content search locates documents; neither their name nor path grants authority.
        # This fixed locator glob is intentional. Untrusted individual blob paths
        # remain literal in blob(); inherited Git pathspec options are already stripped.
        search_env = dict(self.env)
        search_env.pop("GIT_LITERAL_PATHSPECS", None)
        result = subprocess.run(["git", "grep", "-l", "-z", "-F",
                                 '"ExternalConsumerEvidenceRelocation/v1"', head, "--", "*.json"],
                                cwd=self.repo, env=search_env, capture_output=True, timeout=20)
        if result.returncode not in (0, 1):
            raise Unmeasured("RELOCATION_SEARCH_UNAVAILABLE")
        rows = []
        for name in result.stdout.split(b"\0"):
            if not name:
                continue
            path = name.decode().split(":", 1)[1]
            raw, _ = self.blob(head, path)
            doc = strict_json(raw)
            if not isinstance(doc, dict) or doc.get("schema") != "ExternalConsumerEvidenceRelocation/v1":
                continue
            items = doc.get("consumer_artifacts")
            if not isinstance(items, list):
                raise Contradiction("RELOCATION_SHAPE_INVALID")
            for row in items:
                if not isinstance(row, dict):
                    raise Contradiction("RELOCATION_ROW_INVALID")
                rows.append((path, doc.get("source_snapshot"), row))
        self.map_cache[head] = rows
        return rows


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class NativeReader:
    """Only the existing canonical endpoint; no JSON or URL input can supply a pass."""
    def __init__(self, endpoint):
        if endpoint != ENDPOINT:
            raise Unmeasured("UNSUPPORTED_BASE_NATIVE_ENDPOINT")
        self.endpoint = endpoint
        self.key = os.environ.get("MUNERAL_API_KEY", "")
        self.cache = {}
        self.deadline = time.monotonic() + 30

    def request(self, suffix):
        if not re.fullmatch(r"mun_sk_[^\s]+", self.key):
            raise Unmeasured("NATIVE_CREDENTIAL_UNAVAILABLE")
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise Unmeasured("NATIVE_GET_TIME_BOUND")
        req = urllib.request.Request(self.endpoint + suffix, headers={
            "Authorization": "Bearer " + self.key, "User-Agent": "aup-receipt-data/1.0"})
        try:
            with urllib.request.build_opener(NoRedirect()).open(req, timeout=min(10, left)) as response:
                if response.status != 200:
                    raise Unmeasured("NATIVE_GET_NON_200")
                chunks, size = [], 0
                while True:
                    if time.monotonic() >= self.deadline:
                        raise Unmeasured("NATIVE_GET_TIME_BOUND")
                    chunk = response.read1(min(65536, MAX_NATIVE + 1 - size))
                    if time.monotonic() >= self.deadline:
                        raise Unmeasured("NATIVE_GET_TIME_BOUND")
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > MAX_NATIVE:
                        raise Unmeasured("NATIVE_GET_BYTE_BOUND")
                return strict_json(b"".join(chunks))
        except (urllib.error.URLError, TimeoutError, OSError):
            # Exceptions/body/headers can contain credentials; retain only a stable diagnosis.
            raise Unmeasured("NATIVE_GET_UNAVAILABLE_OR_REDIRECT") from None

    def task_and_evidence(self, task):
        if not isinstance(task, str) or not UUID.fullmatch(task):
            raise Contradiction("NATIVE_TASK_UUID_REQUIRED")
        if task not in self.cache:
            self.cache[task] = self.request("/tasks/" + task), self.request("/tasks/" + task + "/evidence")
        return self.cache[task]


def base_policy(git, base, head):
    before = git.blob(base, POLICY_PATH)
    if before is None:
        raise Unmeasured("BASE_NATIVE_POLICY_MISSING")
    after = git.blob(head, POLICY_PATH)
    if after is None or before != after:
        raise Unmeasured("NATIVE_POLICY_COEDIT_OR_REMOVAL")
    doc = strict_json(before[0])
    try:
        policy = doc["work_item_evidence"]
        endpoint = policy["muneral"]["base_url"]
        pattern = policy["ledger_path"]
    except (KeyError, TypeError):
        raise Unmeasured("BASE_NATIVE_POLICY_UNSUPPORTED") from None
    if endpoint != ENDPOINT or not isinstance(pattern, str) or pattern.count("<work-item-id>") != 1:
        raise Unmeasured("BASE_NATIVE_POLICY_UNSUPPORTED")
    safe_path(pattern.replace("<work-item-id>", "task"))
    return endpoint, pattern


def native_binding(reader, task, attachment, digest):
    actual_task, index = reader.task_and_evidence(task)
    if not isinstance(actual_task, dict) or actual_task.get("id") != task:
        raise Contradiction("NATIVE_TASK_ID_MISMATCH")
    creator = actual_task.get("createdById")
    if not isinstance(creator, str) or not UUID.fullmatch(creator):
        raise Unmeasured("NATIVE_CREATOR_ID_UNAVAILABLE")
    ref = attachment.get("evidence_ref")
    delivery = attachment.get("delivery")
    if not isinstance(ref, dict) or not isinstance(delivery, dict):
        raise Contradiction("LEDGER_NATIVE_BINDING_INVALID")
    if not isinstance(index, dict) or index.get("task_id") != task or not isinstance(index.get("evidence"), list):
        raise Contradiction("NATIVE_EVIDENCE_INDEX_INVALID")
    matches = [e for e in index["evidence"] if isinstance(e, dict) and e.get("evidence_id") == delivery.get("evidence_id")]
    if len(matches) != 1:
        raise Unmeasured("NATIVE_EVIDENCE_ID_UNAVAILABLE_OR_AMBIGUOUS")
    e = matches[0]
    expected = {"task_id": task, "uri": ref.get("uri"), "sha256": digest,
                "content_type": "application/json", "created_by_agent_id": creator}
    if not isinstance(expected["uri"], str) or not expected["uri"] or len(expected["uri"]) > 512:
        raise Contradiction("LEDGER_URI_INVALID")
    if any(e.get(k) != v for k, v in expected.items()) or ref.get("contentType") != "application/json":
        raise Contradiction("NATIVE_EVIDENCE_TUPLE_MISMATCH")
    return {"evidence_id": e["evidence_id"], **expected,
            "project_id": actual_task.get("projectId"), "parent_id": actual_task.get("parentId")}


def measure(git, base, head, path, reader=None):
    """A tri-valued DATA result. Reader injection is internal unit-test transport, never CLI input."""
    out = {"schema": "ReceiptDataMeasurement/v1", "scope": SCOPE, "entity_path": path,
           "base": base, "head": head}
    try:
        out.update(_measure(git, base, head, path, reader))
        out.update(verdict="verified", reason="ARCHIVED_DATA_INTEGRITY_MEASURED; foreign source claim unchanged")
    except Contradiction as exc:
        out.update(verdict="failed", reason=str(exc))
    except (Unmeasured, subprocess.TimeoutExpired) as exc:
        out.update(verdict="not_measured", reason=str(exc) if isinstance(exc, Unmeasured) else "GIT_TIME_BOUND")
    return out


def _measure(git, base, head, path, reader):
    safe_path(path)
    if not isinstance(base, str) or not isinstance(head, str) or not OID.fullmatch(base) or not OID.fullmatch(head):
        raise Unmeasured("IMMUTABLE_ARCHIVE_RANGE_UNAVAILABLE")
    try:
        git.ancestor(base, head)
    except Contradiction as exc:
        if str(exc) == "ORIGIN_NOT_ANCESTOR":
            raise Contradiction("BASE_NOT_ANCESTOR") from None
        raise
    endpoint, pattern = base_policy(git, base, head)
    maps = [m for m in git.relocations(head) if path in (m[2].get("old_path"), m[2].get("new_path"))]
    if len(maps) > 1:
        raise Contradiction("AMBIGUOUS_ARCHIVE_ORIGIN")
    if not maps:
        if git.blob(head, path) is None and git.blob(base, path) is not None:
            raise Contradiction("ARCHIVE_RECEIPT_REMOVED")
        raise Unmeasured("ARCHIVE_ORIGIN_UNAVAILABLE")
    index_path, original_rev, mapping = maps[0]
    old, current = safe_path(mapping.get("old_path")), safe_path(mapping.get("new_path"))
    if original_rev == head or old == current:
        raise Contradiction("ARCHIVE_ORIGIN_SELF_LICENSED")
    git.ancestor(original_rev, head)
    original = git.blob(original_rev, old)
    now = git.blob(head, current)
    if original is None:
        raise Unmeasured("ORIGINAL_RECEIPT_UNAVAILABLE")
    if now is None:
        raise Contradiction("ARCHIVE_RECEIPT_REMOVED")
    raw, oid = now
    digest = hashlib.sha256(raw).hexdigest()
    if raw != original[0] or digest != mapping.get("sha256"):
        raise Contradiction("ARCHIVE_BYTES_OR_DIGEST_MISMATCH")
    if path == old and git.blob(base, old) not in (None, original):
        raise Contradiction("DELETED_BASE_RECEIPT_NOT_PRESERVED")
    doc = strict_json(raw)
    if not isinstance(doc, dict) or doc.get("schema") != "ChangeAdmissionReceipt/v1":
        raise Unmeasured("UNSUPPORTED_ARCHIVED_RECEIPT_SCHEMA")
    findings = schema_check.classify(doc)
    if findings["verdict"] != "conformant":
        raise Contradiction("ARCHIVE_SCHEMA_VIOLATION:" + ",".join(findings["codes"]))
    repo, source = doc.get("repo"), doc.get("change_set")
    if not isinstance(repo, dict) or not isinstance(source, dict):
        raise Contradiction("FOREIGN_SOURCE_SHAPE_INVALID")
    if source.get("mode") != "diff" or not OID.fullmatch(source.get("base") or "") or not OID.fullmatch(source.get("head") or ""):
        raise Unmeasured("IMMUTABLE_FOREIGN_SOURCE_RANGE_UNAVAILABLE")
    if repo.get("name") != mapping.get("consumer_repo") or source.get("head") != mapping.get("consumer_head"):
        raise Contradiction("FOREIGN_SOURCE_IDENTITY_MISMATCH")
    task = doc.get("work_item")
    if not isinstance(task, str) or not UUID.fullmatch(task):
        raise Unmeasured("ARCHIVE_NATIVE_WORK_ITEM_UNAVAILABLE")
    ledger_path = pattern.replace("<work-item-id>", task)
    ledger_blob = git.blob(head, ledger_path)
    if ledger_blob is None:
        raise Unmeasured("ARCHIVE_LEDGER_UNAVAILABLE")
    ledger = strict_json(ledger_blob[0])
    if not isinstance(ledger, dict) or not isinstance(ledger.get("attachments"), list):
        raise Contradiction("ARCHIVE_LEDGER_INVALID")
    matches = [a for a in ledger["attachments"] if isinstance(a, dict)
               and isinstance(a.get("evidence_ref"), dict)
               and a["evidence_ref"].get("digest") == "sha256:" + digest]
    if len(matches) != 1:
        raise Unmeasured("ARCHIVE_LEDGER_MATCH_UNAVAILABLE_OR_AMBIGUOUS")
    att = matches[0]
    wi = att.get("work_item")
    if not isinstance(wi, dict):
        raise Contradiction("ARCHIVE_LEDGER_SUBJECT_MISMATCH")
    if (att.get("schema") != "WorkItemEvidenceAttachment/v1" or att.get("attachment_kind") != "evidence"
            or att.get("receipt_path") != current or att.get("receipt_id") != doc.get("receipt_id")
            or wi.get("system") != "muneral" or wi.get("task_id") != task or wi.get("muneral_task_id") != task):
        raise Contradiction("ARCHIVE_LEDGER_SUBJECT_MISMATCH")
    binding = native_binding(reader or NativeReader(endpoint), task, att, digest)
    return {"physical_path": current, "blob": oid, "sha256": digest,
            "origin": {"revision": original_rev, "path": old, "blob": original[1]},
            "relocation_index": index_path, "native_binding": binding,
            "foreign_claim": {"repo": repo.get("name"), "base": source.get("base"), "head": source.get("head"),
                              "admission": doc.get("admission"), "verdicts": doc.get("verdicts")},
            "schema_verdict": findings["verdict"]}


def rederive_claims(repo, doc, candidate_head=None):
    """Gate check: actual native/Git measurement, independent of author output/readback JSON."""
    problems = []
    source = doc.get("change_set") or {}
    base, head = source.get("base"), source.get("head")
    git = GitObjects(repo)
    rows = {v.get("id"): v for v in doc.get("verifiers", []) if isinstance(v, dict)}
    for verdict in doc.get("verdicts", []):
        if not isinstance(verdict, dict) or verdict.get("verdict") != "verified":
            continue
        for vid in verdict.get("verifier_ids", []):
            row = rows.get(vid, {})
            if row.get("kind") != "receipt_data":
                continue
            entity = verdict.get("entity", "")
            claims = row.get("measurements")
            if (row.get("scope") != SCOPE or not entity.startswith("receipt:")
                    or not isinstance(claims, dict) or entity not in row.get("entities", [])):
                problems.append("receipt_data scope/subject invalid: " + entity)
                continue
            actual = measure(git, base, head, entity[len("receipt:"):])
            if actual.get("verdict") != "verified" or claims.get(entity) != actual:
                problems.append("receipt_data independent rederivation refused: " + entity + ": " + actual["reason"])
                continue
            if candidate_head is not None and candidate_head != head:
                current = measure(git, base, candidate_head, entity[len("receipt:"):])
                # A record commit may extend the range, never alter its measured data.
                original_fields = {k: v for k, v in actual.items() if k != "head"}
                current_fields = {k: v for k, v in current.items() if k != "head"}
                if current.get("verdict") != "verified" or original_fields != current_fields:
                    problems.append("receipt_data candidate bytes/bindings differ: " + entity)
    return problems
