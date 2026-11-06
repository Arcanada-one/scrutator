"""Run the AC5 model tests and write REPORT.md (MODEL-ONLY, synthetic). Usage: python run_and_report.py OUT.md"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CASES = {
    "test_c01_complete_generation_and_byte_readback": "C01 complete generation + byte readback",
    "test_c02_one_member_missing": "C02 one member missing",
    "test_c03_stale_digest": "C03 stale digest",
    "test_c04_tampered_bytes_same_length": "C04 tampered bytes",
    "test_c05_non_current_or_foreign_source": "C05 non-current / foreign source",
    "test_c06_missing_dependency": "C06 missing dependency",
    "test_c07_wrong_project": "C07 wrong project",
    "test_c08_revoked_reader_denied_even_with_cached_bytes": "C08 revoked reader (cached bytes remain)",
    "test_c09_competing_promotion_cas_single_winner": "C09 concurrent promotion / CAS race (deterministic)",
    "test_c09b_threaded_race_has_exactly_one_winner": "C09b CAS race, 4 threads x 50 rounds",
    "test_c10_interrupted_after_first_new_member_previous_generation_intact": "C10 interrupted after first new member (77307111 counterexample)",
    "test_c10b_recovery_after_interruption_retry_completes": "C10b retry/recovery after interruption",
    "test_c11_repeated_check_is_idempotent_and_read_only": "C11 repeated idempotent check",
    "test_c12_no_cross_namespace_reads_or_writes": "C12 no cross-namespace reads/writes",
    "test_r1a_same_id_different_manifest_refused_before_any_state_change": "R1a same ID / different manifest refused, no state change",
    "test_r1b_same_manifest_different_bytes_refused_before_any_state_change": "R1b same manifest / different bytes refused, no state change",
    "test_r1c_interrupted_same_id_rewrite_changes_nothing": "R1c interrupted same-ID rewrite changes nothing",
    "test_r1d_conflict_on_staged_candidate_is_prechecked_for_all_members": "R1d conflict pre-checked for all members (no partial write)",
    "test_r1e_exact_retry_only_observes_or_fills_missing": "R1e exact retry observes identical bytes / fills missing only",
    "test_r2a_aba_stale_observation_refused_by_version": "R2a bounded ABA trace (g1->g2->g1) refused by version",
    "test_r2b_served_read_refuses_staged_generation_diagnostic_read_is_labelled": "R2b served read refuses staged generation; diagnostic read labelled",
    "test_r2c_transitive_dependency_chain_gap_refused": "R2c transitive dependency chain gap refused",
    "test_ac4_incomplete_candidate_never_visible_in_accepted_view": "AC4 incomplete candidate absent from accepted view",
    "test_checker_is_not_vacuous_every_faulty_variant_rejected": "META checker rejects every faulty variant",
}


SUBCASES = [
    (
        "Immutable generation ID: same ID, different manifest",
        "test_r1a_same_id_different_manifest_refused_before_any_state_change",
        "OverwriteStore",
    ),
    (
        "Immutable member bytes: same manifest, different bytes",
        "test_r1b_same_manifest_different_bytes_refused_before_any_state_change",
        "OverwriteStore",
    ),
    ("Interrupted same-ID rewrite", "test_r1c_interrupted_same_id_rewrite_changes_nothing", "OverwriteStore"),
    (
        "Conflict detected before any write (no partial state)",
        "test_r1d_conflict_on_staged_candidate_is_prechecked_for_all_members",
        "CheckAsYouGoStore",
    ),
    (
        "Exact retry: observe identical / fill missing only",
        "test_r1e_exact_retry_only_observes_or_fills_missing",
        "(positive control; R1a-d mutants cover the negative side)",
    ),
    (
        "Bounded ABA g1->g2->g1, stale (ptr, version) observation",
        "test_r2a_aba_stale_observation_refused_by_version",
        "PointerOnlyCasStore",
    ),
    (
        "Served read of a staged generation",
        "test_r2b_served_read_refuses_staged_generation_diagnostic_read_is_labelled",
        "UnguardedReadStore",
    ),
    (
        "Transitive dependency chain (a->b->c, c absent)",
        "test_r2c_transitive_dependency_chain_gap_refused",
        "RootOnlyDepStore",
    ),
    (
        "Pre-ranking search isolation (candidate chunks excluded BEFORE ranking, not filtered after)",
        None,
        "model has no ranker; only the post-selection accepted view (AC4) is modelled",
    ),
    (
        "Effect-time external revocation (grant revoked between check and effect)",
        None,
        "revocation is checked at call time only; no check-then-use window modelled",
    ),
    (
        "Durable lost-ack reconciliation (promotion committed, ack lost, caller retries)",
        None,
        "only a stale-expectation retry is refused (C11); recognising one's own committed win is not modelled",
    ),
    (
        "Search top-k isolation (candidate chunks cannot displace accepted chunks in the top-k)",
        None,
        "no ranker / top-k in the model",
    ),
    (
        "Revocation between authorization and disclosure (check-then-disclose window)",
        None,
        "authorization and read are one synchronous step in the model",
    ),
    (
        "External-authority outage / fence (Auth Arcana unreachable, fail-closed)",
        None,
        "the authorization oracle never fails in the model",
    ),
    (
        "Complete versioned closure (dependency pins a specific revision, not just an artifact id)",
        None,
        "deps are artifact ids only; revision-pinned deps not modelled",
    ),
    ("Cyclic dependency closure", None, "not tested"),
    ("Multibyte / non-ASCII member content (byte length vs character length)", None, "all fixtures are ASCII"),
    (
        "Shared member across generations (same artifact+revision in two generations)",
        None,
        "revision is generation-qualified in all fixtures",
    ),
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(out: str) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ac5-"))
    junit, fj = tmp / "j.xml", tmp / "f.json"
    cmd = [sys.executable, "-m", "pytest", "tests/ac5_model", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"]
    env = dict(os.environ, PYTHONPATH="src", AC5_FAULTY_JSON=str(fj))
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    rows = {}
    for tc in ET.parse(junit).getroot().iter("testcase"):
        st = (
            "FAIL"
            if tc.find("failure") is not None or tc.find("error") is not None
            else ("NOT_MEASURED" if tc.find("skipped") is not None else "PASS")
        )
        rows[tc.get("name")] = st
    faulty = json.loads(fj.read_text()) if fj.exists() else {}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    base = subprocess.run(["git", "rev-parse", "origin/main"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    L = [
        "# KB-CONS-01-M — AC5 generation-closure model: REPORT",
        "",
        "**Label: MODEL-ONLY, SYNTHETIC.** In-memory model with synthetic member IDs and bytes. No database, no network, no deployed "
        "Scrutator, no kc-corpus profile/grants, no producer code, no shared/public namespace was touched. "
        "A PASS here is a model result, **not** a live AC5 result, not an acceptance of KB-CONS-01, not a readiness claim.",
        "",
        "## Case table (AC1)",
        "",
        "| Case | Result |",
        "|---|---|",
    ]
    for k, label in CASES.items():
        L.append(f"| {label} | {rows.get(k, 'NOT_MEASURED')} |")
    L += [
        "",
        "## Subcase table: MEASURED vs NOT_MEASURED (R2)",
        "",
        "MEASURED = a model test above passed AND a deliberately broken variant is rejected by its checker. "
        "NOT_MEASURED = no test exists in this PR; it must not be read as pass or fail.",
        "",
        "| Subcase | Status | Rejected mutant / reason NOT_MEASURED |",
        "|---|---|---|",
    ]
    for label, test, mut in SUBCASES:
        L.append(f"| {label} | {rows.get(test, 'NOT_MEASURED') if test else 'NOT_MEASURED'} | {mut} |")
    L += [
        "",
        "Direct reads: `Store.read_member` is the SERVED path and refuses any never-selected (staged) generation "
        "(`not-selected-generation`); `Store.diagnostic_read_member` can read a staged generation and is "
        "**diagnostic-only**, never to be exposed by a served API (R2b).",
        "",
        "Extra rows not exercised by the model (never read as PASS):",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Real Postgres transaction / advisory-lock behaviour of `replace_source_chunks_atomic` | NOT_MEASURED |",
        "| Crash recovery of a real process / DB (only an injected exception is modelled) | NOT_MEASURED |",
        "| Real Auth Arcana / `principal_namespace_grants` revocation latency | NOT_MEASURED |",
        "| Live AC5 on an accepted pilot generation (needs KCLIVE-07 accepted source, KCLIVE-08 assembler) | NOT_MEASURED |",
        "",
        "## Faulty variants rejected by the checker (AC2, AC3, AC4)",
        "",
        "Each faulty variant was run through the same checkers; `rejected=True` means the checker disagreed with it.",
        "",
        "| Case/variant | Rejected | First divergent invariant | Detail |",
        "|---|---|---|---|",
    ]
    for k, v in sorted(faulty.items()):
        L.append(f"| {k} | {v['rejected']} | `{v['invariant']}` | {v['detail']} |")
    L += [
        "",
        "Mapping: `InPlaceStore` = overwrite-in-place of the active generation's source paths (AC2, 77307111 counterexample; "
        "`I1` previous accepted member no longer reads at its old digest). `LastWriterWinsStore` = promotion ignoring the expected "
        "pointer (AC3; `I5`). `LeakyViewStore` = candidate chunks visible in the accepted view (AC4; `I3`). `CachedViewStore` = old "
        "complete bytes served after revocation (AC4; `I4`). `NoValidationStore` = metadata-only promotion (`I2`/`I7`). "
        "`CrossNsStore` = write mirrored to another namespace (`I6`).",
        "",
        "## Provenance",
        "",
        f"- Scrutator base: `origin/main` = `{base}` (required: at or after 0cb546a3). Branch head at run time: `{head}` (working tree may carry uncommitted report).",
        f"- Source sha256: `generation_model.py` = `{sha(HERE / 'generation_model.py')}`",
        f"- Fixture/test sha256: `test_generation_closure_model.py` = `{sha(HERE / 'test_generation_closure_model.py')}`; "
        f"`conftest.py` = `{sha(HERE / 'conftest.py')}`; `run_and_report.py` = `{sha(HERE / 'run_and_report.py')}`",
        "- Fixtures are inline synthetic byte strings (`OLD-*`, `NEW-*`); no external fixture file exists.",
        "- Command: `cd scrutator && PYTHONPATH=src python -m pytest tests/ac5_model -q -p no:cacheprovider` (via `python tests/ac5_model/run_and_report.py REPORT.md`)",
        f"- Exit code: **{r.returncode}**",
        f"- Environment: Python {platform.python_version()}, {platform.platform()}, host `{platform.node()}`; "
        f"pytest summary: `{r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}`",
        "",
        "## Model assumptions",
        "",
        "1. Storage is an in-memory dict keyed by (namespace, key); immutable staging = generation-qualified key `gen/artifact@revision`. "
        "The real mapping to Scrutator `source_path` / `evidence_documents` rows is not implemented.",
        "2. The generation pointer is a single in-memory register guarded by a `threading.Lock` with a version counter; this stands in "
        "for a native atomic CAS that **does not exist in Scrutator today** (see below).",
        "3. Authorization is a set of (reader, namespace) grants consulted on every read; no JWT, token lifetime, cache TTL or "
        "propagation delay is modelled. Revocation is instantaneous.",
        "4. Failure is an injected Python exception after N writes; no process kill, partial page write, WAL, or restart is modelled.",
        "5. The competing-promotion case interleaves deterministically (both validate, then both promote) plus a 4-thread barrier "
        "stress; it does not explore all schedules.",
        "6. Digest domain is SHA-256 over raw bytes + length. Document-ID, revision, receipt and source-hash domains are reduced to "
        "(artifact_id, revision, source_pin); their distinctness is modelled only by separate fields.",
        "7. Dependency closure is a finite same-manifest check; a transitive chain is exercised (R2c); cyclic closure is not modelled.",
        "8b. Immutable identity (R1) is enforced by the model's staging step; the real storage layer must provide the same guarantee (unique/immutable key per generation member). Not implemented here.",
        "8. Old-generation retention/cleanup is out of scope; the model never deletes a generation.",
        "",
        "## What this model does NOT prove",
        "",
        "- That Scrutator's `index_documents()/_persist_prepared()` / `replace_source_chunks_atomic()` behave as the safe model; "
        "the unsafe `InPlaceStore` is a model of the hazard named in 77307111, not a measurement of the real code.",
        "- Real DB transaction semantics, isolation level, lock behaviour, or crash recovery.",
        "- Real current authorization (Auth Arcana, `principal_namespace_grants`, revocation propagation).",
        "- Any live AC5 behaviour, deployment, accepted pilot generation, or the real consumer fetch path (`fetcher.py`, `content_exact`).",
        "- Cross-namespace isolation of the real service (only the model's own I/O log is checked).",
        "",
        "## Native generation-switch mechanism in Scrutator today",
        "",
        "**None found** at `0cb546a3`. Search of `src/` (including `db/migrations/001..006`) finds no generation/manifest/pointer table or "
        "compare-and-swap selection; `replace_source_chunks_atomic()` replaces one `(namespace_id, source_path)` under an advisory lock "
        "per source, and `index_documents()` persists documents independently with mixed results. Existing `CAS` hits are NULL-only "
        "entity repair tools unrelated to source generations. **Blocker for the real implementation:** a native atomic CAS "
        "generation-switch (pointer table + versioned compare-and-set against the expected current pointer, plus generation-qualified "
        "storage identities so staged members never reuse the active generation's `source_path`) must be designed and added; the "
        "validator alone cannot prevent a writer overwriting the previous generation.",
        "",
    ]
    Path(out).write_text("\n".join(L))
    sys.stdout.write(r.stdout[-600:])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
