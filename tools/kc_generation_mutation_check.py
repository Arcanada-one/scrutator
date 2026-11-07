#!/usr/bin/env python3
"""KB-CONS-01-N mutation controls.

For every deliberately broken variant of the generation-selection primitive the targeted native
PostgreSQL tests MUST fail. The pristine tree is run first and MUST pass, otherwise the control is void.

Each mutant is applied to a throw-away COPY of ``src/scrutator`` (the repository tree is never edited);
``PYTHONPATH`` puts the copy first so the tests exercise the mutated code and the mutated schema.sql.

    KC_TEST_PG_ADMIN_DSN=postgresql://postgres@127.0.0.1:<port>/postgres \
        python3 tools/kc_generation_mutation_check.py --out mutation-report.json [--only M01,M07] [--dry-run]

Exit 0 only if the pristine run passes and every mutant is KILLED (the suite failed on it).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "scrutator"
KCG = "db/kc_generations.py"
REPOSITORY = "db/repository.py"
SCHEMA = "db/schema.sql"
DEP = "auth/dependency.py"

T_SEL = "tests/kc_generations/test_seal_and_selection.py"
T_REC = "tests/kc_generations/test_records_and_staging.py"
T_READ = "tests/kc_generations/test_read_isolation_and_retention.py"


@dataclass
class Mutant:
    mid: str
    desc: str
    file: str
    old: str
    new: str
    tests: list[str]
    k: str
    region: tuple[str, str] | None = None  # (start marker, end marker): replace only inside it
    occurrence: int = 0  # which match (0-based) when no region is given
    replace_all: bool = False  # every match (hybrid_search has a 2-way and a 3-way SQL branch)


MUTANTS = [
    Mutant(
        "M01",
        "ABA: CAS drops the revision predicate (value-only fence)",
        KCG,
        "AND revision=$3 ",
        "AND $3::bigint IS NOT NULL ",
        [T_SEL],
        "AC5_09 or R_SEL_06",
    ),
    Mutant(
        "M02",
        "CAS drops the incarnation predicate",
        KCG,
        "incarnation_id=$2::uuid AND",
        "$2::uuid IS NOT NULL AND",
        [T_SEL],
        "AC5_09_scope_incarnation",
    ),
    Mutant(
        "M03",
        "CAS drops the prior-generation predicate",
        KCG,
        "AND generation_id IS NOT DISTINCT FROM $6 ",
        "AND ($6::text IS NULL OR TRUE) ",
        [T_SEL],
        "R_SEL_05",
    ),
    Mutant(
        "M04",
        "pointer-only update: operation row not written (code)",
        KCG,
        "VALUES ($1::uuid,$2,$3,$4,$5,$6::uuid,$7,$8,$9::uuid,$10,$11,$12::jsonb)",
        "SELECT $1::uuid,$2,$3,$4,$5,$6::uuid,$7,$8,$9::uuid,$10,$11,$12::jsonb WHERE false",
        [T_SEL],
        "R_SEL_01 or AC5_11 or AC5_09_two",
    ),
    Mutant(
        "M05",
        "promotion without the sealed-candidate check",
        KCG,
        'if gen["state"] != "sealed" or gen["seal_report"] is None:',
        "if False:",
        [T_SEL],
        "R_SEL_04 or AC5_02_missing_member_refused",
    ),
    Mutant(
        "M06",
        "promotion without current authority",
        KCG,
        'obs = await _check_authority(backend, conn, principal_id=principal_id, scope_id=scope_id, purpose="select")',
        'obs = AuthorityObservation("mut", principal_id, scope_id, "select", "1", "w")',
        [T_SEL],
        "R_AUTH or AC5_05_withdrawn",
    ),
    Mutant(
        "M07",
        "semantic ranking arm without the generation predicate (hybrid_search)",
        REPOSITORY,
        "FROM {arm_rel} c\n                    WHERE c.namespace_id = $2\n                      AND c.embedding_dense IS NOT NULL",
        "FROM chunks c\n                    WHERE c.namespace_id = $2\n                      AND c.embedding_dense IS NOT NULL",
        [T_READ],
        "each_ranking_arm and semantic",
        replace_all=True,
    ),
    Mutant(
        "M08",
        "full-text ranking arm without the generation predicate (hybrid_search)",
        REPOSITORY,
        "FROM {arm_rel} c\n                    WHERE c.namespace_id = $2\n                      AND (c.textsearch_ru",
        "FROM chunks c\n                    WHERE c.namespace_id = $2\n                      AND (c.textsearch_ru",
        [T_READ],
        "each_ranking_arm and fulltext",
        replace_all=True,
    ),
    Mutant(
        "M09",
        "sparse ranking arm without the generation predicate (hybrid_search)",
        REPOSITORY,
        "JOIN {arm_rel} c ON c.id = sv.chunk_id",
        "JOIN chunks c ON c.id = sv.chunk_id",
        [T_READ],
        "each_ranking_arm and sparse",
    ),
    Mutant(
        "M10",
        "filtered search arm without the generation predicate",
        REPOSITORY,
        "FROM {arm_rel} c\n                WHERE c.namespace_id = $2\n                  AND c.embedding_dense IS NOT NULL",
        "FROM chunks c\n                WHERE c.namespace_id = $2\n                  AND c.embedding_dense IS NOT NULL",
        [T_READ],
        "filtered_search",
    ),
    Mutant(
        "M11",
        "current-use check skips the authority backend",
        KCG,
        "return await _check_authority(backend, None, principal_id=principal_id, scope_id=ctx.scope_id, purpose=purpose)",
        'return AuthorityObservation("mut", principal_id, ctx.scope_id, purpose, "1", "w")',
        [T_READ],
        "AC5_08 or R_CTX_04",
    ),
    Mutant(
        "M12",
        "use ignores lease expiry",
        KCG,
        'if row["expired"]:',
        "if False:",
        [T_READ],
        "R_CTX_04",
        region=("async def authorize_use", "async def renew_lease"),
    ),
    Mutant(
        "M13",
        "seal skips the raw body byte/hash readback",
        KCG,
        'if len(raw) != r["expected_bytes"] or sha256_hex(raw) != r["expected_sha256"]:',
        "if False:",
        [T_SEL],
        "AC5_04_mutated_raw or AC5_02_missing_body",
        region=("async def seal_generation", "# ── promotion"),
    ),
    Mutant(
        "M14",
        "seal skips the per-chunk recomputed hash",
        KCG,
        '                    or c["content_hash"] != sha256_hex(c["content"].encode("utf-8"))\n',
        "",
        [T_SEL],
        "AC5_04_mutated_chunk_body",
    ),
    Mutant(
        "M15",
        "seal skips the bound chunk-set digest",
        KCG,
        '!= r["chunk_set_sha256"]',
        '!= r["chunk_set_sha256"] and False',
        [T_SEL],
        "AC5_04_chunk_set_digest",
        region=("async def seal_generation", "# ── promotion"),
    ),
    Mutant(
        "M16",
        "renewal revives an expired lease",
        KCG,
        'if lease["expired"]:',
        "if False:",
        [T_READ],
        "R_CTX_04 or R_GC_05",
    ),
    Mutant(
        "M17",
        "GC ignores valid in-flight leases",
        KCG,
        "if leased:",
        "if False:",
        [T_READ],
        "R_GC_01 or R_GC_04 or R_GC_06",
        region=("async def mark_gc_pending", "async def collect_generation"),
    ),
    Mutant(
        "M18",
        "existing generation identity is not compared (silent reuse)",
        KCG,
        'bytes(existing["manifest"]) != bytes(manifest)',
        "False",
        [T_REC],
        "R_STG_02",
    ),
    Mutant(
        "M19",
        "unknown required relation accepted",
        KCG,
        'if e["relation"] not in _KNOWN_RELATIONS:',
        "if False:",
        [T_REC],
        "R_STG_06",
    ),
    Mutant(
        "M20",
        "legacy routes keep private namespaces visible",
        DEP,
        "return namespace_ids - frozenset(hidden or ())",
        "return namespace_ids",
        [T_READ],
        "legacy_routes",
    ),
    Mutant(
        "M21",
        "exact fetch skips the byte re-hash",
        KCG,
        'if len(raw) != row["expected_bytes"] or sha256_hex(raw) != row["expected_sha256"]:',
        "if False:",
        [T_READ],
        "R_CTX_07",
        region=("async def read_member_exact", "async def read_chunk"),
    ),
    Mutant(
        "M22",
        "related view needs only one endpoint inside the generation",
        KCG,
        "JOIN eligible s ON s.id = g.source_chunk_id JOIN eligible t ON t.id = g.target_chunk_id",
        "JOIN chunks s ON s.id = g.source_chunk_id JOIN chunks t ON t.id = g.target_chunk_id",
        [T_READ],
        "related_view",
    ),
    Mutant(
        "M23",
        "a populated sealed member can be restaged (overwrite)",
        KCG,
        'if m["chunk_set_sha256"] is not None:',
        "if False:",
        [T_REC],
        "R_STG_03 or R_REC_04",
    ),
    Mutant(
        "M24",
        "DB backstop: selection change needs no operation row",
        SCHEMA,
        "IF NOT EXISTS (\n        SELECT 1 FROM kc_selection_operations o",
        "IF FALSE AND NOT EXISTS (\n        SELECT 1 FROM kc_selection_operations o",
        [T_SEL],
        "R_SEL_02",
    ),
    Mutant(
        "M25",
        "DB backstop: revision may reset or skip",
        SCHEMA,
        "IF NEW.revision <> OLD.revision + 1 THEN",
        "IF FALSE THEN",
        [T_SEL],
        "R_SEL_03 or R_SEL_08",
    ),
    Mutant(
        "M26",
        "DB backstop: a populated member may be replaced in place",
        SCHEMA,
        "IF OLD.chunk_set_sha256 IS NOT NULL AND (NEW.chunk_set_sha256",
        "IF FALSE AND (NEW.chunk_set_sha256",
        [T_REC],
        "R_REC_03",
    ),
    Mutant(
        "M27",
        "DB backstop: private paths unprotected (guard returns early)",
        SCHEMA,
        "    IF v_state IS NULL THEN\n        RAISE EXCEPTION 'KC_PRIVATE_PATH",
        "    IF TRUE THEN RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END; END IF;\n    IF v_state IS NULL THEN\n        RAISE EXCEPTION 'KC_PRIVATE_PATH",
        [T_REC],
        "R_REC_05 or R_REC_06",
    ),
    Mutant(
        "M28",
        "default authority backend allows (no real authority)",
        KCG,
        'raise KcGenerationError(\n            "CURRENT_AUTHORITY_UNAVAILABLE", "no explicit private-use authority backend is configured"\n        )',
        'return AuthorityObservation("mut", principal_id, scope_id, purpose, "1", "w")',
        [T_SEL, T_READ],
        "R_AUTH_01 or AC5_08_authority_outage",
    ),
]


def apply(m: Mutant, root: Path) -> None:
    path = root / m.file
    text = path.read_text()
    if m.region:
        a = text.index(m.region[0])
        b = text.index(m.region[1], a)
        seg = text[a:b]
        if seg.count(m.old) < 1:
            raise SystemExit(f"{m.mid}: pattern not found inside region")
        text = text[:a] + seg.replace(m.old, m.new, -1 if m.replace_all else 1) + text[b:]
    else:
        if m.replace_all:
            if m.old not in text:
                raise SystemExit(f"{m.mid}: pattern not found")
            path.write_text(text.replace(m.old, m.new))
            return
        idx = -1
        for _ in range(m.occurrence + 1):
            idx = text.find(m.old, idx + 1)
            if idx < 0:
                raise SystemExit(f"{m.mid}: pattern not found")
        text = text[:idx] + m.new + text[idx + len(m.old) :]
    path.write_text(text)


def run_tests(pythonpath: str, tests: list[str], k: str | None) -> tuple[int, str]:
    env = dict(os.environ, PYTHONPATH=pythonpath)
    cmd = [sys.executable, "-m", "pytest", *tests, "-q", "-x", "-p", "no:cacheprovider"]
    if k:
        cmd += ["-k", k]
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-1500:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mutation-report.json")
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true", help="only check that every pattern applies")
    args = ap.parse_args()
    if not os.environ.get("KC_TEST_PG_ADMIN_DSN") and not args.dry_run:
        print("KC_TEST_PG_ADMIN_DSN is required", file=sys.stderr)
        return 2
    only = {x for x in args.only.split(",") if x}
    chosen = [m for m in MUTANTS if not only or m.mid in only]
    report: dict = {
        "repo_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip(),
        "mutants": [],
    }
    if not args.dry_run:
        t0 = time.time()
        code, tail = run_tests(f"{REPO / 'src'}:{REPO}", [T_SEL, T_REC, T_READ], None)
        report["pristine"] = {"exit": code, "seconds": round(time.time() - t0, 1), "tail": tail[-300:]}
        print(f"pristine exit={code}")
        if code != 0:
            Path(args.out).write_text(json.dumps(report, indent=2))
            return 1
    survived = 0
    for m in chosen:
        tmp = Path(tempfile.mkdtemp(prefix=f"kcmut-{m.mid}-"))
        try:
            shutil.copytree(SRC, tmp / "src" / "scrutator", ignore=shutil.ignore_patterns("__pycache__"))
            apply(m, tmp / "src" / "scrutator")
            if args.dry_run:
                print(f"{m.mid}: applies")
                continue
            t0 = time.time()
            code, tail = run_tests(f"{tmp / 'src'}:{REPO}", m.tests, m.k)
            killed = code != 0
            survived += not killed
            report["mutants"].append(
                {
                    "id": m.mid,
                    "desc": m.desc,
                    "file": m.file,
                    "tests": m.tests,
                    "k": m.k,
                    "exit": code,
                    "killed": killed,
                    "seconds": round(time.time() - t0, 1),
                    "tail": tail[-400:],
                }
            )
            print(f"{m.mid} {'KILLED' if killed else 'SURVIVED'} exit={code} :: {m.desc}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    report["survived"] = survived
    Path(args.out).write_text(json.dumps(report, indent=2))
    return 0 if survived == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
