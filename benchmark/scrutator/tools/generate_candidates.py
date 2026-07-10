#!/usr/bin/env python3
"""Golden-set candidate-generation tooling (Fork 1 — PRD § Technical Approach;
plan Phase 6 Steps 2-4).

Two-LLM-pass disagreement filter, mirroring how the v0 seed (SRCH-0031) was built by hand:

  1. **Pass 1** reads a source document and proposes candidate questions, each tagged
     `class` (factual/multi-hop/temporal) with a stated answer + `gold_source_paths`.
  2. **Pass 2** independently re-answers each candidate's question against the *same*
     source document only — never against search results, which is what keeps the
     golden set non-circular — blind to pass 1's stated answer.
  3. A **judge pass** compares pass 1's and pass 2's answers; candidates where they
     materially disagree are rejected before a human ever reviews them.

HUMAN-ANNOTATION GATE (do not remove): this script writes surviving candidates to a
`*-candidates.jsonl` file for operator review. It does **not** promote candidates to
`gold` and does **not** write directly into any `golden-arcanada-v*.jsonl` file — that
promotion step requires one human (operator) verification pass per row (PRD Fork 1),
which is not automatable. See `tools/README.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mc_client import ModelConnectorClient, ModelConnectorError

VALID_CLASSES = ("factual", "multi-hop", "temporal")

GENERATE_SYSTEM = (
    "You are drafting candidate benchmark questions for a knowledge-base retrieval "
    "recall test. Read the document text given. Propose natural-phrasing questions "
    "that are fully answerable from THIS document alone — never invent facts not in "
    "the text, never phrase a question the way a search engine's output would. "
    'Return ONLY a JSON array: [{"query": "...", "class": "factual|multi-hop|temporal", '
    '"answer": "...", "gold_source_paths": ["..."]}]. "answer" is the fact/date/value '
    "the query is asking about, in your own words, quoting the document where useful."
)

REANSWER_SYSTEM = (
    "You are independently answering a question using ONLY the document text given. "
    "Do not assume any other context. Answer concisely, in your own words. Return ONLY "
    "the answer text, no other commentary."
)

JUDGE_SYSTEM = (
    "You are comparing two independently-produced answers to the same question, both "
    "derived from the same source document. Decide whether they materially agree "
    "(same fact, value, or conclusion, even if phrased differently) or disagree. "
    'Return ONLY JSON: {"agree": true|false, "reason": "..."}.'
)


class Candidate(dict):
    """A candidate row, shaped like a golden-set row plus filter provenance."""


def generate_candidate_questions(client: ModelConnectorClient, doc_path: str, doc_text: str, n: int) -> list[dict]:
    """Pass 1: propose up to `n` candidate questions from one document."""
    prompt = f"Document path: {doc_path}\nNumber of questions requested: {n}\n\nDocument text:\n{doc_text[:8000]}"
    parsed = client.call_json(prompt, system=GENERATE_SYSTEM)
    if not isinstance(parsed, list):
        raise ModelConnectorError(f"expected a JSON array of candidates, got: {type(parsed)}")
    candidates = []
    for item in parsed[:n]:
        if not isinstance(item, dict):
            continue
        if item.get("class") not in VALID_CLASSES:
            continue
        if not item.get("query") or not item.get("answer") or not item.get("gold_source_paths"):
            continue
        candidates.append(dict(item))
    return candidates


def reanswer_question(client: ModelConnectorClient, doc_text: str, question: str) -> str:
    """Pass 2: independently re-derive an answer to `question` from the same document."""
    prompt = f"Question: {question}\n\nDocument text:\n{doc_text[:8000]}"
    return client.call(prompt, system=REANSWER_SYSTEM).strip()


def judge_agreement(client: ModelConnectorClient, question: str, answer_a: str, answer_b: str) -> tuple[bool, str]:
    """Judge pass: do pass-1's and pass-2's answers materially agree?"""
    prompt = f"Question: {question}\n\nAnswer A: {answer_a}\n\nAnswer B: {answer_b}"
    parsed = client.call_json(prompt, system=JUDGE_SYSTEM)
    if not isinstance(parsed, dict) or "agree" not in parsed:
        raise ModelConnectorError(f"expected {{'agree': bool, 'reason': str}}, got: {parsed}")
    return bool(parsed["agree"]), str(parsed.get("reason", ""))


def run_batch(client: ModelConnectorClient, docs: list[tuple[str, str]], n_per_doc: int) -> list[dict]:
    """Run the full 2-pass-plus-judge pipeline over a list of (doc_path, doc_text) pairs.

    Returns only candidates the judge pass marked as agreeing — still pre-human-review.
    """
    survivors = []
    for doc_path, doc_text in docs:
        for candidate in generate_candidate_questions(client, doc_path, doc_text, n_per_doc):
            pass2_answer = reanswer_question(client, doc_text, candidate["query"])
            agree, reason = judge_agreement(client, candidate["query"], candidate["answer"], pass2_answer)
            candidate["pass2_answer"] = pass2_answer
            candidate["judge_agree"] = agree
            candidate["judge_reason"] = reason
            candidate["review_status"] = "candidate"  # never "gold" — human review required
            if agree:
                survivors.append(candidate)
    return survivors


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Golden-set candidate generation (pre-human-review)")
    p.add_argument(
        "--docs-file",
        required=True,
        help="text file, one repo-relative document path per line",
    )
    p.add_argument("--corpus-root", default=".", help="root the document paths are relative to")
    p.add_argument(
        "--connector",
        required=True,
        help="Model Connector connector name — no silent default (e.g. claude-code, gemini)",
    )
    p.add_argument("--model", required=True, help="model alias passed to the connector")
    p.add_argument("--mc-url", default="https://connector.arcanada.one")
    p.add_argument(
        "--api-key",
        default="",
        help="Model Connector bearer key (prefer env var in real runs — never hardcode)",
    )
    p.add_argument("--n-per-doc", type=int, default=3, help="candidate questions requested per document")
    p.add_argument("--out", required=True, help="output *-candidates.jsonl path (pre-human-review)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    corpus_root = Path(args.corpus_root)
    doc_paths = [line.strip() for line in Path(args.docs_file).read_text(encoding="utf-8").splitlines() if line.strip()]

    docs = []
    for rel_path in doc_paths:
        full_path = corpus_root / rel_path
        if not full_path.exists():
            print(f"WARNING: skipping missing document: {full_path}", file=sys.stderr)
            continue
        docs.append((rel_path, full_path.read_text(encoding="utf-8")))

    client = ModelConnectorClient(args.mc_url, connector=args.connector, model=args.model, api_key=args.api_key)
    survivors = run_batch(client, docs, args.n_per_doc)

    with open(args.out, "w", encoding="utf-8") as f:
        for candidate in survivors:
            f.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    print(
        f"{len(survivors)} candidate(s) survived the two-LLM-agreement filter, written to {args.out}.\n"
        "HUMAN REVIEW REQUIRED before promotion to any golden-arcanada-v*.jsonl file — "
        "see tools/README.md.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
