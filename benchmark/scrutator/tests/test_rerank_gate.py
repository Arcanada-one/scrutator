"""SRCH-0031 paired /v1/search rerank-gate tests."""

import pytest
import rerank_gate


def _result(path: str, *, chunk_id: str, score: float, score_kind: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "source_path": path,
        "score": score,
        "citation": {
            "chunk_id": chunk_id,
            "source_path": path,
            "source_type": "md",
            "chunk_index": 0,
            "heading_hierarchy": [],
            "relevance_score": score,
            "score_kind": score_kind,
        },
    }


def _response(score_kind: str, paths: list[str] | None = None) -> dict:
    paths = paths or [f"docs/{i}.md" for i in range(5)]
    return {
        "results": [
            _result(path, chunk_id=f"chunk-{index}", score=5.0 - index, score_kind=score_kind)
            for index, path in enumerate(paths)
        ],
        "total": 5,
        "query": "q",
        "search_time_ms": 12.5,
    }


def test_gold_match_accepts_exact_or_repo_root_prefix_only():
    assert rerank_gate.gold_hit(["docs/a.md"], ["docs/a.md"])
    assert rerank_gate.gold_hit(["docs/a.md"], ["/home/dev/arcanada/docs/a.md"])
    assert not rerank_gate.gold_hit(["docs/a.md"], ["elsewhere/a.md"])
    assert not rerank_gate.gold_hit(["a.md"], ["docs/not-a.md"])


def test_validate_response_requires_exact_mode_and_unique_five_results():
    validated = rerank_gate.validate_response(_response("rrf"), expected_score_kind="rrf")
    assert len(validated) == 5

    with pytest.raises(rerank_gate.InvalidEvidence, match="score_kind"):
        rerank_gate.validate_response(_response("rrf"), expected_score_kind="colbert_rerank")

    short = _response("rrf")
    short["results"].pop()
    with pytest.raises(rerank_gate.InvalidEvidence, match="exactly 5"):
        rerank_gate.validate_response(short, expected_score_kind="rrf")

    duplicate = _response("rrf")
    duplicate["results"][4]["chunk_id"] = duplicate["results"][0]["chunk_id"]
    duplicate["results"][4]["citation"]["chunk_id"] = duplicate["results"][0]["chunk_id"]
    with pytest.raises(rerank_gate.InvalidEvidence, match="duplicate"):
        rerank_gate.validate_response(duplicate, expected_score_kind="rrf")


def test_validate_response_rejects_observed_score_tie():
    tied = _response("colbert_rerank")
    tied["results"][4]["score"] = tied["results"][3]["score"]
    tied["results"][4]["citation"]["relevance_score"] = tied["results"][3]["score"]
    with pytest.raises(rerank_gate.InvalidEvidence, match="tie"):
        rerank_gate.validate_response(tied, expected_score_kind="colbert_rerank")


def test_paired_transition_counts_gain_loss_and_all_gold():
    rows = [
        {
            "id": "F1",
            "class": "factual",
            "gold_source_paths": ["docs/a.md"],
            "off_paths": ["docs/x.md"],
            "on_paths": ["docs/a.md"],
        },
        {
            "id": "F2",
            "class": "factual",
            "gold_source_paths": ["docs/b.md"],
            "off_paths": ["docs/b.md"],
            "on_paths": ["docs/y.md"],
        },
        {
            "id": "M1",
            "class": "multi-hop",
            "gold_source_paths": ["docs/c.md", "docs/d.md"],
            "off_paths": ["docs/c.md", "docs/d.md"],
            "on_paths": ["docs/c.md"],
        },
    ]

    transitions = rerank_gate.summarize_transitions(rows)

    assert transitions["factual"] == {
        "n": 2,
        "off_hits": 1,
        "on_hits": 1,
        "gains": 1,
        "losses": 1,
        "off_all_gold": 1,
        "on_all_gold": 1,
    }
    assert transitions["multi-hop"]["off_hits"] == 1
    assert transitions["multi-hop"]["on_hits"] == 1
    assert transitions["multi-hop"]["off_all_gold"] == 1
    assert transitions["multi-hop"]["on_all_gold"] == 0


def test_verdict_is_eligible_only_for_zero_losses_real_gain_floors_and_latency():
    eligible = {
        "factual": {
            "n": 15,
            "off_hits": 14,
            "on_hits": 15,
            "gains": 1,
            "losses": 0,
            "off_all_gold": 14,
            "on_all_gold": 15,
        },
        "multi-hop": {
            "n": 8,
            "off_hits": 8,
            "on_hits": 8,
            "gains": 0,
            "losses": 0,
            "off_all_gold": 3,
            "on_all_gold": 3,
        },
        "temporal": {
            "n": 10,
            "off_hits": 8,
            "on_hits": 8,
            "gains": 0,
            "losses": 0,
            "off_all_gold": 8,
            "on_all_gold": 8,
        },
    }

    verdict = rerank_gate.decide_verdict(eligible, on_latency_p95_ms=4999.0)
    assert verdict == {"status": "ELIGIBLE_TO_FLIP", "reasons": []}

    loss = {name: dict(values) for name, values in eligible.items()}
    loss["temporal"].update(on_hits=8, gains=1, losses=1)
    assert rerank_gate.decide_verdict(loss, on_latency_p95_ms=1000.0)["status"] == "KEEP_OFF"

    no_gain = {name: dict(values) for name, values in eligible.items()}
    no_gain["factual"].update(on_hits=14, gains=0)
    assert "no paired gain" in " ".join(rerank_gate.decide_verdict(no_gain, on_latency_p95_ms=1000.0)["reasons"])

    assert rerank_gate.decide_verdict(eligible, on_latency_p95_ms=5000.1)["status"] == "KEEP_OFF"


def test_candidate_scope_never_claims_production_flip_eligibility():
    eligible = {
        "factual": {
            "n": 15,
            "off_hits": 14,
            "on_hits": 15,
            "gains": 1,
            "losses": 0,
            "off_all_gold": 14,
            "on_all_gold": 15,
        },
        "multi-hop": {
            "n": 8,
            "off_hits": 8,
            "on_hits": 8,
            "gains": 0,
            "losses": 0,
            "off_all_gold": 3,
            "on_all_gold": 3,
        },
        "temporal": {
            "n": 10,
            "off_hits": 8,
            "on_hits": 8,
            "gains": 0,
            "losses": 0,
            "off_all_gold": 8,
            "on_all_gold": 8,
        },
    }

    verdict = rerank_gate.decide_verdict(eligible, on_latency_p95_ms=1000.0, eligibility_scope="candidate")

    assert verdict == {"status": "CANDIDATE_ELIGIBLE", "reasons": []}


def test_repeated_rows_must_preserve_order_and_hit_bits():
    stable = [
        [{"id": "F1", "off_paths": ["a"], "on_paths": ["b"], "off_hit": False, "on_hit": True}],
        [{"id": "F1", "off_paths": ["a"], "on_paths": ["b"], "off_hit": False, "on_hit": True}],
    ]
    rerank_gate.require_repeat_stability(stable)

    unstable = [stable[0], [{"id": "F1", "off_paths": ["x"], "on_paths": ["b"], "off_hit": False, "on_hit": True}]]
    with pytest.raises(rerank_gate.InvalidEvidence, match="unstable"):
        rerank_gate.require_repeat_stability(unstable)


def test_unexpected_runner_exception_is_invalid_evidence(monkeypatch, capsys, tmp_path):
    def crash(_args):
        raise TypeError("programming defect")

    monkeypatch.setattr(rerank_gate, "run_experiment", crash)

    rc = rerank_gate.main(
        [
            "--off-endpoint",
            "http://127.0.0.1:18310",
            "--on-endpoint",
            "http://127.0.0.1:18311",
            "--golden",
            str(tmp_path / "golden.jsonl"),
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert rc == rerank_gate.INVALID_EVIDENCE_CODE
    assert "UNEXPECTED BENCHMARK ERROR: TypeError" in capsys.readouterr().err
