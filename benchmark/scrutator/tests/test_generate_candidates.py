"""Unit tests for the golden-set candidate-generation tooling (no network — a fake
ModelConnectorClient stands in for the real Model Connector calls)."""

import json

import generate_candidates as gc
import mc_client
import pytest


class _FakeMcClient:
    """Scripted responses per call, in order — no network."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def call(self, prompt, system=None):
        self.calls.append((prompt, system))
        return self._responses.pop(0)

    def call_json(self, prompt, system=None):
        raw = self.call(prompt, system)
        return json.loads(raw) if isinstance(raw, str) else raw


class TestMcClientRequiresExplicitConnector:
    def test_empty_connector_raises(self):
        with pytest.raises(ValueError, match="explicit"):
            mc_client.ModelConnectorClient("http://x", connector="", model="m", api_key="")

    def test_call_json_strips_markdown_fence(self):
        client = mc_client.ModelConnectorClient.__new__(mc_client.ModelConnectorClient)
        # exercise the parsing helper directly without a real call
        client.call = lambda prompt, system=None: '```json\n{"agree": true}\n```'
        assert client.call_json("x") == {"agree": True}


class TestGenerateCandidateQuestions:
    def test_parses_valid_candidates(self):
        response = json.dumps(
            [
                {
                    "query": "What port does X run on?",
                    "class": "factual",
                    "answer": "8310",
                    "gold_source_paths": ["a.md"],
                },
            ]
        )
        client = _FakeMcClient([response])
        candidates = gc.generate_candidate_questions(client, "a.md", "doc text", n=3)
        assert len(candidates) == 1
        assert candidates[0]["class"] == "factual"

    def test_rejects_invalid_class(self):
        response = json.dumps(
            [
                {"query": "q", "class": "not-a-real-class", "answer": "a", "gold_source_paths": ["a.md"]},
            ]
        )
        client = _FakeMcClient([response])
        candidates = gc.generate_candidate_questions(client, "a.md", "doc text", n=3)
        assert candidates == []

    def test_rejects_missing_fields(self):
        response = json.dumps([{"query": "q", "class": "factual"}])  # missing answer/gold_source_paths
        client = _FakeMcClient([response])
        candidates = gc.generate_candidate_questions(client, "a.md", "doc text", n=3)
        assert candidates == []

    def test_non_list_response_raises(self):
        client = _FakeMcClient([json.dumps({"not": "a list"})])
        with pytest.raises(mc_client.ModelConnectorError):
            gc.generate_candidate_questions(client, "a.md", "doc text", n=3)


class TestJudgeAgreement:
    def test_agree_true(self):
        client = _FakeMcClient([json.dumps({"agree": True, "reason": "same fact"})])
        agree, reason = gc.judge_agreement(client, "q", "answer a", "answer b")
        assert agree is True
        assert reason == "same fact"

    def test_agree_false(self):
        client = _FakeMcClient([json.dumps({"agree": False, "reason": "conflicting dates"})])
        agree, _ = gc.judge_agreement(client, "q", "2026-01-01", "2026-02-01")
        assert agree is False

    def test_missing_agree_key_raises(self):
        client = _FakeMcClient([json.dumps({"reason": "oops"})])
        with pytest.raises(mc_client.ModelConnectorError):
            gc.judge_agreement(client, "q", "a", "b")


class TestRunBatch:
    def test_candidate_surviving_judge_is_kept_never_marked_gold(self):
        pass1_response = json.dumps(
            [
                {"query": "q1", "class": "factual", "answer": "gold answer", "gold_source_paths": ["a.md"]},
            ]
        )
        pass2_response = "gold answer restated"
        judge_response = json.dumps({"agree": True, "reason": "matches"})
        client = _FakeMcClient([pass1_response, pass2_response, judge_response])

        survivors = gc.run_batch(client, [("a.md", "doc text")], n_per_doc=3)
        assert len(survivors) == 1
        assert survivors[0]["review_status"] == "candidate"  # never "gold" — human review required
        assert survivors[0]["judge_agree"] is True

    def test_candidate_failing_judge_is_dropped(self):
        pass1_response = json.dumps(
            [
                {"query": "q1", "class": "temporal", "answer": "2026-01-01", "gold_source_paths": ["a.md"]},
            ]
        )
        pass2_response = "2026-06-01"
        judge_response = json.dumps({"agree": False, "reason": "dates differ"})
        client = _FakeMcClient([pass1_response, pass2_response, judge_response])

        survivors = gc.run_batch(client, [("a.md", "doc text")], n_per_doc=3)
        assert survivors == []

    def test_no_candidate_ever_marked_gold_by_this_tool(self):
        """Guards the human-annotation gate: run_batch must never emit review_status='gold'."""
        pass1_response = json.dumps(
            [
                {"query": "q1", "class": "factual", "answer": "a", "gold_source_paths": ["a.md"]},
            ]
        )
        client = _FakeMcClient([pass1_response, "restated", json.dumps({"agree": True, "reason": "ok"})])
        survivors = gc.run_batch(client, [("a.md", "doc text")], n_per_doc=1)
        assert all(c["review_status"] != "gold" for c in survivors)


class TestMainCli:
    def test_main_writes_candidates_jsonl(self, tmp_path, monkeypatch):
        doc = tmp_path / "a.md"
        doc.write_text("Scrutator's embedding server runs on 100.70.137.104.")
        docs_file = tmp_path / "docs.txt"
        docs_file.write_text("a.md\n")
        out_file = tmp_path / "candidates.jsonl"

        pass1_response = json.dumps(
            [
                {"query": "What IP?", "class": "factual", "answer": "100.70.137.104", "gold_source_paths": ["a.md"]},
            ]
        )
        fake_client = _FakeMcClient([pass1_response, "100.70.137.104", json.dumps({"agree": True, "reason": "ok"})])
        monkeypatch.setattr(gc, "ModelConnectorClient", lambda *a, **kw: fake_client)

        rc = gc.main(
            [
                "--docs-file",
                str(docs_file),
                "--corpus-root",
                str(tmp_path),
                "--connector",
                "claude-code",
                "--model",
                "haiku",
                "--out",
                str(out_file),
            ]
        )
        assert rc == 0
        rows = [json.loads(line) for line in out_file.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["review_status"] == "candidate"

    def test_main_skips_missing_documents(self, tmp_path, monkeypatch, capsys):
        docs_file = tmp_path / "docs.txt"
        docs_file.write_text("does-not-exist.md\n")
        out_file = tmp_path / "candidates.jsonl"

        fake_client = _FakeMcClient([])
        monkeypatch.setattr(gc, "ModelConnectorClient", lambda *a, **kw: fake_client)

        rc = gc.main(
            [
                "--docs-file",
                str(docs_file),
                "--corpus-root",
                str(tmp_path),
                "--connector",
                "claude-code",
                "--model",
                "haiku",
                "--out",
                str(out_file),
            ]
        )
        assert rc == 0
        assert out_file.read_text() == ""
