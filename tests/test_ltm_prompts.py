"""Tests for LTM-0013 reflect prompts."""

from scrutator.ltm.prompts import format_reflect_summary


class TestFormatReflectSummary:
    def test_happy_path(self):
        chunks = [
            {"content": "Scrutator is a search engine."},
            {"content": "Scrutator runs on port 8310."},
        ]
        system, user = format_reflect_summary(chunks, ["Scrutator"])
        assert "knowledge graph reflection assistant" in system
        assert "Scrutator" in user
        assert "[0]" in user and "[1]" in user
        assert "search engine" in user
        assert "port 8310" in user

    def test_max_facts_substituted(self):
        system, _ = format_reflect_summary([{"content": "x"}], ["X"], max_facts=3, max_chars=500)
        assert "Maximum 3 meta-facts" in system
        assert "at most 500 chars" in system

    def test_chunks_truncation(self):
        long_chunk = {"content": "y" * 5000}
        _, user = format_reflect_summary([long_chunk], ["X"])
        # individual chunk content is truncated to 1500 chars
        assert user.count("y") <= 1500 + 50  # +slack for entity name etc.

    def test_empty_entity_names_renders_none(self):
        _, user = format_reflect_summary([{"content": "a"}, {"content": "b"}], [])
        assert "(none)" in user

    def test_chunk_without_content_key_safe(self):
        # Defensive: dict without 'content' shouldn't raise
        _, user = format_reflect_summary([{"other": "value"}, {"content": "real"}], ["E"])
        assert "[0]" in user
        assert "real" in user
