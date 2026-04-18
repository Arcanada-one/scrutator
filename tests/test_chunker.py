"""Tests for the Chunking Engine (SRCH-0003)."""

import os

from scrutator.chunker import Chunk, ChunkMetadata, chunk_document
from scrutator.chunker.metadata import detect_language, extract_frontmatter, extract_tags, extract_wikilinks
from scrutator.chunker.models import ChunkRequest
from scrutator.chunker.splitters import semantic_split, split_by_headers, split_code
from scrutator.chunker.tokenizer import token_count, truncate_to_tokens

# --------------- T1: Chunk model creation ---------------


def test_chunk_creation_with_hash():
    meta = ChunkMetadata(source_path="test.md", source_type="markdown")
    chunk = Chunk(content="Hello world", chunk_index=0, metadata=meta)
    assert chunk.id  # uuid generated
    assert chunk.content_hash  # sha256 generated
    assert chunk.content == "Hello world"
    assert chunk.chunk_index == 0
    assert chunk.parent_id is None


def test_chunk_hash_deterministic():
    meta = ChunkMetadata(source_path="test.md", source_type="markdown")
    c1 = Chunk(content="same text", chunk_index=0, metadata=meta)
    c2 = Chunk(content="same text", chunk_index=1, metadata=meta)
    assert c1.content_hash == c2.content_hash


# --------------- T2: ChunkMetadata validation ---------------


def test_metadata_defaults():
    meta = ChunkMetadata(source_path="wiki/notes.md", source_type="markdown")
    assert meta.heading_hierarchy == []
    assert meta.frontmatter == {}
    assert meta.wikilinks == []
    assert meta.tags == []
    assert meta.language is None


def test_chunk_request_defaults():
    req = ChunkRequest(content="test")
    assert req.max_tokens == 512
    assert req.overlap_tokens == 50
    assert req.embed is False


# --------------- T3: token_count accuracy ---------------


def test_token_count_empty():
    assert token_count("") == 0


def test_token_count_known_text():
    text = " ".join(["word"] * 100)
    count = token_count(text)
    assert 100 <= count <= 200  # 100 * 1.3 = 130, within +-30%


# --------------- T4: truncate_to_tokens ---------------


def test_truncate_to_tokens():
    text = " ".join(["word"] * 1000)
    result = truncate_to_tokens(text, 100)
    result_tokens = token_count(result)
    assert result_tokens <= 120  # some tolerance


def test_truncate_short_text():
    text = "short text"
    assert truncate_to_tokens(text, 100) == text


# --------------- T5: extract_frontmatter ---------------


def test_extract_frontmatter():
    text = "---\ntitle: Test\ntags: [a, b]\n---\n\n# Content\nBody here."
    fm, remaining = extract_frontmatter(text)
    assert fm["title"] == "Test"
    assert fm["tags"] == ["a", "b"]
    assert remaining.startswith("# Content")


def test_extract_frontmatter_no_frontmatter():
    text = "# Just a heading\nNo frontmatter here."
    fm, remaining = extract_frontmatter(text)
    assert fm == {}
    assert remaining == text


# --------------- T6: extract_wikilinks ---------------


def test_extract_wikilinks():
    text = "See [[Project Alpha]] and [[beta/notes]] for details."
    links = extract_wikilinks(text)
    assert "Project Alpha" in links
    assert "beta/notes" in links


def test_extract_wikilinks_none():
    assert extract_wikilinks("No links here.") == []


# --------------- T7: extract_tags ---------------


def test_extract_tags():
    text = "Some text #tag1 and #tag2/sub here.\n#not-a-heading"
    tags = extract_tags(text)
    assert "tag1" in tags
    assert "tag2/sub" in tags


def test_extract_tags_ignores_headings():
    text = "## Heading\n\nText #real-tag here."
    tags = extract_tags(text)
    assert "real-tag" in tags
    assert len(tags) == 1  # heading not extracted


# --------------- T8: split_by_headers ---------------


def test_split_by_headers_three_sections():
    text = "# Title\n\nIntro.\n\n## Section 1\n\nContent 1.\n\n## Section 2\n\nContent 2."
    sections = split_by_headers(text)
    assert len(sections) == 3
    assert sections[0][0] == ["# Title"]
    assert sections[1][0] == ["# Title", "## Section 1"]
    assert "Content 1" in sections[1][1]
    assert sections[2][0] == ["# Title", "## Section 2"]


def test_split_by_headers_nested():
    text = "# Title\n\n## A\n\n### A1\n\nDeep.\n\n## B\n\nFlat."
    sections = split_by_headers(text)
    assert len(sections) == 4
    # A1 should have hierarchy [# Title, ## A, ### A1]
    a1 = [s for s in sections if "Deep" in s[1]][0]
    assert len(a1[0]) == 3


# --------------- T9: semantic_split overlap ---------------


def test_semantic_split_short_text():
    text = "Short paragraph."
    result = semantic_split(text, max_tokens=200, overlap_tokens=50)
    assert len(result) == 1


def test_semantic_split_overlap():
    # Create text with distinct paragraphs
    paragraphs = [f"Paragraph {i} " + "word " * 50 for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = semantic_split(text, max_tokens=200, overlap_tokens=50)
    assert len(chunks) > 1
    # Each chunk should be within token limit (approximately)
    for chunk in chunks:
        assert token_count(chunk) <= 300  # generous tolerance


# --------------- T10: split_code Python ---------------


def test_split_code_python():
    code = '''"""Module docstring."""

import os

def func_a():
    """Function A."""
    return 1

def func_b():
    """Function B."""
    return 2

class MyClass:
    """A class."""
    def method(self):
        return 3
'''
    chunks = split_code(code, max_tokens=512)
    assert len(chunks) >= 3  # preamble + func_a + func_b + MyClass (func_b and MyClass might merge)


# --------------- T11: engine: short doc ---------------


def test_engine_short_doc():
    result = chunk_document("A very short text.", "note.txt")
    assert result.strategy_used == "single"
    assert result.total_chunks == 1
    assert result.chunks[0].metadata.source_type == "text"


# --------------- T12: engine: markdown headers ---------------


def test_engine_markdown_headers():
    md = """---
title: Test Doc
---

# Introduction

Welcome to the doc.

## Section One

First section content with enough text to be meaningful.

## Section Two

Second section with different content.
"""
    result = chunk_document(md, "wiki/test.md")
    assert result.strategy_used == "markdown_headers"
    assert result.total_chunks >= 3
    # Frontmatter should be extracted
    assert result.chunks[0].metadata.frontmatter.get("title") == "Test Doc"


# --------------- T13: engine: giant file ---------------


def test_engine_large_markdown():
    # Generate a large markdown file with many sections
    sections = []
    for i in range(50):
        sections.append(f"## Section {i}\n\n" + " ".join([f"word{j}" for j in range(100)]))
    text = "# Large Document\n\n" + "\n\n".join(sections)
    result = chunk_document(text, "large.md", max_tokens=200)
    assert result.total_chunks >= 20
    assert result.strategy_used == "markdown_headers"


def test_engine_parent_child_hierarchy():
    # Create a section that exceeds 4x max_tokens
    huge_section = "## Giant Section\n\n" + "\n\n".join([" ".join(["word"] * 100) for _ in range(30)])
    text = "# Doc\n\nIntro.\n\n" + huge_section
    result = chunk_document(text, "doc.md", max_tokens=100)
    # Should have at least one parent-child relationship
    parent_ids = {c.parent_id for c in result.chunks if c.parent_id}
    assert len(parent_ids) >= 1


# --------------- T14: engine: Python source ---------------


def test_engine_python_source():
    code = '''"""Config module."""

import os

HOST = os.getenv("HOST", "0.0.0.0")

def get_config():
    return {"host": HOST}

class Config:
    host: str = HOST
'''
    result = chunk_document(code, "src/config.py")
    assert result.strategy_used == "code_boundaries"
    assert result.total_chunks >= 2


# --------------- T15: API: POST /v1/chunk ---------------


def test_api_chunk_endpoint():
    from fastapi.testclient import TestClient

    from scrutator.health import app

    client = TestClient(app)
    response = client.post(
        "/v1/chunk",
        json={
            "content": "# Title\n\n## Section A\n\nContent here.\n\n## Section B\n\nMore content.",
            "source_path": "wiki/test.md",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "chunks" in data
    assert data["total_chunks"] >= 2
    assert data["strategy_used"] == "markdown_headers"


# --------------- Language detection ---------------


def test_detect_language_russian():
    assert detect_language("Это русский текст с кириллическими символами") == "ru"


def test_detect_language_english():
    assert detect_language("This is English text with Latin characters") == "en"


# --------------- Real file tests ---------------

# Base path for the arcanada workspace (tests run from Projects/Scrutator/code/)
_ARCANADA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_WIKI = os.path.join(_ARCANADA, "wiki")
_DATARIM = os.path.join(_ARCANADA, "datarim")
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _skip_if_missing(path: str):
    if not os.path.exists(path):
        import pytest

        pytest.skip(f"file not available: {path}")


# --- Python source files ---


def test_real_python_config():
    path = os.path.join(_SRC, "scrutator", "config.py")
    _skip_if_missing(path)
    result = chunk_document(_read(path), "src/scrutator/config.py")
    assert result.total_chunks >= 1
    assert result.strategy_used == "code_boundaries"


def test_real_python_health():
    path = os.path.join(_SRC, "scrutator", "health.py")
    _skip_if_missing(path)
    result = chunk_document(_read(path), "src/scrutator/health.py")
    assert result.total_chunks >= 1
    assert result.strategy_used == "code_boundaries"


def test_real_python_engine():
    path = os.path.join(_SRC, "scrutator", "chunker", "engine.py")
    _skip_if_missing(path)
    result = chunk_document(_read(path), "src/scrutator/chunker/engine.py")
    assert result.total_chunks >= 2
    assert result.strategy_used == "code_boundaries"


def test_real_python_splitters():
    path = os.path.join(_SRC, "scrutator", "chunker", "splitters.py")
    _skip_if_missing(path)
    result = chunk_document(_read(path), "src/scrutator/chunker/splitters.py")
    assert result.total_chunks >= 2
    assert result.strategy_used == "code_boundaries"


def test_real_python_metadata():
    path = os.path.join(_SRC, "scrutator", "chunker", "metadata.py")
    _skip_if_missing(path)
    result = chunk_document(_read(path), "src/scrutator/chunker/metadata.py")
    assert result.total_chunks >= 2
    assert result.strategy_used == "code_boundaries"


def test_real_python_models():
    path = os.path.join(_SRC, "scrutator", "chunker", "models.py")
    _skip_if_missing(path)
    result = chunk_document(_read(path), "src/scrutator/chunker/models.py")
    assert result.total_chunks >= 1
    assert result.strategy_used == "code_boundaries"


# --- Wiki markdown files ---


def test_real_wiki_schema():
    path = os.path.join(_WIKI, "SCHEMA.md")
    _skip_if_missing(path)
    result = chunk_document(_read(path), "wiki/SCHEMA.md")
    assert result.total_chunks >= 1
    assert result.strategy_used == "markdown_headers"


def test_real_wiki_readme():
    path = os.path.join(_WIKI, "README.md")
    _skip_if_missing(path)
    result = chunk_document(_read(path), "wiki/README.md")
    assert result.total_chunks >= 1


def test_real_wiki_security():
    path = os.path.join(_WIKI, "security", "_raw_.md")
    _skip_if_missing(path)
    result = chunk_document(_read(path), "wiki/security/_raw_.md")
    assert result.total_chunks >= 1


def test_real_wiki_dream_report():
    path = os.path.join(_WIKI, "_reports_", "dream-2026-04-14.md")
    _skip_if_missing(path)
    result = chunk_document(_read(path), "wiki/_reports_/dream-2026-04-14.md")
    assert result.total_chunks >= 1


def test_real_wiki_ltm_comparison():
    path = os.path.join(_WIKI, "AI", "Long Term Memory", "_raw_", "Системы долгосрочной памяти - Сравнение.md")
    _skip_if_missing(path)
    result = chunk_document(_read(path), "wiki/AI/Long Term Memory/comparison.md")
    assert result.total_chunks >= 1


# --- Datarim files ---


def test_real_datarim_tasks():
    """Giant file: datarim/tasks.md (9700+ lines)."""
    path = os.path.join(_DATARIM, "tasks.md")
    _skip_if_missing(path)
    content = _read(path)
    result = chunk_document(content, "datarim/tasks.md", max_tokens=512)
    assert result.total_chunks >= 20
    assert result.strategy_used == "markdown_headers"


def test_real_datarim_backlog():
    path = os.path.join(_DATARIM, "backlog.md")
    _skip_if_missing(path)
    result = chunk_document(_read(path), "datarim/backlog.md")
    assert result.total_chunks >= 1


def test_real_datarim_backlog_archive():
    path = os.path.join(_DATARIM, "backlog-archive.md")
    _skip_if_missing(path)
    result = chunk_document(_read(path), "datarim/backlog-archive.md")
    assert result.total_chunks >= 1


def test_real_datarim_progress():
    path = os.path.join(_DATARIM, "progress.md")
    _skip_if_missing(path)
    result = chunk_document(_read(path), "datarim/progress.md")
    assert result.total_chunks >= 1
    assert result.strategy_used == "markdown_headers"


def test_real_datarim_active_context():
    path = os.path.join(_DATARIM, "activeContext.md")
    _skip_if_missing(path)
    result = chunk_document(_read(path), "datarim/activeContext.md")
    assert result.total_chunks >= 1


def test_real_datarim_system_patterns():
    path = os.path.join(_DATARIM, "systemPatterns.md")
    _skip_if_missing(path)
    result = chunk_document(_read(path), "datarim/systemPatterns.md")
    assert result.total_chunks >= 1


def test_real_datarim_reflection():
    """Pick one reflection file."""
    path = os.path.join(_DATARIM, "reflection", "reflection-SRCH-0001.md")
    _skip_if_missing(path)
    result = chunk_document(_read(path), "datarim/reflection/reflection-SRCH-0001.md")
    assert result.total_chunks >= 1
    assert result.strategy_used == "markdown_headers"


# --- API: content size validation ---


def test_api_rejects_oversized_content():
    """API must reject content > 1MB."""
    from fastapi.testclient import TestClient

    from scrutator.health import app

    client = TestClient(app)
    oversized = "x " * 600_000  # ~1.2MB
    response = client.post(
        "/v1/chunk",
        json={"content": oversized, "source_path": "huge.txt"},
    )
    assert response.status_code == 422
