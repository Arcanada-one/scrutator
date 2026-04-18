"""Metadata extraction from document content: frontmatter, WikiLinks, tags, language detection."""

from __future__ import annotations

import re

import yaml

_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_TAG_RE = re.compile(r"(?:^|\s)#([a-zA-Z][a-zA-Z0-9_/-]*)", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Cyrillic character ranges for language detection
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_LATIN_RE = re.compile(r"[a-zA-Z]")


def extract_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter from text. Returns (frontmatter_dict, remaining_text)."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        fm = yaml.safe_load(match.group(1))
        if not isinstance(fm, dict):
            return {}, text
        remaining = text[match.end() :]
        return fm, remaining
    except yaml.YAMLError:
        return {}, text


def extract_wikilinks(text: str) -> list[str]:
    """Extract Obsidian-style [[WikiLinks]] from text."""
    return _WIKILINK_RE.findall(text)


def extract_tags(text: str) -> list[str]:
    """Extract #tags from text (Obsidian-style). Excludes headings."""
    # Filter out markdown headings (lines starting with #)
    lines = text.split("\n")
    filtered = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("# ") or stripped.startswith("## ") or stripped.startswith("### "):
            continue
        if stripped.startswith("#### ") or stripped.startswith("##### ") or stripped.startswith("###### "):
            continue
        filtered.append(line)
    return _TAG_RE.findall("\n".join(filtered))


def detect_language(text: str) -> str | None:
    """Detect language based on character frequency. Returns 'ru', 'en', or None."""
    if not text:
        return None
    cyrillic_count = len(_CYRILLIC_RE.findall(text))
    latin_count = len(_LATIN_RE.findall(text))
    total = cyrillic_count + latin_count
    if total < 10:
        return None
    if cyrillic_count / total > 0.3:
        return "ru"
    if latin_count / total > 0.3:
        return "en"
    return None
