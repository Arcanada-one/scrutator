"""Bounded, non-executing Go source analysis for RelationshipGraph/v1.

Parses package clauses, import paths, build tags and os.Getenv/LookupEnv keys with an
AST-lite comment/string walk. Does not invoke the Go toolchain, does not fabricate a
module identity when go.mod is absent, and does not invent nodes for external imports.
A .go file without a package clause yields no code_unit (no fake bare node).
"""
from __future__ import annotations

import os
import re

GO_EXT = (".go",)
GO_PACKAGE_RE = re.compile(r"(?m)^package\s+([A-Za-z_][A-Za-z0-9_]*)\s*$")
GO_IMPORT_SINGLE = re.compile(r'(?m)^import\s+(?:([A-Za-z_][\w.]*)\s+)?"([^"]+)"\s*$')
GO_IMPORT_BLOCK = re.compile(r"(?ms)^import\s*\((.*?)\)")
GO_IMPORT_LINE = re.compile(r'(?m)^\s*(?:(?:\w+|\.)\s+)?"([^"]+)"\s*$')
GO_BUILD_TAG_RE = re.compile(r"(?m)^//(?:go:build|\s*\+build)\s+(.+)$")
GO_ENV_RE = re.compile(r"""os\.(?:Getenv|LookupEnv)\(\s*"([A-Z_][A-Z0-9_]*)"\s*\)""")
GO_MODULE_RE = re.compile(r"(?m)^module\s+(\S+)\s*$")
GO_SERVICE_IMPORT_HINTS = ("net/http", "google.golang.org/grpc", "github.com/gin-gonic/gin",
                           "github.com/labstack/echo", "github.com/gofiber/fiber")


def strip_comments_go(src: str) -> str:
    """Blank Go comments while preserving string/`raw string` contents (AST-lite).

    Rune literals (`'/'`) are not string-masked — the same accepted approximation as
    strip_comments_rust. Raw strings use backticks and may span lines.
    """
    out: list[str] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '"':
            out.append(c)
            i += 1
            while i < n and src[i] != '"':
                if src[i] == "\\" and i + 1 < n:
                    out.append(src[i:i + 2])
                    i += 2
                    continue
                if src[i] == "\n":
                    break
                out.append(src[i])
                i += 1
            if i < n:
                out.append(src[i])
                i += 1
            continue
        if c == "`":
            out.append(c)
            i += 1
            while i < n and src[i] != "`":
                out.append(src[i])
                i += 1
            if i < n:
                out.append(src[i])
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", src[i:j]))
            i = j
            continue
        out.append(c)
        i += 1
    return "".join(out)


def build_tags_of(src: str) -> list[str]:
    """Collect //go:build and // +build lines from the raw file prefix (before package)."""
    tags: list[str] = []
    for line in src.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("//"):
            m = GO_BUILD_TAG_RE.match(line)
            if m:
                tags.append(m.group(1).strip())
            continue
        break
    return tags


def parse_go_file(src: str) -> dict | None:
    """Return package/imports/build_tags/env_keys, or None when no package clause exists."""
    tags = build_tags_of(src)
    code = strip_comments_go(src)
    m = GO_PACKAGE_RE.search(code)
    if not m:
        return None
    package = m.group(1)
    imports: list[str] = []
    for im in GO_IMPORT_SINGLE.finditer(code):
        imports.append(im.group(2))
    for block in GO_IMPORT_BLOCK.finditer(code):
        for line in GO_IMPORT_LINE.finditer(block.group(1)):
            imports.append(line.group(1))
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    ordered: list[str] = []
    for imp in imports:
        if imp not in seen:
            seen.add(imp)
            ordered.append(imp)
    env_keys = sorted({k for k in GO_ENV_RE.findall(code)})
    return {"package": package, "imports": ordered, "build_tags": tags, "env_keys": env_keys}


def is_go_test(path: str) -> bool:
    return path.endswith("_test.go")


def module_path(go_mod_text: str) -> str | None:
    m = GO_MODULE_RE.search(go_mod_text)
    return m.group(1) if m else None


def norm_path(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")


def import_package_dir(modules: list[tuple[str, str]], import_path: str) -> str | None:
    """Map an import path to a package directory under a declared module; never invent a module."""
    best: tuple[str, str] | None = None
    for mod_dir, mod_path in modules:
        if import_path == mod_path or import_path.startswith(mod_path + "/"):
            if best is None or len(mod_path) > len(best[1]):
                best = (mod_dir, mod_path)
    if best is None:
        return None
    mod_dir, mod_path = best
    rel = "" if import_path == mod_path else import_path[len(mod_path) + 1:]
    prefix = (mod_dir + "/") if mod_dir not in ("", ".") else ""
    return norm_path(prefix + rel) if rel else (mod_dir or ".")


def package_dir_files(tree_paths: list[str], directory: str) -> list[str]:
    want = "" if directory in ("", ".") else directory.rstrip("/")
    out = []
    for p in tree_paths:
        if not p.endswith(".go"):
            continue
        parent = os.path.dirname(p)
        if parent == want:
            out.append(p)
    return sorted(out)


def owning_module(path: str, modules: list[tuple[str, str]]) -> tuple[str, str] | None:
    """Nearest go.mod directory that contains path (longest prefix wins)."""
    best = None
    for mod_dir, mod_path in modules:
        if mod_dir in ("", "."):
            cand = True
        else:
            cand = path == mod_dir or path.startswith(mod_dir.rstrip("/") + "/")
        if not cand:
            continue
        if best is None or len(mod_dir) > len(best[0]):
            best = (mod_dir, mod_path)
    return best
