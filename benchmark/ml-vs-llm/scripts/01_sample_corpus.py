#!/usr/bin/env python3
"""Extract 100 stratified chunks from Scrutator via search API.

Uses Scrutator search API with diverse queries to collect chunks with full content,
then applies stratified sampling by source_path prefix.

Usage:
    python scripts/01_sample_corpus.py [--scrutator-url URL]
"""

import argparse
import json
import os
import random
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import GOLDEN_DIR, SAMPLE_SIZE

SCRUTATOR_URL = os.environ.get("SCRUTATOR_URL", "http://100.70.137.104:8310")
NAMESPACE = "arcanada"

# Diverse seed queries covering all areas of the knowledge base
SEED_QUERIES = [
    # AI & ML
    "AI agent architecture design",
    "machine learning model training",
    "neural network embedding",
    "LLM prompt engineering",
    "Claude Code Cursor Gemini",
    # Projects
    "Scrutator search retrieval engine",
    "Verdicus macOS application Swift",
    "Arganize Telegram assistant",
    "Model Connector API endpoint",
    "Transcribator transcription audio",
    "Long Term Memory knowledge graph",
    "Agent Dreamer wiki organizer",
    "Disk Arcana file sync",
    # Infrastructure
    "PostgreSQL database pgvector",
    "Docker container deployment server",
    "nginx reverse proxy SSL certificate",
    "Tailscale mesh VPN network",
    "GitHub Actions CI CD workflow",
    "Cloudflare DNS domain",
    "HashiCorp Vault secrets",
    # Security
    "authentication OAuth token",
    "encryption security vulnerability",
    "firewall access control",
    # Business & Philosophy
    "business strategy startup",
    "philosophy concept meaning",
    "Obsidian knowledge base notes",
    "social media Telegram LinkedIn",
    # Tools & Development
    "TypeScript NestJS backend API",
    "Python FastAPI development",
    "Rust programming language",
    "Swift SwiftUI iOS macOS",
    "React frontend component",
    # Misc to get edge coverage
    "настройка конфигурация",
    "экосистема проект план",
    "benchmark test evaluation",
    "documentation readme guide",
    "error debugging fix",
    "migration update deploy",
]


def fetch_chunks_via_search(base_url: str, query: str, top_n: int = 100) -> list[dict]:
    """Fetch chunks from Scrutator search API (returns full content)."""
    try:
        resp = requests.post(
            f"{base_url}/v1/search",
            json={"query": query, "namespace": NAMESPACE, "top_n": top_n},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as e:
        print(f"  WARNING: Query '{query[:30]}...' failed: {e}")
        return []


def collect_all_chunks(base_url: str) -> dict[str, list[dict]]:
    """Collect unique chunks via diverse search queries, grouped by source prefix."""
    seen_ids: set[str] = set()
    groups: dict[str, list[dict]] = {}

    for i, query in enumerate(SEED_QUERIES):
        results = fetch_chunks_via_search(base_url, query)
        new_count = 0

        for r in results:
            cid = r.get("chunk_id", "")
            if cid in seen_ids:
                continue

            content = r.get("content", "")
            if not content or len(content) < 50:
                continue

            seen_ids.add(cid)
            new_count += 1
            sp = r.get("source_path", "unknown")
            parts = sp.strip("/").split("/")
            prefix = "/".join(parts[:2]) if len(parts) >= 2 else parts[0] if parts else "unknown"

            groups.setdefault(prefix, []).append({
                "chunk_id": cid,
                "content": content,
                "source_path": sp,
                "source_type": r.get("source_type", ""),
                "namespace": r.get("namespace", NAMESPACE),
                "token_count": r.get("token_count", 0),
                "chunk_index": r.get("chunk_index", 0),
                "metadata": r.get("metadata", {}),
            })

        print(f"  [{i + 1}/{len(SEED_QUERIES)}] '{query[:35]}' → {len(results)} results, {new_count} new (total: {len(seen_ids)})")

    return groups


def stratified_sample(groups: dict[str, list[dict]], n: int) -> list[dict]:
    """Stratified sample: proportional allocation across source groups."""
    total = sum(len(v) for v in groups.values())
    if total <= n:
        return [chunk for chunks in groups.values() for chunk in chunks]

    sampled = []
    remaining = n

    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

    for i, (prefix, chunks) in enumerate(sorted_groups):
        groups_left = len(sorted_groups) - i
        share = max(1, round(len(chunks) / total * n))
        share = min(share, remaining - max(0, groups_left - 1))
        share = min(share, len(chunks))
        share = max(0, share)

        if share > 0:
            selected = random.sample(chunks, share)
            sampled.extend(selected)
            remaining -= share

        if remaining <= 0:
            break

    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scrutator-url", default=SCRUTATOR_URL)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()

    random.seed(42)

    print(f"Scrutator API: {args.scrutator_url}")
    print(f"Collecting chunks via {len(SEED_QUERIES)} search queries...")
    groups = collect_all_chunks(args.scrutator_url)
    total_chunks = sum(len(v) for v in groups.values())
    print(f"\nCollected {total_chunks} unique chunks in {len(groups)} source groups:")
    for prefix, chunks in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"  {prefix}: {len(chunks)} chunks")

    sample = stratified_sample(groups, args.sample_size)
    print(f"\nSampled {len(sample)} chunks (target: {args.sample_size})")

    os.makedirs(GOLDEN_DIR, exist_ok=True)
    output_path = os.path.join(GOLDEN_DIR, "corpus_100.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_path}")

    source_dist = {}
    for c in sample:
        parts = c["source_path"].strip("/").split("/")
        prefix = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
        source_dist[prefix] = source_dist.get(prefix, 0) + 1

    print(f"\nSource distribution:")
    for prefix, count in sorted(source_dist.items(), key=lambda x: -x[1]):
        print(f"  {prefix}: {count}")

    avg_len = sum(len(c["content"]) for c in sample) / len(sample) if sample else 0
    print(f"Avg content length: {avg_len:.0f} chars")


if __name__ == "__main__":
    main()
