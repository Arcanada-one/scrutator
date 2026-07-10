#!/usr/bin/env python3
"""Create golden dataset by auto-annotating chunks via Claude Opus (OpenRouter).

Annotates 70 chunks automatically, leaving 30 for manual annotation.
Generates: golden_ner.json, golden_re.json, golden_class.json

Usage:
    python scripts/02_create_golden.py [--manual-first 30]
"""

import argparse
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    AREA_LABELS,
    ENTITY_TYPES,
    GOLDEN_DIR,
    MANUAL_ANNOTATION_SIZE,
    OPENROUTER_API_KEY,
    OPENROUTER_URL,
)

OPUS_MODEL = "anthropic/claude-sonnet-4"  # Cost-effective for annotation


def call_opus(prompt: str, system: str, max_tokens: int = 2000) -> dict:
    """Call Claude via OpenRouter for golden annotation."""
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPUS_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=120,
    )
    if resp.status_code != 200:
        print(f"  API error {resp.status_code}: {resp.text[:200]}")
        return {"text": "[]", "input_tokens": 0, "output_tokens": 0}

    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return {
        "text": text,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }


def parse_json_response(text: str):
    """Parse JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for sc, ec in [("[", "]"), ("{", "}")]:
            si = text.find(sc)
            ei = text.rfind(ec)
            if si != -1 and ei > si:
                try:
                    return json.loads(text[si : ei + 1])
                except json.JSONDecodeError:
                    continue
    return None


def annotate_ner(content: str) -> list[dict] | None:
    """Extract entities via Opus for golden dataset."""
    system = (
        "You are a meticulous knowledge graph entity annotator.\n"
        "Extract ALL named entities from the text. Be thorough.\n"
        "Return ONLY a JSON array.\n"
        f'Each entity: {{"name": "exact text", "type": "one of: {", ".join(ENTITY_TYPES)}"}}\n'
        "Include every entity you can find, even minor ones."
    )
    result = call_opus(f"Extract all entities:\n\n{content[:4000]}", system)
    parsed = parse_json_response(result["text"])
    if isinstance(parsed, list):
        return parsed, result
    return None, result


def annotate_re(content: str, entities: list[dict]) -> list[dict] | None:
    """Extract relations via Opus for golden dataset."""
    entity_list = "\n".join(f"- {e['name']} ({e['type']})" for e in entities)
    system = (
        "You are a meticulous knowledge graph relationship annotator.\n"
        "Extract ALL relationships between the given entities.\n"
        "Return ONLY a JSON array.\n"
        '{"source": "entity_name", "relation": "verb_phrase", "target": "entity_name"}\n'
        "Relations: works_on, depends_on, part_of, created_by, located_in, related_to, "
        "uses, implements, extends, etc."
    )
    prompt = f"Entities:\n{entity_list}\n\nSource text:\n{content[:4000]}"
    result = call_opus(prompt, system)
    parsed = parse_json_response(result["text"])
    if isinstance(parsed, list):
        return parsed, result
    return None, result


def annotate_classification(content: str) -> str | None:
    """Classify chunk into one of the areas."""
    system = (
        "You are a text classifier. Classify the text into exactly one area.\n"
        f"Valid areas: {', '.join(AREA_LABELS)}\n"
        "Return ONLY the area label (one word), nothing else."
    )
    result = call_opus(f"Classify:\n\n{content[:2000]}", system, max_tokens=20)
    area = result["text"].strip().lower()
    # Normalize
    for label in AREA_LABELS:
        if label.lower() == area:
            return label, result
    return area, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-first", type=int, default=MANUAL_ANNOTATION_SIZE,
                        help="Number of chunks reserved for manual annotation")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without calling API")
    args = parser.parse_args()

    corpus_path = os.path.join(GOLDEN_DIR, "corpus_100.json")
    if not os.path.exists(corpus_path):
        print(f"ERROR: {corpus_path} not found. Run 01_sample_corpus.py first.")
        sys.exit(1)

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    manual_chunks = corpus[: args.manual_first]
    auto_chunks = corpus[args.manual_first :]
    print(f"Corpus: {len(corpus)} chunks")
    print(f"  Manual annotation (skipped): {len(manual_chunks)}")
    print(f"  Auto annotation (Opus): {len(auto_chunks)}")

    if args.dry_run:
        print("Dry run — exiting.")
        return

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    # Annotate auto chunks
    golden_ner = []
    golden_re = []
    golden_class = []
    total_input = 0
    total_output = 0
    errors = 0

    for i, chunk in enumerate(auto_chunks):
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]
        print(f"\n[{i + 1}/{len(auto_chunks)}] Annotating {chunk_id[:12]}...")

        # NER
        entities, ner_result = annotate_ner(content)
        if entities:
            golden_ner.append({"chunk_id": chunk_id, "entities": entities})
            total_input += ner_result["input_tokens"]
            total_output += ner_result["output_tokens"]
        else:
            print(f"  WARNING: NER failed for {chunk_id[:12]}")
            golden_ner.append({"chunk_id": chunk_id, "entities": []})
            errors += 1

        # RE (only if we got entities)
        if entities and len(entities) >= 2:
            relations, re_result = annotate_re(content, entities)
            if relations:
                golden_re.append({"chunk_id": chunk_id, "triples": relations})
                total_input += re_result["input_tokens"]
                total_output += re_result["output_tokens"]
            else:
                golden_re.append({"chunk_id": chunk_id, "triples": []})
                errors += 1
        else:
            golden_re.append({"chunk_id": chunk_id, "triples": []})

        # Classification
        area, class_result = annotate_classification(content)
        golden_class.append({"chunk_id": chunk_id, "area": area})
        total_input += class_result["input_tokens"]
        total_output += class_result["output_tokens"]

        # Rate limit: ~2 req/s
        time.sleep(0.5)

    # Save auto-annotated golden
    for name, data in [
        ("golden_ner.json", golden_ner),
        ("golden_re.json", golden_re),
        ("golden_class.json", golden_class),
    ]:
        path = os.path.join(GOLDEN_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {path} ({len(data)} entries)")

    # Save empty templates for manual annotation
    manual_ner = [{"chunk_id": c["chunk_id"], "entities": []} for c in manual_chunks]
    manual_re = [{"chunk_id": c["chunk_id"], "triples": []} for c in manual_chunks]
    manual_class = [{"chunk_id": c["chunk_id"], "area": ""} for c in manual_chunks]

    for name, data in [
        ("golden_manual_30_ner.json", manual_ner),
        ("golden_manual_30_re.json", manual_re),
        ("golden_manual_30_class.json", manual_class),
    ]:
        path = os.path.join(GOLDEN_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {path} (manual template, {len(data)} entries)")

    # Cost report
    # Sonnet pricing: $3/M input, $15/M output
    cost = (total_input * 3.0 + total_output * 15.0) / 1_000_000
    print(f"\n--- Cost Report ---")
    print(f"Total input tokens: {total_input:,}")
    print(f"Total output tokens: {total_output:,}")
    print(f"Estimated cost: ${cost:.4f}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
