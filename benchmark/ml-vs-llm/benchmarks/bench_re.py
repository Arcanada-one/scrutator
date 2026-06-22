"""Relation Extraction Benchmark: Phi-4-mini (Ollama) vs Claude Haiku vs GPT-4o-mini.

Runs all models on golden RE dataset (70 chunks), measures:
- F1 exact match (source, relation, target)
- F1 partial match (source, target only)
- Latency, cost

Usage:
    python benchmarks/bench_re.py [--chunks N] [--models phi4,haiku,gpt4omini]
"""

import argparse
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    GOLDEN_DIR,
    GPT4OMINI_MODEL,
    HAIKU_MODEL,
    PHI4_OLLAMA_MODEL,
    PRICING,
    RESULTS_DIR,
)
from utils.llm_client import OpenRouterClient
from utils.metrics import compute_cost, compute_re_f1


def load_golden_re() -> dict[str, list[dict]]:
    """Load golden RE: {chunk_id: [triples]}."""
    with open(os.path.join(GOLDEN_DIR, "golden_re.json")) as f:
        data = json.load(f)
    return {item["chunk_id"]: item["triples"] for item in data}


def load_corpus() -> dict[str, dict]:
    with open(os.path.join(GOLDEN_DIR, "corpus_100.json")) as f:
        data = json.load(f)
    return {item["chunk_id"]: item for item in data}


def load_golden_ner() -> dict[str, list[dict]]:
    """Load NER golden to provide entities context for RE."""
    with open(os.path.join(GOLDEN_DIR, "golden_ner.json")) as f:
        data = json.load(f)
    return {item["chunk_id"]: item["entities"] for item in data}


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

RE_PROMPT = (
    "You are a knowledge graph relationship extractor.\n"
    "Given a text and a list of entities found in it, extract relationships.\n"
    "Return ONLY a JSON array. Each element:\n"
    '{"source": "entity_name", "relation": "verb_phrase", "target": "entity_name"}\n'
    "Use short relation names: works_on, depends_on, part_of, created_by, "
    "located_in, related_to, uses, generates, etc.\n"
    "Maximum 15 relations per text."
)


def call_phi4(text: str, entities: list[dict]) -> dict:
    """Call Phi-4-mini via Ollama for relation extraction."""
    entity_list = ", ".join(e.get("name", "") for e in entities)
    prompt = (
        f"{RE_PROMPT}\n\n"
        f"Entities: {entity_list}\n\n"
        f"Text:\n{text[:3000]}"
    )

    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": PHI4_OLLAMA_MODEL,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 2000},
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_text = data.get("response", "")
        latency_ms = (time.perf_counter() - start) * 1000

        # Parse JSON — Phi-4 may return {"relations": [...]} or [...]
        # Also normalizes subject/verb/object → source/relation/target
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                # Could be {"relations": [...]} or a single relation object
                if "relations" in parsed or "triples" in parsed:
                    relations = parsed.get("relations", parsed.get("triples", []))
                elif "source" in parsed or "subject" in parsed:
                    # Single relation object — wrap in list
                    relations = [parsed]
                else:
                    relations = []
            elif isinstance(parsed, list):
                relations = parsed
            else:
                relations = []
            # Normalize Phi-4's subject/verb/object format
            normalized = []
            for r in relations:
                if not isinstance(r, dict):
                    continue
                src = r.get("source", r.get("subject", ""))
                rel = r.get("relation", r.get("verb", ""))
                tgt = r.get("target", r.get("object", ""))
                # Skip malformed entries (e.g. list instead of string)
                if not isinstance(src, str) or not isinstance(tgt, str):
                    continue
                if not isinstance(rel, str):
                    rel = str(rel) if rel else ""
                normalized.append({"source": src, "relation": rel, "target": tgt})
            relations = normalized
        except json.JSONDecodeError:
            relations = []

        return {
            "relations": relations,
            "latency_ms": round(latency_ms, 1),
            "input_tokens": 0,
            "output_tokens": 0,
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "relations": [],
            "latency_ms": round(latency_ms, 1),
            "input_tokens": 0,
            "output_tokens": 0,
            "error": str(e),
        }


def run_phi4(chunks: list[dict], golden_ner: dict[str, list[dict]]) -> list[dict]:
    """Run Phi-4-mini on all chunks."""
    results = []
    for chunk in chunks:
        cid = chunk["chunk_id"]
        entities = golden_ner.get(cid, [])
        resp = call_phi4(chunk["content"], entities)
        results.append({"chunk_id": cid, **resp})
    return results


def run_llm_re(
    client: OpenRouterClient,
    chunks: list[dict],
    golden_ner: dict[str, list[dict]],
    model: str,
    sleep_between: float = 0.3,
) -> list[dict]:
    """Run LLM RE on chunks."""
    results = []
    for chunk in chunks:
        cid = chunk["chunk_id"]
        entities = golden_ner.get(cid, [])
        try:
            resp = client.extract_relations(chunk["content"], entities, model)
            results.append({
                "chunk_id": cid,
                "relations": resp["relations"],
                "latency_ms": resp["latency_ms"],
                "input_tokens": resp["input_tokens"],
                "output_tokens": resp["output_tokens"],
            })
        except Exception as e:
            print(f"  ERROR {cid[:8]}: {e}")
            results.append({
                "chunk_id": cid,
                "relations": [],
                "latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e),
            })
        if sleep_between > 0:
            time.sleep(sleep_between)
    return results


def evaluate(predictions: list[dict], golden: dict[str, list[dict]], model_name: str) -> dict:
    """Evaluate RE predictions."""
    all_pred_exact = []
    all_gold_exact = []
    latencies = []
    total_in = 0
    total_out = 0
    errors = 0

    for pred in predictions:
        cid = pred["chunk_id"]
        if cid not in golden:
            continue
        all_pred_exact.extend(pred["relations"])
        all_gold_exact.extend(golden[cid])
        latencies.append(pred["latency_ms"])
        total_in += pred["input_tokens"]
        total_out += pred["output_tokens"]
        if "error" in pred:
            errors += 1

    exact = compute_re_f1(all_pred_exact, all_gold_exact, partial=False)
    partial = compute_re_f1(all_pred_exact, all_gold_exact, partial=True)
    cost = compute_cost(total_in, total_out, model_name, PRICING)

    return {
        "model": model_name,
        "exact_match": exact,
        "partial_match": partial,
        "chunks_processed": len(predictions) - errors,
        "chunks_errors": errors,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "cost_usd": cost,
        "latency_mean_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "latency_p50_ms": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0,
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if latencies else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="RE Benchmark")
    parser.add_argument("--chunks", type=int, default=0, help="Limit chunks (0=all)")
    parser.add_argument("--models", default="phi4,haiku,gpt4omini", help="Comma-separated")
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",")]

    golden = load_golden_re()
    golden_ner = load_golden_ner()
    corpus = load_corpus()
    golden_ids = list(golden.keys())
    chunks = [corpus[cid] for cid in golden_ids if cid in corpus]
    if args.chunks > 0:
        chunks = chunks[: args.chunks]

    print(f"RE Benchmark: {len(chunks)} chunks, models: {model_list}")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    # --- Phi-4-mini ---
    if "phi4" in model_list:
        print(f"\n[1/3] Phi-4-mini ({PHI4_OLLAMA_MODEL}) via Ollama...")
        preds = run_phi4(chunks, golden_ner)
        metrics = evaluate(preds, golden, PHI4_OLLAMA_MODEL)
        all_results["phi4"] = {"predictions": preds, "metrics": metrics}
        print(f"  Exact F1={metrics['exact_match']['f1']:.3f}  Partial F1={metrics['partial_match']['f1']:.3f}")
        print(f"  Latency p50={metrics['latency_p50_ms']:.0f}ms  Cost=${metrics['cost_usd']:.4f}")

    # --- Haiku ---
    if "haiku" in model_list:
        print(f"\n[2/3] Claude Haiku ({HAIKU_MODEL})...")
        client = OpenRouterClient()
        preds = run_llm_re(client, chunks, golden_ner, HAIKU_MODEL)
        metrics = evaluate(preds, golden, HAIKU_MODEL)
        all_results["haiku"] = {"predictions": preds, "metrics": metrics}
        print(f"  Exact F1={metrics['exact_match']['f1']:.3f}  Partial F1={metrics['partial_match']['f1']:.3f}")
        print(f"  Latency p50={metrics['latency_p50_ms']:.0f}ms  Cost=${metrics['cost_usd']:.4f}")

    # --- GPT-4o-mini ---
    if "gpt4omini" in model_list:
        print(f"\n[3/3] GPT-4o-mini ({GPT4OMINI_MODEL})...")
        client = OpenRouterClient()
        preds = run_llm_re(client, chunks, golden_ner, GPT4OMINI_MODEL)
        metrics = evaluate(preds, golden, GPT4OMINI_MODEL)
        all_results["gpt4omini"] = {"predictions": preds, "metrics": metrics}
        print(f"  Exact F1={metrics['exact_match']['f1']:.3f}  Partial F1={metrics['partial_match']['f1']:.3f}")
        print(f"  Latency p50={metrics['latency_p50_ms']:.0f}ms  Cost=${metrics['cost_usd']:.4f}")

    # Save results
    output = {
        "benchmark": "relation_extraction",
        "chunks_total": len(chunks),
        "golden_chunks": len(golden),
        "models": {k: v["metrics"] for k, v in all_results.items()},
        "predictions": {k: v["predictions"] for k, v in all_results.items()},
    }

    out_path = os.path.join(RESULTS_DIR, "re_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {out_path}")

    print(f"\n{'Model':<25} {'Exact F1':>9} {'Part F1':>8} {'p50ms':>7} {'Cost$':>8}")
    print("-" * 60)
    for key, data in all_results.items():
        m = data["metrics"]
        print(f"{m['model']:<25} {m['exact_match']['f1']:>9.3f} {m['partial_match']['f1']:>8.3f} {m['latency_p50_ms']:>7.0f} {m['cost_usd']:>8.4f}")


if __name__ == "__main__":
    main()
