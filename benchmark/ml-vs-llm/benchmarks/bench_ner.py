"""NER Benchmark: GLiNER2 vs Claude Haiku vs GPT-4o-mini.

Runs all 3 models on golden NER dataset (70 chunks), measures:
- F1, precision, recall (overall + per entity type)
- Latency (per chunk)
- Cost (LLM only)

Usage:
    python benchmarks/bench_ner.py [--chunks N] [--models gliner,haiku,gpt4omini]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ENTITY_TYPES,
    GLINER2_MODEL,
    GOLDEN_DIR,
    GPT4OMINI_MODEL,
    HAIKU_MODEL,
    PRICING,
    RESULTS_DIR,
)
from utils.llm_client import OpenRouterClient
from utils.metrics import compute_cost, compute_ner_f1, compute_ner_f1_by_type


def load_golden_ner() -> dict[str, list[dict]]:
    """Load golden NER: {chunk_id: [entities]}."""
    with open(os.path.join(GOLDEN_DIR, "golden_ner.json")) as f:
        data = json.load(f)
    return {item["chunk_id"]: item["entities"] for item in data}


def load_corpus() -> dict[str, dict]:
    """Load corpus: {chunk_id: chunk_data}."""
    with open(os.path.join(GOLDEN_DIR, "corpus_100.json")) as f:
        data = json.load(f)
    return {item["chunk_id"]: item for item in data}


def run_gliner(chunks: list[dict], entity_types: list[str]) -> list[dict]:
    """Run GLiNER2 on chunks. Returns per-chunk results."""
    from gliner import GLiNER

    model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    results = []

    for chunk in chunks:
        text = chunk["content"]
        start = time.perf_counter()
        raw = model.predict_entities(text, entity_types, threshold=0.4)
        latency_ms = (time.perf_counter() - start) * 1000

        entities = [{"name": e["text"], "type": e["label"]} for e in raw]
        results.append({
            "chunk_id": chunk["chunk_id"],
            "entities": entities,
            "latency_ms": round(latency_ms, 1),
            "input_tokens": 0,
            "output_tokens": 0,
        })

    return results


def run_llm(
    client: OpenRouterClient,
    chunks: list[dict],
    model: str,
    sleep_between: float = 0.3,
) -> list[dict]:
    """Run LLM NER on chunks via OpenRouter."""
    results = []

    for chunk in chunks:
        text = chunk["content"]
        try:
            resp = client.extract_entities(text, model)
            results.append({
                "chunk_id": chunk["chunk_id"],
                "entities": resp["entities"],
                "latency_ms": resp["latency_ms"],
                "input_tokens": resp["input_tokens"],
                "output_tokens": resp["output_tokens"],
            })
        except Exception as e:
            print(f"  ERROR {chunk['chunk_id'][:8]}: {e}")
            results.append({
                "chunk_id": chunk["chunk_id"],
                "entities": [],
                "latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e),
            })

        if sleep_between > 0:
            time.sleep(sleep_between)

    return results


def evaluate(predictions: list[dict], golden: dict[str, list[dict]], model_name: str) -> dict:
    """Evaluate predictions against golden, return aggregate metrics."""
    all_pred = []
    all_gold = []
    latencies = []
    total_input = 0
    total_output = 0
    errors = 0

    for pred in predictions:
        cid = pred["chunk_id"]
        if cid not in golden:
            continue

        all_pred.extend(pred["entities"])
        all_gold.extend(golden[cid])
        latencies.append(pred["latency_ms"])
        total_input += pred["input_tokens"]
        total_output += pred["output_tokens"]
        if "error" in pred:
            errors += 1

    overall = compute_ner_f1(all_pred, all_gold)
    by_type = compute_ner_f1_by_type(all_pred, all_gold)
    cost = compute_cost(total_input, total_output, model_name, PRICING)

    return {
        "model": model_name,
        "overall": overall,
        "by_type": by_type,
        "chunks_processed": len(predictions) - errors,
        "chunks_errors": errors,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "cost_usd": cost,
        "latency_mean_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "latency_p50_ms": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0,
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if latencies else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="NER Benchmark")
    parser.add_argument("--chunks", type=int, default=0, help="Limit chunks (0=all)")
    parser.add_argument("--models", default="gliner,haiku,gpt4omini", help="Comma-separated model list")
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",")]

    # Load data
    golden = load_golden_ner()
    corpus = load_corpus()
    golden_ids = list(golden.keys())

    # Filter corpus to golden chunks only
    chunks = [corpus[cid] for cid in golden_ids if cid in corpus]
    if args.chunks > 0:
        chunks = chunks[: args.chunks]

    print(f"NER Benchmark: {len(chunks)} chunks, models: {model_list}")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    # --- GLiNER ---
    if "gliner" in model_list:
        print(f"\n[1/3] GLiNER2 ({GLINER2_MODEL})...")
        preds = run_gliner(chunks, ENTITY_TYPES)
        metrics = evaluate(preds, golden, GLINER2_MODEL)
        all_results["gliner"] = {"predictions": preds, "metrics": metrics}
        print(
            f"  F1={metrics['overall']['f1']:.3f}  P={metrics['overall']['precision']:.3f}  "
            f"R={metrics['overall']['recall']:.3f}"
        )
        print(f"  Latency p50={metrics['latency_p50_ms']:.0f}ms  Cost=${metrics['cost_usd']:.4f}")

    # --- Haiku ---
    if "haiku" in model_list:
        print(f"\n[2/3] Claude Haiku ({HAIKU_MODEL})...")
        client = OpenRouterClient()
        preds = run_llm(client, chunks, HAIKU_MODEL)
        metrics = evaluate(preds, golden, HAIKU_MODEL)
        all_results["haiku"] = {"predictions": preds, "metrics": metrics}
        print(
            f"  F1={metrics['overall']['f1']:.3f}  P={metrics['overall']['precision']:.3f}  "
            f"R={metrics['overall']['recall']:.3f}"
        )
        print(f"  Latency p50={metrics['latency_p50_ms']:.0f}ms  Cost=${metrics['cost_usd']:.4f}")

    # --- GPT-4o-mini ---
    if "gpt4omini" in model_list:
        print(f"\n[3/3] GPT-4o-mini ({GPT4OMINI_MODEL})...")
        client = OpenRouterClient()
        preds = run_llm(client, chunks, GPT4OMINI_MODEL)
        metrics = evaluate(preds, golden, GPT4OMINI_MODEL)
        all_results["gpt4omini"] = {"predictions": preds, "metrics": metrics}
        print(
            f"  F1={metrics['overall']['f1']:.3f}  P={metrics['overall']['precision']:.3f}  "
            f"R={metrics['overall']['recall']:.3f}"
        )
        print(f"  Latency p50={metrics['latency_p50_ms']:.0f}ms  Cost=${metrics['cost_usd']:.4f}")

    # Save results
    output = {
        "benchmark": "ner",
        "chunks_total": len(chunks),
        "golden_chunks": len(golden),
        "models": {k: v["metrics"] for k, v in all_results.items()},
        "predictions": {k: v["predictions"] for k, v in all_results.items()},
    }

    out_path = os.path.join(RESULTS_DIR, "ner_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {out_path}")

    # Summary table
    print(f"\n{'Model':<30} {'F1':>6} {'Prec':>6} {'Rec':>6} {'p50ms':>7} {'Cost$':>8}")
    print("-" * 65)
    for _key, data in all_results.items():
        m = data["metrics"]
        print(
            f"{m['model']:<30} {m['overall']['f1']:>6.3f} {m['overall']['precision']:>6.3f} "
            f"{m['overall']['recall']:>6.3f} {m['latency_p50_ms']:>7.0f} {m['cost_usd']:>8.4f}"
        )


if __name__ == "__main__":
    main()
