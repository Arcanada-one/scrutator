"""Classification Benchmark: SetFit (zero-shot NLI) vs Claude Haiku vs GPT-4o-mini.

Classifies chunks into 6 areas: AI, arcana, security, business, tools, philosophy.

Usage:
    python benchmarks/bench_classification.py [--chunks N] [--models setfit,haiku,gpt4omini]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    AREA_LABELS,
    GOLDEN_DIR,
    GPT4OMINI_MODEL,
    HAIKU_MODEL,
    PRICING,
    RESULTS_DIR,
)
from utils.llm_client import OpenRouterClient
from utils.metrics import compute_classification_metrics, compute_cost


def load_golden_class() -> dict[str, str]:
    """Load golden classification: {chunk_id: area}."""
    with open(os.path.join(GOLDEN_DIR, "golden_class.json")) as f:
        data = json.load(f)
    return {item["chunk_id"]: item["area"] for item in data}


def load_corpus() -> dict[str, dict]:
    with open(os.path.join(GOLDEN_DIR, "corpus_100.json")) as f:
        data = json.load(f)
    return {item["chunk_id"]: item for item in data}


# Area descriptions for zero-shot NLI
AREA_DESCRIPTIONS = {
    "AI": "artificial intelligence, machine learning, neural networks, LLM, embeddings, NLP",
    "arcana": "Arcanada ecosystem, personal projects, wiki organization, knowledge base",
    "security": "cybersecurity, encryption, authentication, access control, vulnerabilities",
    "business": "entrepreneurship, startups, revenue, marketing, business strategy",
    "tools": "software tools, development tools, IDE, CLI, DevOps, infrastructure",
    "philosophy": "philosophy, ethics, meaning of life, consciousness, human values",
}


def run_setfit(chunks: list[dict], labels: list[str]) -> list[dict]:
    """Run zero-shot classification via embedding similarity (SentenceTransformer)."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

    # Encode area descriptions
    label_texts = [f"{label}: {AREA_DESCRIPTIONS[label]}" for label in labels]
    label_embeddings = st_model.encode(label_texts, normalize_embeddings=True)

    results = []
    for chunk in chunks:
        text = chunk["content"][:512]
        start = time.perf_counter()

        text_emb = st_model.encode([text], normalize_embeddings=True)
        similarities = np.dot(text_emb, label_embeddings.T)[0]
        predicted_idx = int(np.argmax(similarities))
        predicted_label = labels[predicted_idx]

        latency_ms = (time.perf_counter() - start) * 1000
        results.append({
            "chunk_id": chunk["chunk_id"],
            "area": predicted_label,
            "confidence": round(float(similarities[predicted_idx]), 4),
            "latency_ms": round(latency_ms, 1),
            "input_tokens": 0,
            "output_tokens": 0,
        })

    return results


def run_llm_classify(
    client: OpenRouterClient,
    chunks: list[dict],
    model: str,
    labels: list[str],
    sleep_between: float = 0.3,
) -> list[dict]:
    """Run LLM classification."""
    results = []
    for chunk in chunks:
        try:
            resp = client.classify_area(chunk["content"], model)
            # Normalize area label
            area = resp["area"].strip().lower()
            # Match to valid labels (case-insensitive)
            matched = next((l for l in labels if l.lower() == area), area)
            results.append({
                "chunk_id": chunk["chunk_id"],
                "area": matched,
                "latency_ms": resp["latency_ms"],
                "input_tokens": resp["input_tokens"],
                "output_tokens": resp["output_tokens"],
            })
        except Exception as e:
            print(f"  ERROR {chunk['chunk_id'][:8]}: {e}")
            results.append({
                "chunk_id": chunk["chunk_id"],
                "area": "unknown",
                "latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e),
            })
        if sleep_between > 0:
            time.sleep(sleep_between)
    return results


def evaluate(predictions: list[dict], golden: dict[str, str], model_name: str) -> dict:
    """Evaluate classification predictions."""
    pred_labels = []
    gold_labels = []
    latencies = []
    total_in = 0
    total_out = 0
    errors = 0

    for pred in predictions:
        cid = pred["chunk_id"]
        if cid not in golden:
            continue
        pred_labels.append(pred["area"])
        gold_labels.append(golden[cid])
        latencies.append(pred["latency_ms"])
        total_in += pred["input_tokens"]
        total_out += pred["output_tokens"]
        if "error" in pred:
            errors += 1

    metrics = compute_classification_metrics(pred_labels, gold_labels)
    cost = compute_cost(total_in, total_out, model_name, PRICING)

    return {
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "per_class": metrics["per_class"],
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
    parser = argparse.ArgumentParser(description="Classification Benchmark")
    parser.add_argument("--chunks", type=int, default=0, help="Limit chunks (0=all)")
    parser.add_argument("--models", default="setfit,haiku,gpt4omini", help="Comma-separated")
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",")]

    golden = load_golden_class()
    corpus = load_corpus()
    golden_ids = list(golden.keys())
    chunks = [corpus[cid] for cid in golden_ids if cid in corpus]
    if args.chunks > 0:
        chunks = chunks[: args.chunks]

    print(f"Classification Benchmark: {len(chunks)} chunks, models: {model_list}")
    print(f"Labels: {AREA_LABELS}")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    # --- SetFit ---
    if "setfit" in model_list:
        print("\n[1/3] SetFit (zero-shot NLI)...")
        preds = run_setfit(chunks, AREA_LABELS)
        metrics = evaluate(preds, golden, "setfit-paraphrase-MiniLM")
        all_results["setfit"] = {"predictions": preds, "metrics": metrics}
        print(f"  Accuracy={metrics['accuracy']:.3f}  F1 macro={metrics['f1_macro']:.3f}")
        print(f"  Latency p50={metrics['latency_p50_ms']:.0f}ms  Cost=${metrics['cost_usd']:.4f}")

    # --- Haiku ---
    if "haiku" in model_list:
        print(f"\n[2/3] Claude Haiku ({HAIKU_MODEL})...")
        client = OpenRouterClient()
        preds = run_llm_classify(client, chunks, HAIKU_MODEL, AREA_LABELS)
        metrics = evaluate(preds, golden, HAIKU_MODEL)
        all_results["haiku"] = {"predictions": preds, "metrics": metrics}
        print(f"  Accuracy={metrics['accuracy']:.3f}  F1 macro={metrics['f1_macro']:.3f}")
        print(f"  Latency p50={metrics['latency_p50_ms']:.0f}ms  Cost=${metrics['cost_usd']:.4f}")

    # --- GPT-4o-mini ---
    if "gpt4omini" in model_list:
        print(f"\n[3/3] GPT-4o-mini ({GPT4OMINI_MODEL})...")
        client = OpenRouterClient()
        preds = run_llm_classify(client, chunks, GPT4OMINI_MODEL, AREA_LABELS)
        metrics = evaluate(preds, golden, GPT4OMINI_MODEL)
        all_results["gpt4omini"] = {"predictions": preds, "metrics": metrics}
        print(f"  Accuracy={metrics['accuracy']:.3f}  F1 macro={metrics['f1_macro']:.3f}")
        print(f"  Latency p50={metrics['latency_p50_ms']:.0f}ms  Cost=${metrics['cost_usd']:.4f}")

    # Save results
    output = {
        "benchmark": "classification",
        "chunks_total": len(chunks),
        "golden_chunks": len(golden),
        "labels": AREA_LABELS,
        "models": {k: v["metrics"] for k, v in all_results.items()},
        "predictions": {k: v["predictions"] for k, v in all_results.items()},
    }

    out_path = os.path.join(RESULTS_DIR, "class_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {out_path}")

    print(f"\n{'Model':<35} {'Acc':>6} {'F1m':>6} {'p50ms':>7} {'Cost$':>8}")
    print("-" * 65)
    for key, data in all_results.items():
        m = data["metrics"]
        print(f"{m['model']:<35} {m['accuracy']:>6.3f} {m['f1_macro']:>6.3f} {m['latency_p50_ms']:>7.0f} {m['cost_usd']:>8.4f}")


if __name__ == "__main__":
    main()
