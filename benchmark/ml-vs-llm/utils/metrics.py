"""Evaluation metrics for ML vs LLM benchmark."""

import time
from collections import Counter

import numpy as np


def compute_ner_f1(predicted: list[dict], golden: list[dict]) -> dict:
    """Compute NER F1 on (name, type) pairs.

    Args:
        predicted: [{"name": "...", "type": "..."}]
        golden: [{"name": "...", "type": "..."}]

    Returns:
        {"precision", "recall", "f1", "support", "tp", "fp", "fn"}
    """
    pred_set = {(e["name"].lower().strip(), e["type"].lower().strip()) for e in predicted}
    gold_set = {(e["name"].lower().strip(), e["type"].lower().strip()) for e in golden}

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": len(gold_set),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_ner_f1_by_type(predicted: list[dict], golden: list[dict]) -> dict[str, dict]:
    """Compute NER F1 per entity type."""
    all_types = {e["type"].lower().strip() for e in golden} | {e["type"].lower().strip() for e in predicted}

    results = {}
    for etype in sorted(all_types):
        pred_filtered = [e for e in predicted if e["type"].lower().strip() == etype]
        gold_filtered = [e for e in golden if e["type"].lower().strip() == etype]
        results[etype] = compute_ner_f1(pred_filtered, gold_filtered)

    return results


def compute_re_f1(predicted: list[dict], golden: list[dict], partial: bool = False) -> dict:
    """Compute Relation Extraction F1.

    Args:
        predicted: [{"source": "...", "relation": "...", "target": "..."}]
        golden: same format
        partial: if True, match on (source, target) only (ignore relation label)
    """
    def _key(e):
        s = e["source"].lower().strip()
        t = e["target"].lower().strip()
        if partial:
            return (s, t)
        return (s, e["relation"].lower().strip(), t)

    pred_set = {_key(e) for e in predicted}
    gold_set = {_key(e) for e in golden}

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": len(gold_set),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_classification_metrics(predicted: list[str], golden: list[str]) -> dict:
    """Compute accuracy and macro-F1 for classification."""
    assert len(predicted) == len(golden), "Lengths must match"

    correct = sum(1 for p, g in zip(predicted, golden) if p == g)
    accuracy = correct / len(golden) if golden else 0.0

    # Per-class F1
    all_labels = sorted(set(golden) | set(predicted))
    per_class = {}
    f1_scores = []

    for label in all_labels:
        tp = sum(1 for p, g in zip(predicted, golden) if p == label and g == label)
        fp = sum(1 for p, g in zip(predicted, golden) if p == label and g != label)
        fn = sum(1 for p, g in zip(predicted, golden) if p != label and g == label)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[label] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
        f1_scores.append(f1)

    return {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(np.mean(f1_scores), 4) if f1_scores else 0.0,
        "per_class": per_class,
        "support": len(golden),
        "confusion": dict(Counter(zip(golden, predicted))),
    }


def compute_ndcg(ranked_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """Compute nDCG@k for reranking."""
    relevant_set = set(relevant_ids)

    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG: all relevant docs at top
    ideal_count = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_count))

    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def compute_mrr(ranked_ids: list[str], relevant_ids: list[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(ranked_ids):
        if doc_id in relevant_set:
            return round(1.0 / (i + 1), 4)
    return 0.0


def measure_latency(func, *args, runs: int = 3, **kwargs) -> dict:
    """Measure function latency over multiple runs.

    Returns: {"p50_ms", "p95_ms", "mean_ms", "runs"}
    """
    times = []
    result = None
    for _ in range(runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    times_arr = np.array(times)
    return {
        "p50_ms": round(float(np.percentile(times_arr, 50)), 1),
        "p95_ms": round(float(np.percentile(times_arr, 95)), 1),
        "mean_ms": round(float(np.mean(times_arr)), 1),
        "runs": runs,
        "last_result": result,
    }


def compute_cost(input_tokens: int, output_tokens: int, model: str, pricing: dict) -> float:
    """Compute cost in USD for a given model."""
    if model not in pricing:
        return 0.0
    p = pricing[model]
    cost = (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
    return round(cost, 6)
