"""End-to-End Recall Benchmark: 4 reranking strategies on 36 LTM-0002 queries.

Compares recall@K of:
  1. No rerank (Scrutator original RRF order)
  2. BGE-Reranker-v2-m3 (ML cross-encoder, free)
  3. Claude Haiku (LLM via OpenRouter)
  4. GPT-4o-mini (LLM via OpenRouter)

Golden answers come from LTM-0002 benchmark JSONL files (snippet-based matching).

Usage:
    python benchmarks/bench_e2e_recall.py --queries-dir /path/to/queries
    python benchmarks/bench_e2e_recall.py --queries-dir /path/to/queries --limit 2  # smoke test
    python benchmarks/bench_e2e_recall.py --queries-dir /path/to/queries --models no_rerank,bge
"""

import argparse
import json
import os
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests as http_requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    GPT4OMINI_MODEL,
    HAIKU_MODEL,
    PRICING,
    RERANKER_MODEL,
    RESULTS_DIR,
    SCRUTATOR_URL,
)
from utils.llm_client import OpenRouterClient
from utils.metrics import compute_cost, compute_mrr, compute_ndcg


# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------

def load_queries(queries_dir: str, limit: int = 0) -> list[dict]:
    """Load queries from LTM-0002 JSONL files.

    Returns: [{"id", "class", "question", "golden_snippets": [str]}]
    """
    queries = []
    for fname in ["factual.jsonl", "temporal.jsonl", "multi-hop.jsonl"]:
        fpath = os.path.join(queries_dir, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                snippets = [
                    s["snippet"]
                    for s in entry.get("ground_truth", {}).get("sources", [])
                    if s.get("snippet")
                ]
                answer_summary = entry.get("ground_truth", {}).get("answer_summary", "")
                queries.append({
                    "id": entry["id"],
                    "class": entry["class"],
                    "question": entry["question"],
                    "golden_snippets": snippets,
                    "answer_summary": answer_summary,
                })
    if limit > 0:
        queries = queries[:limit]
    print(f"  Loaded {len(queries)} queries from {queries_dir}")
    return queries


# ---------------------------------------------------------------------------
# Scrutator search
# ---------------------------------------------------------------------------

def scrutator_search(query: str, top_n: int = 20) -> list[dict]:
    """Search Scrutator and return top-N results."""
    try:
        resp = http_requests.post(
            f"{SCRUTATOR_URL}/v1/search",
            json={"query": query, "top_n": top_n, "namespace": "arcanada"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        return [
            {
                "id": r.get("chunk_id", r.get("id", "")),
                "content": r.get("content", ""),
                "score": r.get("score", 0),
            }
            for r in results
        ]
    except Exception as e:
        print(f"  Scrutator error for '{query[:40]}': {e}")
        return []


# ---------------------------------------------------------------------------
# Reranking strategies
# ---------------------------------------------------------------------------

def rerank_none(candidates: list[dict]) -> list[str]:
    """Return IDs in original Scrutator order (by score desc)."""
    return [c["id"] for c in candidates]


def rerank_bge(query: str, candidates: list[dict], model) -> tuple[list[str], float]:
    """BGE-Reranker cross-encoder. Returns (ranked_ids, latency_ms)."""
    if not candidates:
        return [], 0.0
    pairs = [(query, c["content"][:500]) for c in candidates]
    start = time.perf_counter()
    scores = model.predict(pairs)
    latency_ms = (time.perf_counter() - start) * 1000
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c["id"] for c, _ in scored], latency_ms


def rerank_llm(
    query: str,
    candidates: list[dict],
    client: OpenRouterClient,
    model_name: str,
) -> tuple[list[str], float, int, int]:
    """LLM reranking. Returns (ranked_ids, latency_ms, in_tokens, out_tokens)."""
    if not candidates:
        return [], 0.0, 0, 0
    try:
        resp = client.rerank(query, candidates, model_name)
        return (
            resp["ranked_ids"],
            resp["latency_ms"],
            resp["input_tokens"],
            resp["output_tokens"],
        )
    except Exception as e:
        print(f"  LLM rerank error ({model_name}): {e}")
        return [c["id"] for c in candidates], 0.0, 0, 0


# ---------------------------------------------------------------------------
# Golden snippet matching
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalize for matching: lowercase, collapse whitespace, strip."""
    return _WS_RE.sub(" ", text.lower().strip())


def snippet_match(content: str, snippet: str, threshold: float = 0.8) -> bool:
    """Check if golden snippet appears in content."""
    nc = normalize(content)
    ns = normalize(snippet)
    # 1. Exact substring
    if ns in nc:
        return True
    # 2. Fuzzy match on sliding window
    if len(ns) < 10:
        return False
    ratio = SequenceMatcher(None, ns, nc).ratio()
    if ratio >= threshold:
        return True
    # 3. Try matching against a window of similar length
    win_len = len(ns)
    best = 0.0
    step = max(1, win_len // 4)
    for i in range(0, max(1, len(nc) - win_len + 1), step):
        window = nc[i : i + win_len]
        r = SequenceMatcher(None, ns, window).ratio()
        if r > best:
            best = r
        if best >= threshold:
            return True
    return best >= threshold


def evaluate_recall(
    ranked_ids: list[str],
    candidates_map: dict[str, str],
    golden_snippets: list[str],
    k: int = 5,
    answer_summary: str = "",
) -> dict:
    """Check how many golden snippets are found in top-K results.

    Uses two-tier matching:
    1. Snippet-level: check if each golden snippet appears in top-K content
    2. Answer-level: if no snippet hits, check if answer_summary appears in top-K

    Args:
        ranked_ids: ordered list of chunk IDs
        candidates_map: {chunk_id: content}
        golden_snippets: list of golden text snippets
        k: top-K to evaluate
        answer_summary: fallback answer text to match

    Returns: {"recall_at_k": float, "hits": int, "total": int, "hit_details": [...]}
    """
    top_k_ids = ranked_ids[:k]
    top_k_contents = [candidates_map.get(cid, "") for cid in top_k_ids]

    hits = 0
    details = []
    for snippet in golden_snippets:
        found = False
        for content in top_k_contents:
            if snippet_match(content, snippet):
                found = True
                break
        hits += int(found)
        details.append({"snippet": snippet[:80], "found": found})

    # Fallback: if no snippet matched but answer_summary is in top-K content
    if hits == 0 and answer_summary:
        na = normalize(answer_summary)
        for content in top_k_contents:
            if na in normalize(content):
                hits = 1
                details.append({"snippet": f"[answer_summary] {answer_summary[:60]}", "found": True})
                break

    total = max(len(golden_snippets), 1)  # at least 1 to avoid division by zero
    return {
        "recall_at_k": round(hits / total, 4) if total > 0 else 0.0,
        "hits": hits,
        "total": total,
        "hit_details": details,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="E2E Recall Benchmark (SRCH-0018)")
    parser.add_argument(
        "--queries-dir",
        required=True,
        help="Path to LTM-0002 queries dir (contains factual.jsonl, temporal.jsonl, multi-hop.jsonl)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit queries (0=all)")
    parser.add_argument(
        "--models",
        default="no_rerank,bge,haiku,gpt4omini",
        help="Comma-separated strategies",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Scrutator top-N candidates")
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",")]
    queries = load_queries(args.queries_dir, limit=args.limit)
    if not queries:
        print("ERROR: No queries loaded")
        sys.exit(1)

    print(f"\nE2E Recall Benchmark: {len(queries)} queries, strategies: {model_list}")
    print(f"Scrutator: {SCRUTATOR_URL}, namespace=arcanada, top_n={args.top_n}")
    print("=" * 70)

    # Step 1: Fetch candidates from Scrutator for all queries
    print("\n[Step 1] Fetching candidates from Scrutator...")
    queries_data = []
    for i, q in enumerate(queries):
        candidates = scrutator_search(q["question"], top_n=args.top_n)
        candidates_map = {c["id"]: c["content"] for c in candidates}
        queries_data.append({
            **q,
            "candidates": candidates,
            "candidates_map": candidates_map,
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(queries):
            print(f"  {i + 1}/{len(queries)} queries fetched")
    time.sleep(0.5)

    # Step 2: Run each reranking strategy
    all_results = {}

    # --- No rerank ---
    if "no_rerank" in model_list:
        print("\n[Strategy 1] No rerank (Scrutator original order)...")
        strategy_results = []
        for qd in queries_data:
            ranked_ids = rerank_none(qd["candidates"])
            r5 = evaluate_recall(ranked_ids, qd["candidates_map"], qd["golden_snippets"], k=5, answer_summary=qd.get("answer_summary", ""))
            r10 = evaluate_recall(ranked_ids, qd["candidates_map"], qd["golden_snippets"], k=10, answer_summary=qd.get("answer_summary", ""))
            strategy_results.append({
                "query_id": qd["id"],
                "query_class": qd["class"],
                "recall_at_5": r5["recall_at_k"],
                "recall_at_10": r10["recall_at_k"],
                "hits_5": r5["hits"],
                "hits_10": r10["hits"],
                "total_snippets": r5["total"],
                "latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            })
        all_results["no_rerank"] = strategy_results

    # --- BGE-Reranker ---
    if "bge" in model_list:
        print("\n[Strategy 2] BGE-Reranker-v2-m3...")
        from sentence_transformers import CrossEncoder
        bge_model = CrossEncoder(RERANKER_MODEL)
        strategy_results = []
        for i, qd in enumerate(queries_data):
            ranked_ids, lat = rerank_bge(qd["question"], qd["candidates"], bge_model)
            r5 = evaluate_recall(ranked_ids, qd["candidates_map"], qd["golden_snippets"], k=5, answer_summary=qd.get("answer_summary", ""))
            r10 = evaluate_recall(ranked_ids, qd["candidates_map"], qd["golden_snippets"], k=10, answer_summary=qd.get("answer_summary", ""))
            strategy_results.append({
                "query_id": qd["id"],
                "query_class": qd["class"],
                "recall_at_5": r5["recall_at_k"],
                "recall_at_10": r10["recall_at_k"],
                "hits_5": r5["hits"],
                "hits_10": r10["hits"],
                "total_snippets": r5["total"],
                "latency_ms": round(lat, 1),
                "input_tokens": 0,
                "output_tokens": 0,
            })
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(queries_data)} queries reranked")
        all_results["bge"] = strategy_results

    # --- Haiku ---
    if "haiku" in model_list:
        print(f"\n[Strategy 3] Claude Haiku ({HAIKU_MODEL})...")
        client = OpenRouterClient()
        strategy_results = []
        for i, qd in enumerate(queries_data):
            ranked_ids, lat, in_t, out_t = rerank_llm(
                qd["question"], qd["candidates"], client, HAIKU_MODEL
            )
            r5 = evaluate_recall(ranked_ids, qd["candidates_map"], qd["golden_snippets"], k=5, answer_summary=qd.get("answer_summary", ""))
            r10 = evaluate_recall(ranked_ids, qd["candidates_map"], qd["golden_snippets"], k=10, answer_summary=qd.get("answer_summary", ""))
            strategy_results.append({
                "query_id": qd["id"],
                "query_class": qd["class"],
                "recall_at_5": r5["recall_at_k"],
                "recall_at_10": r10["recall_at_k"],
                "hits_5": r5["hits"],
                "hits_10": r10["hits"],
                "total_snippets": r5["total"],
                "latency_ms": round(lat, 1),
                "input_tokens": in_t,
                "output_tokens": out_t,
            })
            time.sleep(0.3)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(queries_data)} queries reranked")
        all_results["haiku"] = strategy_results

    # --- GPT-4o-mini ---
    if "gpt4omini" in model_list:
        print(f"\n[Strategy 4] GPT-4o-mini ({GPT4OMINI_MODEL})...")
        client = OpenRouterClient()
        strategy_results = []
        for i, qd in enumerate(queries_data):
            ranked_ids, lat, in_t, out_t = rerank_llm(
                qd["question"], qd["candidates"], client, GPT4OMINI_MODEL
            )
            r5 = evaluate_recall(ranked_ids, qd["candidates_map"], qd["golden_snippets"], k=5, answer_summary=qd.get("answer_summary", ""))
            r10 = evaluate_recall(ranked_ids, qd["candidates_map"], qd["golden_snippets"], k=10, answer_summary=qd.get("answer_summary", ""))
            strategy_results.append({
                "query_id": qd["id"],
                "query_class": qd["class"],
                "recall_at_5": r5["recall_at_k"],
                "recall_at_10": r10["recall_at_k"],
                "hits_5": r5["hits"],
                "hits_10": r10["hits"],
                "total_snippets": r5["total"],
                "latency_ms": round(lat, 1),
                "input_tokens": in_t,
                "output_tokens": out_t,
            })
            time.sleep(0.3)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(queries_data)} queries reranked")
        all_results["gpt4omini"] = strategy_results

    # Step 3: Aggregate and save
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    summary = {}
    for strategy, results in all_results.items():
        # Overall
        r5_avg = sum(r["recall_at_5"] for r in results) / len(results)
        r10_avg = sum(r["recall_at_10"] for r in results) / len(results)
        lat_avg = sum(r["latency_ms"] for r in results) / len(results) if results else 0
        total_in = sum(r["input_tokens"] for r in results)
        total_out = sum(r["output_tokens"] for r in results)

        # Determine model name for cost
        model_for_cost = {
            "no_rerank": RERANKER_MODEL,
            "bge": RERANKER_MODEL,
            "haiku": HAIKU_MODEL,
            "gpt4omini": GPT4OMINI_MODEL,
        }.get(strategy, RERANKER_MODEL)
        cost = compute_cost(total_in, total_out, model_for_cost, PRICING)

        # Per-class breakdown
        per_class = {}
        for cls in ["factual", "temporal", "multi-hop"]:
            cls_results = [r for r in results if r["query_class"] == cls]
            if cls_results:
                per_class[cls] = {
                    "recall_at_5": round(sum(r["recall_at_5"] for r in cls_results) / len(cls_results), 4),
                    "recall_at_10": round(sum(r["recall_at_10"] for r in cls_results) / len(cls_results), 4),
                    "count": len(cls_results),
                }

        summary[strategy] = {
            "recall_at_5": round(r5_avg, 4),
            "recall_at_10": round(r10_avg, 4),
            "latency_mean_ms": round(lat_avg, 1),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "cost_usd": cost,
            "queries_evaluated": len(results),
            "per_class": per_class,
        }

    # Print summary table
    print(f"\n{'Strategy':<20} {'recall@5':>9} {'recall@10':>10} {'lat_ms':>8} {'cost$':>8}")
    print("-" * 60)
    for strategy, s in summary.items():
        print(
            f"{strategy:<20} {s['recall_at_5']:>9.3f} {s['recall_at_10']:>10.3f} "
            f"{s['latency_mean_ms']:>8.0f} {s['cost_usd']:>8.4f}"
        )

    # Per-class breakdown
    print(f"\nPer-class recall@5:")
    print(f"{'Strategy':<20} {'factual':>9} {'temporal':>10} {'multi-hop':>10}")
    print("-" * 52)
    for strategy, s in summary.items():
        pc = s["per_class"]
        print(
            f"{strategy:<20} "
            f"{pc.get('factual', {}).get('recall_at_5', 0):>9.3f} "
            f"{pc.get('temporal', {}).get('recall_at_5', 0):>10.3f} "
            f"{pc.get('multi-hop', {}).get('recall_at_5', 0):>10.3f}"
        )

    # Save full results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "benchmark": "e2e_recall",
        "queries_total": len(queries),
        "top_n_candidates": args.top_n,
        "scrutator_url": SCRUTATOR_URL,
        "summary": summary,
        "per_query": {strategy: results for strategy, results in all_results.items()},
    }
    out_path = os.path.join(RESULTS_DIR, "e2e_recall_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
