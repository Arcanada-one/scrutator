"""Reranking Benchmark: BGE-Reranker-v2-m3 vs Claude Haiku vs GPT-4o-mini.

For each query:
1. Scrutator hybrid search → top-20 candidates
2. Rerank with each model
3. Evaluate against golden relevance (top-5 from golden = relevant)

Usage:
    python benchmarks/bench_reranking.py [--queries N] [--models reranker,haiku,gpt4omini]
"""

import argparse
import json
import os
import sys
import time

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

# 50 test queries spanning all 6 areas
TEST_QUERIES = [
    # AI (10)
    "embedding models for knowledge graph",
    "how does BGE-M3 hybrid search work",
    "entity extraction from text chunks",
    "LLM vs ML for NER tasks",
    "vector database indexing strategies",
    "knowledge graph deduplication",
    "transformer architecture for NLP",
    "RAG retrieval augmented generation",
    "fine-tuning language models",
    "prompt engineering best practices",
    # Arcana (10)
    "Arcanada ecosystem architecture",
    "Agent Dreamer wiki organization",
    "Scrutator search engine design",
    "Model Connector API integration",
    "datarim workflow management",
    "Verdicus macOS application",
    "Ops Bot monitoring system",
    "Personal Assistant telegram bot",
    "Disk Arcana file synchronization",
    "Long Term Memory project",
    # Security (8)
    "SSL certificate management Cloudflare",
    "HashiCorp Vault secret storage",
    "Tailscale mesh VPN setup",
    "server access control policies",
    "API authentication bearer tokens",
    "SSH key management GitHub",
    "rate limiting and DDoS protection",
    "Docker container security",
    # Business (7)
    "startup revenue model",
    "content marketing strategy",
    "social media engagement optimization",
    "freelance business management",
    "SaaS pricing models",
    "customer acquisition cost",
    "project management methodology",
    # Tools (8)
    "Docker compose deployment",
    "CI/CD GitHub Actions pipeline",
    "Nginx reverse proxy configuration",
    "PostgreSQL performance tuning",
    "Redis caching strategies",
    "NestJS FastAPI framework comparison",
    "Obsidian knowledge management",
    "git workflow branching strategy",
    # Philosophy (7)
    "meaning of human life",
    "AI ethics and consciousness",
    "rules of robotics for AI agents",
    "digital immortality concept",
    "knowledge and wisdom distinction",
    "human-AI collaboration future",
    "philosophy of technology",
]


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
                "content": r.get("content", r.get("content_preview", "")),
                "score": r.get("score", 0),
            }
            for r in results
        ]
    except Exception as e:
        print(f"  Scrutator search error: {e}")
        return []


def run_bge_reranker(queries_data: list[dict]) -> list[dict]:
    """Rerank using BGE-Reranker-v2-m3 via CrossEncoder."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(RERANKER_MODEL)
    results = []

    for qd in queries_data:
        query = qd["query"]
        candidates = qd["candidates"]
        if not candidates:
            results.append({
                "query": query,
                "ranked_ids": [],
                "latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            })
            continue

        pairs = [(query, c["content"][:500]) for c in candidates]
        start = time.perf_counter()
        scores = model.predict(pairs)
        latency_ms = (time.perf_counter() - start) * 1000

        scored = list(zip(candidates, scores, strict=False))
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_ids = [c["id"] for c, _ in scored]

        results.append({
            "query": query,
            "ranked_ids": ranked_ids,
            "latency_ms": round(latency_ms, 1),
            "input_tokens": 0,
            "output_tokens": 0,
        })

    return results


def run_llm_rerank(
    client: OpenRouterClient,
    queries_data: list[dict],
    model: str,
    sleep_between: float = 0.3,
) -> list[dict]:
    """Rerank using LLM."""
    results = []
    for qd in queries_data:
        query = qd["query"]
        candidates = qd["candidates"]
        if not candidates:
            results.append({
                "query": query,
                "ranked_ids": [],
                "latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            })
            continue

        try:
            resp = client.rerank(query, candidates, model)
            results.append({
                "query": query,
                "ranked_ids": resp["ranked_ids"],
                "latency_ms": resp["latency_ms"],
                "input_tokens": resp["input_tokens"],
                "output_tokens": resp["output_tokens"],
            })
        except Exception as e:
            print(f"  ERROR rerank '{query[:30]}': {e}")
            results.append({
                "query": query,
                "ranked_ids": [],
                "latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e),
            })
        if sleep_between > 0:
            time.sleep(sleep_between)
    return results


def evaluate(
    predictions: list[dict],
    golden_ranking: dict[str, list[str]],
    model_name: str,
) -> dict:
    """Evaluate reranking: nDCG@5, MRR."""
    ndcgs = []
    mrrs = []
    latencies = []
    total_in = 0
    total_out = 0
    errors = 0

    for pred in predictions:
        query = pred["query"]
        if query not in golden_ranking or not pred["ranked_ids"]:
            if "error" in pred:
                errors += 1
            continue

        relevant = golden_ranking[query]
        ndcg = compute_ndcg(pred["ranked_ids"], relevant, k=5)
        mrr = compute_mrr(pred["ranked_ids"], relevant)
        ndcgs.append(ndcg)
        mrrs.append(mrr)
        latencies.append(pred["latency_ms"])
        total_in += pred["input_tokens"]
        total_out += pred["output_tokens"]

    cost = compute_cost(total_in, total_out, model_name, PRICING)

    return {
        "model": model_name,
        "ndcg_at_5": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0,
        "mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0,
        "queries_evaluated": len(ndcgs),
        "queries_errors": errors,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "cost_usd": cost,
        "latency_mean_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "latency_p50_ms": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0,
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if latencies else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Reranking Benchmark")
    parser.add_argument("--queries", type=int, default=0, help="Limit queries (0=all 50)")
    parser.add_argument("--models", default="reranker,haiku,gpt4omini", help="Comma-separated")
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",")]
    queries = TEST_QUERIES
    if args.queries > 0:
        queries = queries[: args.queries]

    print(f"Reranking Benchmark: {len(queries)} queries, models: {model_list}")
    print("=" * 60)

    # Step 1: Get candidates from Scrutator for each query
    print("\nFetching candidates from Scrutator...")
    queries_data = []
    for i, q in enumerate(queries):
        candidates = scrutator_search(q, top_n=20)
        queries_data.append({"query": q, "candidates": candidates})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(queries)} queries fetched")

    # Golden ranking = Scrutator's original top-5 as relevant
    # (We use the search engine's own ranking as ground truth since
    # these are hybrid search results combining dense+sparse+FTS)
    golden_ranking = {}
    for qd in queries_data:
        if qd["candidates"]:
            golden_ranking[qd["query"]] = [c["id"] for c in qd["candidates"][:5]]

    print(f"  {len(golden_ranking)} queries with candidates")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    # --- BGE-Reranker ---
    if "reranker" in model_list:
        print("\n[1/3] BGE-Reranker-v2-m3...")
        preds = run_bge_reranker(queries_data)
        metrics = evaluate(preds, golden_ranking, RERANKER_MODEL)
        all_results["reranker"] = {"predictions": preds, "metrics": metrics}
        print(f"  nDCG@5={metrics['ndcg_at_5']:.3f}  MRR={metrics['mrr']:.3f}")
        print(f"  Latency p50={metrics['latency_p50_ms']:.0f}ms  Cost=${metrics['cost_usd']:.4f}")

    # --- Haiku ---
    if "haiku" in model_list:
        print(f"\n[2/3] Claude Haiku ({HAIKU_MODEL})...")
        client = OpenRouterClient()
        preds = run_llm_rerank(client, queries_data, HAIKU_MODEL)
        metrics = evaluate(preds, golden_ranking, HAIKU_MODEL)
        all_results["haiku"] = {"predictions": preds, "metrics": metrics}
        print(f"  nDCG@5={metrics['ndcg_at_5']:.3f}  MRR={metrics['mrr']:.3f}")
        print(f"  Latency p50={metrics['latency_p50_ms']:.0f}ms  Cost=${metrics['cost_usd']:.4f}")

    # --- GPT-4o-mini ---
    if "gpt4omini" in model_list:
        print(f"\n[3/3] GPT-4o-mini ({GPT4OMINI_MODEL})...")
        client = OpenRouterClient()
        preds = run_llm_rerank(client, queries_data, GPT4OMINI_MODEL)
        metrics = evaluate(preds, golden_ranking, GPT4OMINI_MODEL)
        all_results["gpt4omini"] = {"predictions": preds, "metrics": metrics}
        print(f"  nDCG@5={metrics['ndcg_at_5']:.3f}  MRR={metrics['mrr']:.3f}")
        print(f"  Latency p50={metrics['latency_p50_ms']:.0f}ms  Cost=${metrics['cost_usd']:.4f}")

    # Save results
    output = {
        "benchmark": "reranking",
        "queries_total": len(queries),
        "queries_with_candidates": len(golden_ranking),
        "models": {k: v["metrics"] for k, v in all_results.items()},
        "predictions": {k: v["predictions"] for k, v in all_results.items()},
    }

    out_path = os.path.join(RESULTS_DIR, "rerank_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {out_path}")

    print(f"\n{'Model':<30} {'nDCG@5':>7} {'MRR':>6} {'p50ms':>7} {'Cost$':>8}")
    print("-" * 60)
    for _key, data in all_results.items():
        m = data["metrics"]
        print(
            f"{m['model']:<30} {m['ndcg_at_5']:>7.3f} {m['mrr']:>6.3f} "
            f"{m['latency_p50_ms']:>7.0f} {m['cost_usd']:>8.4f}"
        )


if __name__ == "__main__":
    main()
