#!/usr/bin/env python3
"""LTM-0009: Run 36 queries (10 factual / 11 multi-hop / 15 temporal) against Scrutator LTM and compute recall metrics.

Usage:
  python3 ltm-bench-query.py [--namespace ltm-bench-datarim-kb] [--expand-entities] [--no-expand-entities]

Outputs JSON report compatible with existing benchmark report format.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

SCRUTATOR = "http://100.70.137.104:8310"
QUERIES_DIR = str(Path(__file__).parent.parent / "queries")
DEFAULT_NS = "ltm-bench-datarim-kb"
REPORTS_DIR_V4 = str(Path(__file__).parent.parent / "reports/v4/scrutator")
REPORTS_DIR_V5 = str(Path(__file__).parent.parent / "reports/v5/scrutator")


def recall(query: str, namespace: str, expand_entities: bool, limit: int = 20) -> dict:
    payload = json.dumps({
        "query": query,
        "namespace": namespace,
        "limit": limit,
        "expand_entities": expand_entities,
    }).encode()
    req = urllib.request.Request(
        f"{SCRUTATOR}/v1/ltm/recall",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())
    data["_latency_ms"] = (time.time() - t0) * 1000
    return data


def normalise_path(p: str) -> str:
    return p.replace("\\", "/").lstrip("./").lstrip("/").lower()


def tail_matches(a: str, b: str) -> bool:
    a, b = normalise_path(a), normalise_path(b)
    if a == b:
        return True
    if a.endswith("/" + b):
        return True
    if b.endswith("/" + a):
        return True
    return False


def file_path_matches(retrieved: str, truth: str) -> bool:
    return tail_matches(retrieved, truth)


def snippet_jaccard(a: str, b: str, threshold: float = 0.8) -> bool:
    import re
    ta = set(re.findall(r"[\w]+", a.lower()))
    tb = set(re.findall(r"[\w]+", b.lower()))
    if not ta or not tb:
        return False
    # Substring containment first
    if a.lower() in b.lower() or b.lower() in a.lower():
        return True
    inter = len(ta & tb)
    union = len(ta) + len(tb) - inter
    return (inter / union) >= threshold if union > 0 else False


def source_matches(retrieved_file: str, retrieved_snippet: str,
                   truth_file: str, truth_snippet: str) -> bool:
    if file_path_matches(retrieved_file, truth_file):
        return True
    if retrieved_snippet.strip() and truth_snippet.strip():
        return snippet_jaccard(retrieved_snippet, truth_snippet)
    return False


def load_queries(corpus: str) -> list:
    queries = []
    for fname in ["factual.jsonl", "multi-hop.jsonl", "temporal.jsonl"]:
        path = Path(QUERIES_DIR) / fname
        for line in open(path):
            q = json.loads(line)
            if q.get("corpus") == corpus:
                queries.append(q)
    return queries


def per_query_metrics(query_id: str, retrieved: list, truth_sources: list) -> dict:
    def any_hit(rs):
        return any(
            source_matches(r["source_path"], r["content"], t["file"], t["snippet"])
            for r in rs
            for t in truth_sources
        )

    top1 = retrieved[:1]
    top5 = retrieved[:5]
    hits_in_top5 = sum(
        1 for r in top5
        if any(source_matches(r["source_path"], r["content"], t["file"], t["snippet"])
               for t in truth_sources)
    )

    return {
        "query_id": query_id,
        "recall_at_1": 1 if any_hit(top1) else 0,
        "recall_at_5": 1 if any_hit(top5) else 0,
        "precision_at_5": hits_in_top5 / len(top5) if top5 else 0,
        "num_truth_sources": len(truth_sources),
        "num_retrieved": len(retrieved),
        "hit_by": "path" if top5 and any(
            file_path_matches(r["source_path"], t["file"])
            for r in top5 for t in truth_sources
        ) else ("snippet" if any_hit(top5) else "none"),
    }


def main():
    namespace = DEFAULT_NS
    expand_entities = False
    with_meta_facts = False
    score_factor: float | None = None

    for i, arg in enumerate(sys.argv):
        if arg == "--namespace" and i + 1 < len(sys.argv):
            namespace = sys.argv[i + 1]
        elif arg == "--expand-entities":
            expand_entities = True
        elif arg == "--no-expand-entities":
            expand_entities = False
        elif arg == "--with-meta-facts":
            with_meta_facts = True
            expand_entities = True  # meta-facts enrichment requires entity expansion
        elif arg == "--score-factor" and i + 1 < len(sys.argv):
            score_factor = float(sys.argv[i + 1])

    queries = load_queries("datarim-kb")
    if with_meta_facts:
        mode = f"with-meta-facts-factor-{score_factor:.1f}" if score_factor is not None else "with-meta-facts"
    else:
        mode = "with-entities" if expand_entities else "no-entities"

    print(f"LTM-0009 Query Benchmark")
    print(f"  namespace={namespace} expand_entities={expand_entities}")
    print(f"  queries={len(queries)} mode={mode}")
    print()

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    per_query = []
    latencies = []

    for i, q in enumerate(queries):
        try:
            data = recall(q["question"], namespace, expand_entities)
            results = data.get("results", [])
            latency = data.get("_latency_ms", 0)
            latencies.append(latency)

            metrics = per_query_metrics(q["id"], results, q["ground_truth"]["sources"])
            per_query.append(metrics)

            status = "HIT" if metrics["recall_at_5"] > 0 else "miss"
            hit_by = metrics["hit_by"]
            print(f"  [{i+1}/{len(queries)}] {q['id']}: {status} (by={hit_by}) "
                  f"retrieved={metrics['num_retrieved']} latency={latency:.0f}ms")
        except Exception as e:
            print(f"  [{i+1}/{len(queries)}] {q['id']}: ERROR {e}")
            per_query.append({
                "query_id": q["id"],
                "recall_at_1": 0, "recall_at_5": 0, "precision_at_5": 0,
                "num_truth_sources": len(q["ground_truth"]["sources"]),
                "num_retrieved": 0, "hit_by": "none",
            })
            latencies.append(0)

    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    # Aggregate
    n = len(per_query)
    agg = {
        "count": n,
        "mean_recall_at_1": sum(m["recall_at_1"] for m in per_query) / n if n else 0,
        "mean_recall_at_5": sum(m["recall_at_5"] for m in per_query) / n if n else 0,
        "mean_precision_at_5": sum(m["precision_at_5"] for m in per_query) / n if n else 0,
    }

    # By class
    by_class = {}
    for cls in ["factual", "multi-hop", "temporal"]:
        cls_metrics = [m for m in per_query if m["query_id"].startswith(cls.split("-")[0])]
        cn = len(cls_metrics)
        by_class[cls] = {
            "count": cn,
            "mean_recall_at_1": sum(m["recall_at_1"] for m in cls_metrics) / cn if cn else 0,
            "mean_recall_at_5": sum(m["recall_at_5"] for m in cls_metrics) / cn if cn else 0,
            "mean_precision_at_5": sum(m["precision_at_5"] for m in cls_metrics) / cn if cn else 0,
        }

    # Latency
    sorted_lat = sorted(latencies)
    lat_stats = {
        "p50_ms": sorted_lat[len(sorted_lat)//2] if sorted_lat else 0,
        "p95_ms": sorted_lat[int(len(sorted_lat)*0.95)] if sorted_lat else 0,
        "min_ms": min(sorted_lat) if sorted_lat else 0,
        "max_ms": max(sorted_lat) if sorted_lat else 0,
        "mean_ms": sum(sorted_lat)/len(sorted_lat) if sorted_lat else 0,
        "count": len(sorted_lat),
    }

    # Hit mechanism breakdown
    path_hits = sum(1 for m in per_query if m["hit_by"] == "path")
    snippet_hits = sum(1 for m in per_query if m["hit_by"] == "snippet")
    misses = sum(1 for m in per_query if m["hit_by"] == "none")

    report = {
        "framework": "scrutator",
        "corpus": "datarim-kb",
        "run_date": time.strftime("%Y-%m-%d"),
        "run_started_at": started_at,
        "run_completed_at": completed_at,
        "llm_model": "cursor/auto (via MC)",
        "mode": mode,
        "expand_entities": expand_entities,
        "with_meta_facts": with_meta_facts,
        "score_factor": score_factor,
        "per_query": per_query,
        "per_query_latency_ms": latencies,
        "aggregate_all": agg,
        "aggregate_by_class": by_class,
        "latency": lat_stats,
        "hit_mechanism": {
            "path_hits": path_hits,
            "snippet_hits": snippet_hits,
            "misses": misses,
        },
    }

    # Save report — v5 dir for meta-facts sweep, v4 dir for legacy modes
    out_dir = Path(REPORTS_DIR_V5 if with_meta_facts else REPORTS_DIR_V4)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{time.strftime('%Y-%m-%d')}.datarim-kb.{mode}.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport: {out_file}")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: recall@1={agg['mean_recall_at_1']:.3f} recall@5={agg['mean_recall_at_5']:.3f} "
          f"precision@5={agg['mean_precision_at_5']:.3f}")
    print(f"  path_hits={path_hits} snippet_hits={snippet_hits} misses={misses}")
    print(f"  latency p50={lat_stats['p50_ms']:.0f}ms p95={lat_stats['p95_ms']:.0f}ms")
    print(f"  by_class: factual={by_class['factual']['mean_recall_at_5']:.3f} "
          f"multi-hop={by_class['multi-hop']['mean_recall_at_5']:.3f} "
          f"temporal={by_class['temporal']['mean_recall_at_5']:.3f}")


if __name__ == "__main__":
    main()
