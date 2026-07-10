#!/usr/bin/env python3
import json, time, urllib.request, statistics, sys

ENDPOINT = "http://100.70.137.104:8310/v1/search"
GOLDEN = "golden-arcanada-v0.jsonl"
NAMESPACE = "arcanada"

def search(query, limit, rerank=None):
    body = {"query": query, "namespace": NAMESPACE, "limit": limit, "min_score": 0.0, "include_content": False}
    if rerank is not None:
        body["rerank"] = rerank
    data = json.dumps(body).encode()
    req = urllib.request.Request(ENDPOINT, data=data, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as resp:
        out = json.loads(resp.read())
    dt = (time.time() - t0) * 1000
    return out, dt

def tail_hit(gold_paths, returned_paths):
    for g in gold_paths:
        for r in returned_paths:
            if r == g or r.endswith("/" + g) or g.endswith("/" + r):
                return True
    return False

def main():
    rows = [json.loads(l) for l in open(GOLDEN) if l.strip()]
    results = []
    for row in rows:
        qid, cls, query, gold = row["id"], row["class"], row["query"], row["gold_source_paths"]

        out5, lat5 = search(query, 5)
        paths5 = [r["source_path"] for r in out5.get("results", [])]
        hit5 = tail_hit(gold, paths5)

        out1, lat1 = search(query, 1)
        paths1 = [r["source_path"] for r in out1.get("results", [])]
        hit1 = tail_hit(gold, paths1)

        results.append({
            "id": qid, "class": cls, "query": query, "gold": gold,
            "returned_top5": paths5, "hit@5": hit5, "hit@1": hit1,
            "latency_ms_top5": round(lat5, 1),
        })
        print(f"{qid:5s} [{cls:10s}] hit@1={str(hit1):5s} hit@5={str(hit5):5s} lat={lat5:.0f}ms  {query[:60]}")

    # Aggregate
    by_class = {}
    for r in results:
        by_class.setdefault(r["class"], []).append(r)

    print("\n=== Per-class recall ===")
    summary = {}
    for cls, rs in by_class.items():
        n = len(rs)
        r5 = sum(1 for r in rs if r["hit@5"]) / n
        r1 = sum(1 for r in rs if r["hit@1"]) / n
        lat_p50 = statistics.median(r["latency_ms_top5"] for r in rs)
        summary[cls] = {"n": n, "recall@1": r1, "recall@5": r5, "latency_p50_ms": lat_p50}
        print(f"{cls:12s} N={n:3d}  recall@1={r1:.3f}  recall@5={r5:.3f}  latency_p50={lat_p50:.0f}ms")

    n_all = len(results)
    r5_all = sum(1 for r in results if r["hit@5"]) / n_all
    r1_all = sum(1 for r in results if r["hit@1"]) / n_all
    lat_all = statistics.median(r["latency_ms_top5"] for r in results)
    print(f"{'OVERALL':12s} N={n_all:3d}  recall@1={r1_all:.3f}  recall@5={r5_all:.3f}  latency_p50={lat_all:.0f}ms")
    summary["overall"] = {"n": n_all, "recall@1": r1_all, "recall@5": r5_all, "latency_p50_ms": lat_all}

    with open("results-detail.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open("results-summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
