"""Load / performance test for the RAG pipeline.

Fires N concurrent requests through one shared pipeline and reports throughput
(QPS) + latency percentiles (p50/p95/p99) with a retrieval-vs-generation
breakdown and per-request cost. This is the "...test LLM outputs for ...
performance" / "monitored cost/throughput" loop, made concrete.

    python loadtest.py                         # 60 reqs, concurrency 8, $0 extractive
    python loadtest.py --n 200 --concurrency 16
    python loadtest.py --gen ollama            # measure a real LLM backend's throughput

Concurrency is real for IO-bound backends (cloud/Ollama LLMs); for the local
extractive path numpy releases the GIL so it still parallelizes the vector math.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import config
from rag import RagPipeline

GOLD = config.BASE_DIR / "eval_questions.json"


def _questions(n: int) -> List[str]:
    try:
        qs = [q["question"] for q in json.loads(Path(GOLD).read_text(encoding="utf-8"))]
    except Exception:  # noqa: BLE001
        qs = []
    if not qs:
        qs = ["Which wearable sensors measure cortisol in sweat?"]
    return [qs[i % len(qs)] for i in range(n)]


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[i]


def run(n: int, concurrency: int, k: int, gen_backend: Optional[str]) -> dict:
    pipe = RagPipeline.from_index(gen_backend=gen_backend or None)
    questions = _questions(n)
    lat: List[float] = []
    retr: List[float] = []
    gen: List[float] = []
    cost = 0.0
    errors = 0

    def one(q: str):
        t0 = time.perf_counter()
        res = pipe.answer(q, k=k)
        return (time.perf_counter() - t0) * 1000.0, res.usage

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(one, q) for q in questions]
        for f in as_completed(futures):
            try:
                ms, usage = f.result()
                lat.append(ms)
                retr.append(usage.get("retrieval_ms", 0.0) or 0.0)
                gen.append(usage.get("generation_ms", 0.0) or 0.0)
                cost += usage.get("est_cost_usd", 0.0) or 0.0
            except Exception:  # noqa: BLE001
                errors += 1
    wall = time.perf_counter() - t_start
    return {
        "n": n, "concurrency": concurrency, "k": k, "wall_s": wall, "errors": errors,
        "qps": (n / wall) if wall else 0.0, "lat": lat, "retr": retr, "gen": gen, "cost": cost,
        "embed": pipe.retriever.embedder.name, "vec": pipe.retriever.store.backend,
        "genname": pipe.generator.name,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Throughput + latency load test for the RAG pipeline.")
    ap.add_argument("--n", type=int, default=60, help="total requests")
    ap.add_argument("--concurrency", type=int, default=8, help="concurrent workers")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--gen", default=None, help="generation backend (extractive | ollama | claude | ...)")
    args = ap.parse_args()

    r = run(args.n, args.concurrency, args.k, args.gen)
    lat = r["lat"]
    mean = sum(lat) / len(lat) if lat else 0.0
    print("=" * 66)
    print("RAG LOAD TEST")
    print("=" * 66)
    print(f"requests={r['n']}  concurrency={r['concurrency']}  k={r['k']}  errors={r['errors']}")
    print(f"embed={r['embed']}  vector={r['vec']}  gen={r['genname']}")
    print("-" * 66)
    print(f"throughput   : {r['qps']:.1f} req/s   (wall {r['wall_s']:.2f}s)")
    print(
        f"latency  ms  : mean {mean:.1f}   p50 {_pct(lat, 50):.1f}   "
        f"p95 {_pct(lat, 95):.1f}   p99 {_pct(lat, 99):.1f}   max {(max(lat) if lat else 0):.1f}"
    )
    rmean = sum(r["retr"]) / len(r["retr"]) if r["retr"] else 0.0
    gmean = sum(r["gen"]) / len(r["gen"]) if r["gen"] else 0.0
    print(f"  breakdown  : retrieval mean {rmean:.1f}ms   generation mean {gmean:.1f}ms")
    print(f"est cost     : ${r['cost']:.6f} total   (${(r['cost'] / r['n'] if r['n'] else 0):.6f}/req)")
    print("=" * 66)


if __name__ == "__main__":
    main()
