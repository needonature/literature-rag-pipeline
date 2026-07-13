"""Evaluation / QA harness for the RAG pipeline.

Measures, over a gold question set (eval_questions.json):
  * retrieval_recall@k  - open-question check: did we retrieve a passage that
                          actually contains the answer evidence (in its text, or
                          via fact-card metadata)? No pre-distilled answer cards.
  * answer_keyword_hit  - does the generated answer mention the expected term?
  * citation_faithful   - does every cited tag actually come from a retrieved
                          passage (i.e. no hallucinated citations)?
  * latency / est. cost - aggregated, for performance & cost monitoring.

    python evaluate.py                 # uses the configured backends
    python evaluate.py --k 8 --gen extractive
    python evaluate.py --json report.json

This is intentionally backend-agnostic: run it on the free extractive backend in
CI, and re-run against Claude/OpenAI/Gemini before shipping a model change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import config
from rag import RagPipeline
from schema import RagAnswer

GOLD_PATH = config.BASE_DIR / "eval_questions.json"


def _retrieval_hit(
    result: RagAnswer, expect_meta: Dict[str, str], keywords: List[str]
) -> bool:
    """Open-question retrieval check: did we surface a passage that actually
    contains the answer evidence?

    Works for BOTH corpora:
      * fact-card corpus  -> matches the structured metadata, OR
      * raw-abstract corpus (open Q&A) -> the evidence terms appear in the
        passage TEXT, so the system has to genuinely retrieve the right paper
        rather than fetch a pre-distilled answer card.
    Evidence = the expected metadata values (e.g. "glucose" + "tears"), falling
    back to the expected keywords when no metadata is given.
    """
    evidence = [str(v).lower() for v in expect_meta.values() if v] or [
        k.lower() for k in keywords
    ]
    for c in result.contexts:
        # (a) structured metadata match (fact-card corpus)
        if expect_meta and all(
            str(want).lower() in str(c.metadata.get(key) or "").lower()
            for key, want in expect_meta.items()
        ):
            return True
        # (b) evidence present in passage text (abstract / open-question corpus)
        text = (c.text or "").lower()
        if evidence and all(term in text for term in evidence):
            return True
    return False


def _pmid_match(result: RagAnswer, expect_pmids: List[str]) -> bool:
    if not expect_pmids:
        return True
    got = {str(c.metadata.get("pmid")) for c in result.contexts if c.metadata.get("pmid")}
    return any(str(p) in got for p in expect_pmids)


def _keyword_hit(result: RagAnswer, keywords: List[str]) -> bool:
    if not keywords:
        return True
    ans = result.answer.lower()
    return any(kw.lower() in ans for kw in keywords)


def _citation_faithful(result: RagAnswer) -> bool:
    retrieved_tags = {c.citation for c in result.contexts}
    for tag in result.citations:
        if tag not in retrieved_tags:
            return False
    return True


def run(k: int, gen_backend: str, gold_path: Path) -> Dict[str, Any]:
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    pipe = RagPipeline.from_index(gen_backend=gen_backend or None)

    rows: List[Dict[str, Any]] = []
    agg = {
        "retrieval_recall": 0,
        "pmid_recall": 0,
        "answer_keyword_hit": 0,
        "citation_faithful": 0,
        "latency_ms": 0.0,
        "est_cost_usd": 0.0,
    }
    for q in gold:
        res = pipe.answer(q["question"], k=k)
        ret = _retrieval_hit(res, q.get("expect_metadata", {}), q.get("expect_keywords", []))
        pmid = _pmid_match(res, q.get("expect_pmids", []))
        kw = _keyword_hit(res, q.get("expect_keywords", []))
        faith = _citation_faithful(res)
        lat = res.usage.get("retrieval_ms", 0.0) + res.usage.get("generation_ms", 0.0)
        cost = res.usage.get("est_cost_usd", 0.0)

        agg["retrieval_recall"] += int(ret)
        agg["pmid_recall"] += int(pmid)
        agg["answer_keyword_hit"] += int(kw)
        agg["citation_faithful"] += int(faith)
        agg["latency_ms"] += lat
        agg["est_cost_usd"] += cost

        rows.append(
            {
                "question": q["question"],
                "retrieval_hit": ret,
                "answer_keyword_hit": kw,
                "citation_faithful": faith,
                "n_contexts": len(res.contexts),
                "top_citation": res.citations[0] if res.citations else None,
                "latency_ms": round(lat, 1),
            }
        )

    n = max(1, len(gold))
    report = {
        "n_questions": len(gold),
        "k": k,
        "embed_backend": pipe.retriever.embedder.name,
        "vector_backend": pipe.retriever.store.backend,
        "gen_backend": pipe.generator.name,
        "metrics": {
            "retrieval_recall@k": round(agg["retrieval_recall"] / n, 3),
            "pmid_recall@k": round(agg["pmid_recall"] / n, 3),
            "answer_keyword_hit": round(agg["answer_keyword_hit"] / n, 3),
            "citation_faithfulness": round(agg["citation_faithful"] / n, 3),
            "avg_latency_ms": round(agg["latency_ms"] / n, 1),
            "total_est_cost_usd": round(agg["est_cost_usd"], 6),
        },
        "per_question": rows,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the RAG pipeline against a gold question set.")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--gen", default=None, help="generation backend to evaluate")
    ap.add_argument("--gold", default=str(GOLD_PATH))
    ap.add_argument("--json", default=None, help="optional path to write the full JSON report")
    args = ap.parse_args()

    report = run(args.k, args.gen, Path(args.gold))

    print("=" * 70)
    print("RAG EVALUATION REPORT")
    print("=" * 70)
    print(f"questions={report['n_questions']}  k={report['k']}")
    print(f"embed={report['embed_backend']}  vector={report['vector_backend']}  gen={report['gen_backend']}")
    print("-" * 70)
    for key, val in report["metrics"].items():
        print(f"  {key:<24} {val}")
    print("-" * 70)
    for r in report["per_question"]:
        flag = "OK " if r["retrieval_hit"] else "MISS"
        print(f"  [{flag}] kw={int(r['answer_keyword_hit'])} cite={int(r['citation_faithful'])}  {r['question'][:60]}")
    print("=" * 70)

    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"full report -> {args.json}")


if __name__ == "__main__":
    main()
