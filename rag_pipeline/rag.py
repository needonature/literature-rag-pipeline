"""RagPipeline — wire retrieval + generation into one auditable call.

    pipe = RagPipeline.from_index()           # loads vector index + embedder + LLM backend
    result = pipe.answer("Which wearable sensors measure cortisol in sweat?")
    print(result.answer); print(result.citations)

Every answer carries its retrieved evidence, the PMIDs it cites, and a usage
record (tokens / latency / estimated cost) so outputs are auditable and the cost
of each query is observable — the "build it, test it, watch the cost" discipline.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from generator import BaseGenerator, get_generator
from retriever import Retriever
from schema import RagAnswer


class RagPipeline:
    def __init__(self, retriever: Retriever, generator: BaseGenerator):
        self.retriever = retriever
        self.generator = generator

    @classmethod
    def from_index(
        cls,
        index_dir: Optional[Path] = None,
        gen_backend: Optional[str] = None,
    ) -> "RagPipeline":
        retriever = Retriever.from_index(index_dir)
        generator = get_generator(gen_backend)
        return cls(retriever, generator)

    def answer(
        self,
        question: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, str]] = None,
    ) -> RagAnswer:
        t0 = time.perf_counter()
        contexts = self.retriever.retrieve(question, k=k, filters=filters)
        t_retrieve = time.perf_counter() - t0

        if not contexts:
            return RagAnswer(
                question=question,
                answer="No relevant passages were found in the indexed corpus for this question.",
                citations=[],
                contexts=[],
                usage={"retrieval_ms": round(t_retrieve * 1000, 1), "generation_ms": 0.0, "n_contexts": 0},
            )

        t1 = time.perf_counter()
        gen = self.generator.generate(question, contexts)
        t_generate = time.perf_counter() - t1

        usage = dict(gen.get("usage", {}))
        usage.update(
            {
                "retrieval_ms": round(t_retrieve * 1000, 1),
                "generation_ms": round(t_generate * 1000, 1),
                "n_contexts": len(contexts),
                "embed_backend": self.retriever.embedder.name,
                "vector_backend": self.retriever.store.backend,
            }
        )
        return RagAnswer(
            question=question,
            answer=gen.get("answer", ""),
            citations=gen.get("citations", []),
            contexts=contexts,
            usage=usage,
        )


def pretty_print(result: RagAnswer) -> None:
    line = "=" * 78
    print(line)
    print(f"Q: {result.question}")
    print(line)
    print(result.answer)
    print()
    print("Citations:", ", ".join(result.citations) if result.citations else "(none)")
    print("-" * 78)
    print("Evidence (top retrieved):")
    for i, c in enumerate(result.contexts, 1):
        tag = c.citation
        meta_bits = []
        for key in ("biomarker", "biofluid", "biosensor_principle", "year", "journal"):
            if c.metadata.get(key):
                meta_bits.append(f"{key}={c.metadata[key]}")
        print(f"  {i}. [{tag}] score={c.score:.3f}  {'  '.join(meta_bits)}")
        snippet = c.text.replace("\n", " ")
        print(f"     {snippet[:160]}{'...' if len(snippet) > 160 else ''}")
    print("-" * 78)
    u = result.usage
    print(
        "Usage: "
        f"retrieval={u.get('retrieval_ms')}ms  generation={u.get('generation_ms')}ms  "
        f"backend={u.get('backend', self_backend(result))}  "
        f"in_tok={u.get('input_tokens','-')} out_tok={u.get('output_tokens','-')} "
        f"est_cost=${u.get('est_cost_usd', 0.0)}"
    )
    print(line)


def self_backend(result: RagAnswer) -> str:
    return result.usage.get("embed_backend", "?")
