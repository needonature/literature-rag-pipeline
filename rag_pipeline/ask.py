"""Ask the RAG assistant a question from the terminal.

    python ask.py "Which wearable sensors measure cortisol in sweat?"
    python ask.py --k 8 --gen extractive "Non-invasive glucose monitoring in tears?"
    python ask.py --filter biofluid=Sweat "What biomarkers are measured?"
    RAG_GEN_BACKEND=ollama python ask.py "..."     # real LLM generation, still free

Defaults are zero-cost (local embeddings + extractive generation).
"""
from __future__ import annotations

import argparse

from rag import RagPipeline, pretty_print


def parse_filters(items):
    filters = {}
    for it in items or []:
        if "=" in it:
            key, val = it.split("=", 1)
            filters[key.strip()] = val.strip()
    return filters or None


def main() -> None:
    ap = argparse.ArgumentParser(description="Query the biosensor-literature RAG assistant.")
    ap.add_argument("question", nargs="+", help="your question")
    ap.add_argument("--k", type=int, default=None, help="number of passages to retrieve")
    ap.add_argument("--gen", default=None, help="generation backend: extractive|ollama|claude|openai|gemini")
    ap.add_argument("--filter", action="append", default=[], help="metadata filter, e.g. biofluid=Sweat")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = ap.parse_args()

    question = " ".join(args.question)
    pipe = RagPipeline.from_index(gen_backend=args.gen)
    result = pipe.answer(question, k=args.k, filters=parse_filters(args.filter))

    if args.json:
        import json

        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        pretty_print(result)


if __name__ == "__main__":
    main()
