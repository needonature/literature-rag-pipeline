"""Build the vector index from the EXISTING pipeline outputs (offline, $0 by default).

    python build_index.py                       # local embeddings, default sources
    python build_index.py --abstracts pubmed_extract_2014-2025_Full_with_Keywords_sample.csv
    python build_index.py --embed-backend vertex   # cloud embeddings (needs GCP creds)

Steps:  ingest -> chunk -> embed -> store -> persist  (all under rag_pipeline/index/)
Re-runnable and deterministic; safe to delete the index/ folder and rebuild.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import config
from embeddings import get_embedder
from ingest import build_corpus
from text_utils import chunk_document
from vector_store import get_store


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the biosensor-literature RAG index.")
    ap.add_argument("--abstracts", nargs="+", default=None, help="one or more abstracts CSV filenames (defaults to config)")
    ap.add_argument("--no-fact-cards", action="store_true", help="skip the distilled fact-card docs")
    ap.add_argument("--abstract-limit", type=int, default=None, help="cap #abstracts (for a quick demo)")
    ap.add_argument("--embed-backend", default=None, help="local | tfidf | openai | vertex")
    ap.add_argument("--index-dir", default=None)
    args = ap.parse_args()

    index_dir = Path(args.index_dir or config.INDEX_DIR)
    print(config.summary())

    # 1) ingest existing outputs ------------------------------------------------
    corpus = build_corpus(
        abstracts_csv=args.abstracts,
        include_fact_cards=not args.no_fact_cards,
        abstract_limit=args.abstract_limit,
    )
    if not corpus:
        raise SystemExit("No documents ingested — check DATA_DIR and source filenames in config.py.")

    # 2) chunk long docs --------------------------------------------------------
    chunks = []
    for doc in corpus:
        chunks.extend(chunk_document(doc, config.CHUNK_MAX_WORDS, config.CHUNK_OVERLAP_WORDS))
    print(f"[build] {len(corpus)} documents -> {len(chunks)} chunks")

    # 3) embed (fit first for TF-IDF; no-op for neural backends) -----------------
    embedder = get_embedder(args.embed_backend)
    texts = [c.text for c in chunks]
    t0 = time.perf_counter()
    embedder.fit(texts)
    embeddings = embedder.encode(texts, normalize=True)
    print(f"[build] embedded with {embedder.name}  dim={embedder.dim}  in {time.perf_counter()-t0:.1f}s")

    # 4) store + persist --------------------------------------------------------
    store = get_store()
    store.build(chunks, embeddings)
    store.save(index_dir)
    embedder.save(index_dir)
    print(f"[build] index saved to {index_dir}  (backend={store.backend})")
    print("[build] done. Try:  python ask.py \"Which wearable sensors measure cortisol in sweat?\"")


if __name__ == "__main__":
    main()
