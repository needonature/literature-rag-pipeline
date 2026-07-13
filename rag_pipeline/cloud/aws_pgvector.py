"""AWS path #2 — pgvector: open-source vector database, vector search in plain SQL.

This is the AWS analog of BigQuery VECTOR_SEARCH: the same Postgres + pgvector
code runs on a LOCAL Docker Postgres for dev and on Amazon RDS / Aurora
PostgreSQL in prod — just point RAG_PG_DSN at the managed instance. Retrieval is
ordinary SQL:

    SELECT ... ORDER BY embedding <=> :query_vec LIMIT k     -- cosine distance

Embeddings come from ANY project embedder: the default TF-IDF/LSA (offline, $0,
great for a local demo) or Amazon Bedrock Titan in prod (RAG_EMBED_BACKEND=bedrock).

Run it locally (no AWS needed) — see cloud/DEPLOY_AWS.md for the one-line Docker:
    docker run -d --name ragpg -p 5432:5432 \
      -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=rag pgvector/pgvector:pg16
    pip install -r requirements-aws.txt
    python cloud/aws_pgvector.py "wearable sensors that measure cortisol in sweat"
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import numpy as np

# Allow running directly as a script (python cloud/aws_pgvector.py ...) by putting
# the repo root (rag_pipeline/) on the path so `import config` etc. resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from embeddings import BaseEmbedder, get_embedder
from ingest import build_corpus
from text_utils import chunk_document


def _connect():
    import psycopg2
    from pgvector.psycopg2 import register_vector

    conn = psycopg2.connect(config.PG_DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def build_pgvector(limit: Optional[int] = None) -> BaseEmbedder:
    """Embed the abstract corpus and load (doc, vector) rows into Postgres+pgvector."""
    from psycopg2.extras import execute_values

    # Abstract-only corpus (matches the abstract-only eval index).
    docs = build_corpus(include_fact_cards=False, abstract_limit=limit)
    chunks = []
    for d in docs:
        chunks.extend(chunk_document(d, config.CHUNK_MAX_WORDS, config.CHUNK_OVERLAP_WORDS))

    embedder = get_embedder()
    texts = [c.text for c in chunks]
    embedder.fit(texts)
    X = embedder.encode(texts, normalize=True)
    dim = int(X.shape[1])

    tbl = config.PG_TABLE
    conn = _connect()
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {tbl}")
        cur.execute(
            f"""CREATE TABLE {tbl} (
                   id serial PRIMARY KEY, doc_id text, pmid text, title text,
                   biomarker text, biofluid text, body text, embedding vector({dim}))"""
        )
        rows = [
            (
                c.doc_id, c.metadata.get("pmid"), c.metadata.get("title"),
                c.metadata.get("biomarker"), c.metadata.get("biofluid"),
                c.text, X[i].tolist(),
            )
            for i, c in enumerate(chunks)
        ]
        execute_values(
            cur,
            f"INSERT INTO {tbl} (doc_id, pmid, title, biomarker, biofluid, body, embedding) VALUES %s",
            rows,
        )
        # Open-source ANN index (cosine). Brute-force is exact below this size anyway.
        cur.execute(f"CREATE INDEX ON {tbl} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")
    host = config.PG_DSN.split("@")[-1]
    print(f"[pgvector] loaded {len(rows)} chunks (dim={dim}, embedder={embedder.name}) into '{tbl}' @ {host}")
    return embedder


def search(question: str, embedder: BaseEmbedder, k: int = 6):
    """k-NN over pgvector, entirely in SQL (cosine distance operator <=>)."""
    qv = embedder.encode([question], normalize=True)[0].tolist()
    tbl = config.PG_TABLE
    conn = _connect()
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT pmid, title, biomarker, biofluid, left(body, 80) AS snippet,
                       1 - (embedding <=> %s::vector) AS cosine_sim
                FROM {tbl}
                ORDER BY embedding <=> %s::vector
                LIMIT %s""",
            (qv, qv, k),
        )
        return cur.fetchall()


def main() -> None:
    ap = argparse.ArgumentParser(description="pgvector RAG retrieval over the biosensor corpus.")
    ap.add_argument("question", nargs="?", default="wearable sensors that measure cortisol in sweat")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="cap #abstracts (quick demo)")
    args = ap.parse_args()

    embedder = build_pgvector(limit=args.limit)
    print(f"\nQ: {args.question}\n")
    for pmid, title, bm, bf, snippet, sim in search(args.question, embedder, args.k):
        label = title or snippet
        extra = f"  ({bm}/{bf})" if bm else ""
        print(f"  {sim:.3f}  [PMID:{pmid or 'n/a'}] {label[:72]}{extra}")


if __name__ == "__main__":
    main()
