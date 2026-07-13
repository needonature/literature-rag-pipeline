"""Central configuration for the Wearable Biosensor Literature RAG assistant.

Every knob is overridable via an environment variable so the same code runs
unchanged on a laptop (local, zero-cost) or on GCP (Vertex AI + Gemini).
Nothing here triggers a paid API call; backends are only constructed lazily.
"""
from __future__ import annotations

import os
from pathlib import Path

# Workaround for the common macOS/Anaconda + PyTorch duplicate-OpenMP abort
# ("libiomp5.dylib already initialized"). Set before torch is imported anywhere.
# For a production image you'd instead ensure a single OpenMP runtime is linked.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent                 # .../rag_pipeline
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", BASE_DIR.parent))  # the pipeline folder (existing outputs)
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", BASE_DIR / "index"))

# --------------------------------------------------------------------------- #
# Knowledge sources — these are REUSED from the existing ~$100 OpenAI Batch run.
#   * ABSTRACTS_CSV   : clean PMID/Title/Abstract/Keywords rows (unstructured text)
#   * STRUCTURED_JSONS: the distilled Biomarker<->Biosensor extraction results
# Both are folded into one corpus so retrieval can hit either rich abstracts or
# the curated "fact cards" that the extraction pipeline already produced.
# --------------------------------------------------------------------------- #
# Abstract-only corpus (no fact cards) so the eval can't "cheat" by retrieving a
# pre-distilled answer card. These two files are the raw PubMed abstracts of the
# NER-confirmed biosensor papers, both time windows (2014-2019 + 2021plus), each
# carrying its PMID. Comma-separated; build_corpus splits and concatenates them.
ABSTRACTS_CSV = os.environ.get(
    "RAG_ABSTRACTS_CSV",
    "biosensor_ner_2014to2019_A_P_T_A.csv,biosensor_ner_2021plus_A_P_T_A.csv",
)
STRUCTURED_JSONS = [
    "combined_output_2014to2025.json",
    "combined_output_2021plus.json",
    "combined_output_real_updated.json",
]

# --------------------------------------------------------------------------- #
# Embedding backend:  tfidf (default, free, offline) | local | openai | vertex
#   - "tfidf"  : TF-IDF + LSA dense embeddings via scikit-learn. Zero downloads,
#                no torch, runs anywhere instantly. Chosen as the local default
#                because it is rock-solid (the neural path can hard-abort on
#                Anaconda+MKL builds due to a duplicate-OpenMP conflict).
#   - "local"  : neural sentence-transformer (all-MiniLM-L6-v2) via transformers.
#                Best semantics; used in the Docker image / clean envs / cloud.
#                On Anaconda+MKL run with KMP_DUPLICATE_LIB_OK=TRUE.
#   - "openai" / "vertex" : cloud embeddings (opt-in, keyed).
# --------------------------------------------------------------------------- #
EMBED_BACKEND = os.environ.get("RAG_EMBED_BACKEND", "tfidf")
LOCAL_EMBED_MODEL = os.environ.get("RAG_LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_EMBED_MODEL = os.environ.get("RAG_OPENAI_EMBED_MODEL", "text-embedding-3-small")
# text-embedding-004 was shut down 2026-01-14; gemini-embedding-001 is the successor.
# It defaults to 3072-dim but supports Matryoshka truncation to 768/1536/3072.
VERTEX_EMBED_MODEL = os.environ.get("RAG_VERTEX_EMBED_MODEL", "gemini-embedding-001")
VERTEX_EMBED_DIM = int(os.environ.get("RAG_VERTEX_EMBED_DIM", "768"))  # 768|1536|3072

# --------------------------------------------------------------------------- #
# Generation backend:  extractive (default, free) | ollama | claude | openai | gemini
#   - "extractive"  : no LLM, no cost; stitches a grounded answer from retrieved
#                     sentences. Keeps the whole pipeline runnable for $0 / in CI.
#   - "ollama"      : local LLM (free) for real generation if you have Ollama.
#   - claude/openai/gemini : real cloud LLMs, enabled only when a key is present.
# --------------------------------------------------------------------------- #
GEN_BACKEND = os.environ.get("RAG_GEN_BACKEND", "extractive")
CLAUDE_MODEL = os.environ.get("RAG_CLAUDE_MODEL", "claude-sonnet-4-6")
OPENAI_GEN_MODEL = os.environ.get("RAG_OPENAI_GEN_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.environ.get("RAG_OLLAMA_MODEL", "llama3.1:8b")
GEMINI_MODEL = os.environ.get("RAG_GEMINI_MODEL", "gemini-2.5-flash")  # 1.5 retired on Vertex; 2.5-flash current

# --------------------------------------------------------------------------- #
# Retrieval / prompting
# --------------------------------------------------------------------------- #
TOP_K = int(os.environ.get("RAG_TOP_K", "6"))
# Weight on semantic (vector) score vs. lexical (keyword) score in hybrid search.
HYBRID_ALPHA = float(os.environ.get("RAG_HYBRID_ALPHA", "0.7"))
MAX_CONTEXT_CHARS = int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "6000"))
CHUNK_MAX_WORDS = int(os.environ.get("RAG_CHUNK_MAX_WORDS", "220"))
CHUNK_OVERLAP_WORDS = int(os.environ.get("RAG_CHUNK_OVERLAP_WORDS", "40"))

# --------------------------------------------------------------------------- #
# GCP (cloud-native path). Only read when EMBED_BACKEND/GEN_BACKEND == vertex/gemini
# or when running the cloud/ helpers.
# --------------------------------------------------------------------------- #
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
BQ_DATASET = os.environ.get("RAG_BQ_DATASET", "biosensor_rag")
BQ_TABLE = os.environ.get("RAG_BQ_TABLE", "fact_cards")
VERTEX_INDEX_ENDPOINT = os.environ.get("RAG_VERTEX_INDEX_ENDPOINT", "")
VERTEX_DEPLOYED_INDEX_ID = os.environ.get("RAG_VERTEX_DEPLOYED_INDEX_ID", "")

# --------------------------------------------------------------------------- #
# AWS (cloud-native path; mirror of the GCP block). Read only when
# EMBED_BACKEND/GEN_BACKEND == bedrock or when running the cloud/aws_* helpers.
# --------------------------------------------------------------------------- #
AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
BEDROCK_EMBED_MODEL = os.environ.get("RAG_BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
BEDROCK_EMBED_DIM = int(os.environ.get("RAG_BEDROCK_EMBED_DIM", "1024"))
# Any Claude model id you've enabled in the Bedrock console works here.
BEDROCK_GEN_MODEL = os.environ.get("RAG_BEDROCK_GEN_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")
# pgvector (open-source vector DB): local Docker for dev, Amazon RDS / Aurora in prod.
PG_DSN = os.environ.get("RAG_PG_DSN", os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag"))
PG_TABLE = os.environ.get("RAG_PG_TABLE", "biosensor_chunks")
# Amazon OpenSearch (cloud-native managed k-NN). Service "aoss" (serverless) or "es".
OPENSEARCH_HOST = os.environ.get("RAG_OPENSEARCH_HOST", "")
OPENSEARCH_INDEX = os.environ.get("RAG_OPENSEARCH_INDEX", "biosensor-rag")
OPENSEARCH_SERVICE = os.environ.get("RAG_OPENSEARCH_SERVICE", "aoss")

# --------------------------------------------------------------------------- #
# Approximate pricing (USD / 1M tokens) — for the cost meter only. Edit freely;
# used for *relative* cost monitoring in eval, never for billing.
# --------------------------------------------------------------------------- #
PRICING = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "claude-sonnet-4-6": {"in": 3.0, "out": 15.0},
    "claude-haiku-4-5-20251001": {"in": 1.0, "out": 5.0},
    "gemini-1.5-flash": {"in": 0.075, "out": 0.30},
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50},
    "text-embedding-3-small": {"in": 0.02, "out": 0.0},
    "text-embedding-004": {"in": 0.025, "out": 0.0},
    "gemini-embedding-001": {"in": 0.15, "out": 0.0},
    "amazon.titan-embed-text-v2:0": {"in": 0.02, "out": 0.0},
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {"in": 3.0, "out": 15.0},
    "_local": {"in": 0.0, "out": 0.0},
    "_extractive": {"in": 0.0, "out": 0.0},
}


def summary() -> str:
    return (
        f"DATA_DIR        = {DATA_DIR}\n"
        f"INDEX_DIR       = {INDEX_DIR}\n"
        f"EMBED_BACKEND   = {EMBED_BACKEND}\n"
        f"GEN_BACKEND     = {GEN_BACKEND}\n"
        f"TOP_K           = {TOP_K}  HYBRID_ALPHA={HYBRID_ALPHA}\n"
    )
