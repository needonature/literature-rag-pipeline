# Literature RAG Pipeline

A retrieval-augmented generation (RAG) pipeline that answers natural-language
questions over a curated PubMed literature corpus, **grounded with PMID
citations on every claim**. The architecture is domain-agnostic; it is built
and evaluated here on a wearable-biosensor research corpus.

It is built **on top of** the existing extraction pipeline in this repo. That
pipeline already did the expensive part — a 3-stage funnel that distilled **~9M
PubMed title+abstracts (2014–2019, via SQLite)** into a structured
`Biomarker ↔ Biosensor` table, for **~$100 total**, with cost spent where it mattered:

| Stage | Method | Kept | Cost |
|---|---|---|---|
| 1. NER pre-filter | 6 biofluids + a metabolite-name dictionary | ~74,000 | **$0** |
| 2. Doc classification | GPT-4o — "is this a wearable/biosensor paper?" (cheap model, high volume) | ~400 | **$90** |
| 3. Structured extraction | o3 — 6 fields/paper (expensive model, tiny volume) | table | **$10** |

This project turns that curated knowledge into an **interactive, context-aware
internal tool**: ask a question, get a grounded, cited answer. No new API spend
is required — the default path runs entirely locally for **$0**.

> Honest framing: the original pipeline is
> *information extraction* (documents → structured table). RAG is the other
> direction (question → retrieve relevant documents → grounded answer). This repo
> adds the missing retrieval+generation layer and reuses the extraction output as
> the knowledge base, so together they form a complete, defensible RAG system.

---

## Architecture

```
 INGEST (build_index.py, offline, $0)                  QUERY (ask.py / app.py)
 ┌───────────────────────────────────────┐            ┌────────────────────────────────────┐
 │ existing outputs (reused, no re-spend) │            │ user question                       │
 │  • abstracts CSV  (PMID/Title/Abstract)│            │        │                            │
 │  • combined_output_*.json (fact cards) │            │        ▼                            │
 │            │                           │            │  embed query (same embedder)        │
 │            ▼                           │            │        │                            │
 │  Document[]  →  chunk (overlap)        │            │        ▼                            │
 │            │                           │            │  HYBRID retrieve  top-k             │
 │            ▼                           │            │   semantic (vector) + lexical       │
 │  embed  (TF-IDF/LSA | MiniLM |         │            │   + metadata filters                │
 │          OpenAI | Vertex)              │            │        │                            │
 │            │                           │            │        ▼                            │
 │            ▼                           │            │  build grounded prompt (+citations) │
 │  vector store (FAISS | numpy)  ────────┼───persist──┼──►     │                            │
 │            │                           │  index/    │        ▼                            │
 │            ▼                           │            │  generate (extractive | Ollama |    │
 │  index/  (embeddings + docs + meta)    │            │   Claude | OpenAI | Gemini)         │
 └───────────────────────────────────────┘            │        │                            │
                                                       │        ▼                            │
                                                       │  answer + PMID citations + evidence │
                                                       │  + usage (tokens / latency / cost)  │
                                                       └────────────────────────────────────┘
```

Every backend is **pluggable via environment variables**, so the same code runs
locally for free or on GCP (Vertex AI + Gemini + BigQuery) without edits.

---

## Quickstart (zero cost, ~2 minutes)

```bash
cd rag_pipeline
pip install -r requirements.txt            # core deps: numpy, scikit-learn, Flask (no torch needed)

# 1) Build the index. A tiny SYNTHETIC sample corpus is bundled (../sample_data/) so this
#    runs out of the box — no data setup, no downloads, no API calls:
python build_index.py --abstracts sample_data/abstracts_sample.csv --no-fact-cards
#   (default = TF-IDF/LSA embeddings, offline & instant. For neural embeddings add
#    RAG_EMBED_BACKEND=local ; on Anaconda+MKL prefix KMP_DUPLICATE_LIB_OK=TRUE.)

# 2) Ask a question from the terminal
python ask.py "Which wearable sensors measure cortisol in sweat?"
python ask.py --filter biofluid=Sweat "What biomarkers are measured in sweat?"

# 3) Run it as an internal web tool
python app.py                              # http://localhost:8080

# 4) Evaluate retrieval + answer quality
python evaluate.py
```

> **📦 Data note.** The full ~639-abstract PubMed corpus and the reported evaluation
> numbers are **not** reproducible from this repo alone — the raw abstracts are **not
> bundled** (publisher copyright + size). A small **synthetic** sample (`sample_data/`,
> author-written, illustrative) ships so the pipeline runs end to end. To reproduce the
> real results, place your own PubMed abstract CSVs at the repo root and point
> `RAG_ABSTRACTS_CSV` / `RAG_DATA_DIR` at them (see `rag_pipeline/config.py`).

**Want real LLM generation, still free?** Install [Ollama](https://ollama.com),
then `RAG_GEN_BACKEND=ollama python ask.py "..."`.
**Have an API key?** `RAG_GEN_BACKEND=claude` / `openai` / `gemini`.

---

## LangChain implementation (alternative)

`langchain_rag.py` is a second, independent implementation of the same RAG — same
corpus, same PMID-cited output — assembled from **off-the-shelf LangChain
components** instead of hand-rolled code. It shows framework fluency alongside the
from-scratch build; the two are not wired together.

LangChain pieces used: `RecursiveCharacterTextSplitter` (chunking),
`TFIDFRetriever` (default retrieval — scikit-learn, $0) or `Chroma` +
`OpenAIEmbeddings` (open-source vector DB, opt-in), `create_retrieval_chain` +
`create_stuff_documents_chain` (the RAG chain), and `ChatOpenAI` / `ChatAnthropic`
/ `ChatOllama` (pluggable LLM).

```bash
# core requirements + the two LangChain packages (default path needs no key, no torch)
pip install -r requirements.txt -r requirements-langchain.txt
python langchain_rag.py "Which wearable sensors measure cortisol in sweat?"
```

Env switches (all optional):

| Variable | Default | Other values |
|---|---|---|
| `LCRAG_RETRIEVER` | `tfidf` ($0, scikit-learn) | `chroma` (open-source vector DB; needs `OPENAI_API_KEY` + `langchain-chroma`) |
| `LCRAG_LLM` | `none` (returns cited evidence) | `openai` · `anthropic` · `ollama` |
| `LCRAG_TOP_K` | `6` | any integer |

The default (`tfidf` + `none`) runs offline at $0; set `LCRAG_LLM=openai` (etc.) for
a synthesized, PMID-cited answer.

---

## AWS path (mirror of the GCP path)

The same pluggable design runs on AWS — only the backend modules change. Full
runbook in `cloud/DEPLOY_AWS.md`.

- `cloud/aws_bedrock.py` — Amazon Bedrock: Titan Text Embeddings v2 + Claude
  generation (`RAG_EMBED_BACKEND=bedrock` / `RAG_GEN_BACKEND=bedrock`).
- `cloud/aws_pgvector.py` — **pgvector**, an open-source vector database; k-NN in
  plain SQL (`ORDER BY embedding <=> :q`). Local Docker Postgres for dev, Amazon
  RDS / Aurora in prod.
- `cloud/aws_opensearch.py` — Amazon OpenSearch Serverless k-NN (cloud-native
  managed vector search).

The pgvector path runs **fully locally at $0** (no AWS account) — an open-source
vector-DB demo you can show on a laptop:

```bash
docker run -d --name ragpg -p 5432:5432 \
  -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=rag pgvector/pgvector:pg16
pip install -r requirements.txt -r requirements-aws.txt
export RAG_PG_DSN="postgresql://rag:rag@localhost:5432/rag"
python cloud/aws_pgvector.py "wearable sensors that measure cortisol in sweat"
```

---

## MCP server (agent tooling)

`mcp_server.py` exposes the RAG as **Model Context Protocol** tools, so any MCP
client (Claude Desktop, an agent loop) can call it as a step in a multi-step
workflow — function-calling over your own corpus, via the open standard.

```bash
pip install -r requirements.txt -r requirements-mcp.txt   # Python >= 3.10
python build_index.py --no-fact-cards
python mcp_server.py        # stdio MCP server
```

Tools: `search_biosensor_literature(query, k)` (top-k abstracts + PMID citations +
scores) and `answer_biosensor_question(question, k)` (grounded, cited answer +
usage). Register under `mcpServers` in an MCP client config, e.g. Claude Desktop's
`claude_desktop_config.json`.

---

## The pieces

| File | Responsibility |
|---|---|
| `config.py` | one place for every knob; all overridable by env var |
| `schema.py` | `Document`, `RetrievedChunk`, `RagAnswer` dataclasses |
| `ingest.py` | reuse existing outputs → `Document[]` (abstracts + distilled fact cards) |
| `text_utils.py` | tokenization + overlapping chunking |
| `embeddings.py` | pluggable embedder: **TF-IDF/LSA** (default, offline) / local HF MiniLM / OpenAI / Vertex |
| `vector_store.py` | pluggable store: **FAISS** if present else **numpy**; save/load |
| `retriever.py` | **hybrid** semantic+lexical retrieval with metadata filters |
| `generator.py` | pluggable LLM: **extractive** (free) / Ollama / Claude / OpenAI / Gemini |
| `rag.py` | orchestration → `RagAnswer` with citations + usage (tokens/latency/cost) |
| `build_index.py` / `ask.py` / `app.py` | CLI build, CLI query, web tool |
| `evaluate.py` + `eval_questions.json` | QA harness + gold question set (accuracy / reliability) |
| `loadtest.py` | concurrent load test → throughput (QPS) + p50/p95/p99 latency + cost |
| `mcp_server.py` | RAG exposed as Model Context Protocol tools for agent/function-calling use |
| `cloud/` | GCP (`gcp_*`, `DEPLOY_GCP.md`): Vertex AI, BigQuery, Cloud Run · AWS (`aws_*`, `DEPLOY_AWS.md`): Bedrock, pgvector, OpenSearch |

### Design decisions worth defending
- **Hybrid retrieval** (`HYBRID_ALPHA`): dense vectors catch paraphrases; lexical
  overlap catches exact analyte names / IDs that embeddings smear. Blending both
  is more robust than either alone, and it's one tunable knob.
- **Two document types in one index:** rich *abstracts* for depth + distilled
  *fact cards* for precision. The fact cards are the reused $100 output, so the
  retriever can return a crisp grounded fact and its supporting abstract.
- **Pluggable embeddings, robust default:** the local default is TF-IDF/LSA
  (needs only scikit-learn — no torch, no model download, runs anywhere). Neural
  sentence-transformer (MiniLM) and cloud (Vertex `gemini-embedding-001`) embedders
  are drop-in upgrades selected by one env var — the Vertex path has been run live
  on GCP (see Evaluation).
- **$0 default, real LLM optional:** the extractive backend keeps everything free
  and CI-friendly; flip one env var for Ollama/Claude/OpenAI/Gemini generation.
- **Auditable by construction:** every answer ships its retrieved evidence, the
  PMIDs it cites, and a usage/cost record.

---

## Evaluation

`evaluate.py` scores the pipeline over a gold question set. Metrics:

- **retrieval_recall@k** — did we retrieve a passage that actually contains the
  answer evidence (expected biomarker/biofluid/principle in the passage text, or
  the structured metadata)? Generator-independent.
- **pmid_recall@k** — did the top-k include one of the **gold PMIDs** (the
  corpus papers that genuinely answer the question)? Paper-level, stricter.
- **answer_keyword_hit** — does the generated answer mention the expected term?
- **citation_faithfulness** — is every `[PMID:…]` the model **actually wrote**
  backed by a retrieved passage (i.e. no hallucinated citations)?
- **avg_latency_ms / total_est_cost_usd** — performance + cost monitoring.

Run it on the free `extractive` backend in CI, and re-run against a cloud LLM
before shipping a model change.

### Gold question sets

| File | What it tests |
|---|---|
| `eval_questions.json` | 60 keyword-dense questions ("measures *X* in *Y*") — favors lexical match |
| `eval_questions_paraphrase_60.json` | 60 paraphrased questions (synonyms only — "stress hormone"/"perspiration", never the literal *cortisol*/*sweat*) — favors semantic match |

Both carry corpus-grounded `expect_pmids` so `pmid_recall@k` is a real metric.

### Measured results

**Embedding ablation — `retrieval_recall@6`, same `extractive` generator, only the
embedder changed (run live on GCP Vertex):**

| Question set | TF-IDF/LSA | Vertex `gemini-embedding-001` (768-d) | Δ |
|---|---|---|---|
| keyword-dense (60) | 0.950 | 0.967 | **+1.7 pts** |
| paraphrase (60) | 0.717 | 0.817 | **+10.0 pts** |

Neural embeddings buy **semantic generalization**: a negligible gain when the
query shares words with the documents, a large gain when it doesn't. The cost is
latency — each query adds an embedding-API round-trip (~20 ms → ~130 ms).

**Full neural + LLM stack — paraphrase (60), Vertex `gemini-embedding-001` +
`gemini-2.5-flash`:**

| Metric | extractive (TF-IDF) | neural + Gemini |
|---|---|---|
| retrieval_recall@6 | 0.717 | 0.817 |
| pmid_recall@6 | — | 0.833 |
| answer_keyword_hit | 0.500 | 0.817 |
| citation_faithfulness | 1.0 | 1.0 (earned: 57/60 cite real PMIDs, 0 hallucinated) |
| latency / cost (60 q) | 21 ms · $0 | 4.4 s · $0.040 |

### Eval hygiene: two vacuous metrics, found and fixed

A good eval has to be honest about itself. Two metrics were structurally pinned
at `1.0` and have been fixed:

- **`pmid_recall@k`** was `1.0` because no question had `expect_pmids`, so the
  check auto-passed. Fixed by populating corpus-grounded gold PMIDs (corpus
  abstracts containing all evidence terms ∪ matching fact-card records, ∩ the
  corpus). It now reports a real **0.833** and drops when retrieval misses the
  right paper.
- **`citation_faithfulness`** was `1.0` because the LLM backends reported the
  *retrieved* PMIDs as the citations — trivially a subset of themselves. Fixed by
  parsing the `[PMID:…]` tags the model **actually wrote** (`parse_cited_tags`).
  It now genuinely catches hallucinated citations (verified against an injected
  fake `PMID:99999999`); the surviving `1.0` is earned, not structural.

> Honest caveats to keep: the lexical score is token-set overlap, not BM25; there
> is no neural reranker yet; "faithfulness" checks citation *provenance* (the cited
> PMID was retrieved), not semantic entailment of the claim; broad biofluid-only
> questions have large gold sets, so their `pmid_recall` is easy.

This is the "test LLM outputs for accuracy, reliability, and performance" loop,
made concrete — including being skeptical of the test itself.

---

## Cost / "no new API spend"

- **Indexing:** local embeddings (or TF-IDF) → **$0**.
- **Retrieval:** local vector math → **$0**.
- **Generation:** `extractive` or `ollama` → **$0**. Cloud LLMs are opt-in and
  metered (every answer reports `est_cost_usd`).
- The **$100 already spent** is reused as the knowledge base — never re-run.

