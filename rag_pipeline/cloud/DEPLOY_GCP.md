# Deploying the Biosensor RAG assistant on GCP

This is the runbook for deploying on GCP. Everything below is **optional** — the pipeline runs locally for
free — but it's the path you'd take to ship the tool to a team.

There are two GCP integration points, usable independently:

| Concern            | Local (default, free)        | GCP (cloud-native)                          |
|--------------------|------------------------------|---------------------------------------------|
| Embeddings         | `all-MiniLM-L6-v2` (local)   | Vertex AI `text-embedding-004`              |
| Vector search      | FAISS / numpy (in-process)   | Vertex AI Vector Search **or** BigQuery `VECTOR_SEARCH` |
| Generation (LLM)   | extractive / Ollama          | Gemini on Vertex AI                         |
| Data warehouse     | local CSV/JSON               | BigQuery table `biosensor_rag.fact_cards`   |
| Serving            | Flask (`app.py`)             | Cloud Run (container)                       |
| CI/CD              | —                            | Cloud Build (`cloudbuild.yaml`)             |

---

## 0. One-time project setup
```bash
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=us-central1
gcloud config set project "$GOOGLE_CLOUD_PROJECT"
gcloud services enable aiplatform.googleapis.com bigquery.googleapis.com \
    run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud auth application-default login
```

## A. BigQuery: warehouse + serverless vector search
```bash
python -c "from cloud.gcp_bigquery import load_fact_cards_to_bq; load_fact_cards_to_bq()"
# one-time: create a BQ<->Vertex connection named bq_vertex_conn, then run setup_sql()
python -c "from cloud.gcp_bigquery import setup_sql, run_sql; run_sql(setup_sql())"
# query semantically, entirely in SQL:
python -c "from cloud.gcp_bigquery import bq_semantic_search; \
print(bq_semantic_search('wearable sensors that measure cortisol in sweat'))"
```
Analysts can also just `SELECT ... GROUP BY biomarker` — see `biomarker_rollup_sql()`.

## B. Vertex AI Vector Search (managed ANN) — reuse local embeddings
```bash
# create an empty stream-update index sized to your embedding dim (384 for MiniLM)
python -c "from cloud.gcp_vertex import create_index; \
print(create_index('biosensor-rag', 384))"
# deploy it to an index endpoint in the console / gcloud, then set:
export RAG_VERTEX_INDEX_ENDPOINT=projects/.../indexEndpoints/...
export RAG_VERTEX_DEPLOYED_INDEX_ID=biosensor_rag_deployed
# push the vectors you already built locally (no re-embedding cost):
python -c "from cloud.gcp_vertex import push_local_index_to_vertex as p; p('projects/.../indexes/...')"
```

## C. Serve on Cloud Run
```bash
# build the local index first so it's baked into the image (or rebuild in-container)
python build_index.py --embed-backend tfidf

# one command: build -> push -> deploy
gcloud builds submit --config cloud/cloudbuild.yaml \
    --substitutions=_REGION="$GOOGLE_CLOUD_LOCATION",_SERVICE=biosensor-rag .

# to run the cloud-native backends on the service:
gcloud run services update biosensor-rag --region "$GOOGLE_CLOUD_LOCATION" \
    --set-env-vars=RAG_EMBED_BACKEND=vertex,RAG_GEN_BACKEND=gemini
```

## Notes
- **Why managed vector search at scale:** the local FAISS/numpy index is perfect
  for the distilled corpus (~1k–10k docs). For the full ~14M-article corpus you
  move to Vertex Vector Search or BigQuery's ANN index so retrieval stays
  sub-second and you don't operate your own vector DB.
- **Cost control:** embeddings are computed once and reused (pushed to the cloud
  index), so switching to GCP doesn't re-incur embedding cost. Generation cost is
  metered per request (see the `usage` block every answer returns).
- **Portability (AWS/Azure):** the backend interfaces (`embeddings.py`,
  `vector_store.py`, `generator.py`) are provider-agnostic. The AWS analog is
  Bedrock embeddings + OpenSearch k-NN + Bedrock Claude; the Azure analog is
  Azure OpenAI + Azure AI Search. Same code, different `*_backend` modules.
- **Data privacy (public health):** PMIDs/abstracts are public, but the same
  pattern over PII would add Google DLP de-identification before indexing.
