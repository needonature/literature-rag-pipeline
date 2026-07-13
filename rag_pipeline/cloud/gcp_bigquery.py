"""GCP path #2 — BigQuery as the data warehouse + serverless vector search.

Why this exists: the distilled Biomarker<->Biosensor table is exactly the kind of
structured asset an analytics team wants in a warehouse — queryable with plain
SQL by analysts, and (newer) searchable semantically *inside* BigQuery via
ML.GENERATE_EMBEDDING + VECTOR_SEARCH, with no separate vector DB to operate.

Functions are import-safe; they only call GCP when invoked.
    load_fact_cards_to_bq()      -> create dataset+table, load the JSON rows
    run_sql(sql)                 -> execute arbitrary SQL (helper)
    semantic_search_sql(...)     -> the VECTOR_SEARCH query string
    bq_semantic_search(question) -> end-to-end serverless retrieval in SQL
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from ingest import load_fact_card_documents


def _client():
    if not config.GCP_PROJECT:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set; configure GCP before using BigQuery helpers.")
    from google.cloud import bigquery

    return bigquery.Client(project=config.GCP_PROJECT)


# --------------------------------------------------------------------------- #
# 1) Load the structured fact cards into BigQuery
# --------------------------------------------------------------------------- #
def load_fact_cards_to_bq(dataset: Optional[str] = None, table: Optional[str] = None) -> str:
    from google.cloud import bigquery

    client = _client()
    dataset = dataset or config.BQ_DATASET
    table = table or config.BQ_TABLE
    ds_ref = bigquery.Dataset(f"{config.GCP_PROJECT}.{dataset}")
    ds_ref.location = config.GCP_LOCATION
    client.create_dataset(ds_ref, exists_ok=True)

    rows: List[Dict[str, Any]] = []
    for d in load_fact_card_documents():
        m = d.metadata
        rows.append(
            {
                "doc_id": d.doc_id,
                "text": d.text,
                "pmid": m.get("pmid"),
                "biomarker": m.get("biomarker"),
                "biofluid": m.get("biofluid"),
                "biosensor_principle": m.get("biosensor_principle"),
                "application": m.get("application"),
                "experiment_type": m.get("experiment_type"),
            }
        )

    table_id = f"{config.GCP_PROJECT}.{dataset}.{table}"
    schema = [
        bigquery.SchemaField("doc_id", "STRING"),
        bigquery.SchemaField("text", "STRING"),
        bigquery.SchemaField("pmid", "STRING"),
        bigquery.SchemaField("biomarker", "STRING"),
        bigquery.SchemaField("biofluid", "STRING"),
        bigquery.SchemaField("biosensor_principle", "STRING"),
        bigquery.SchemaField("application", "STRING"),
        bigquery.SchemaField("experiment_type", "STRING"),
    ]
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")
    client.load_table_from_json(rows, table_id, job_config=job_config).result()
    print(f"[bq] loaded {len(rows)} rows into {table_id}")
    return table_id


def run_sql(sql: str):
    client = _client()
    return list(client.query(sql).result())


# --------------------------------------------------------------------------- #
# 2) One-time setup SQL: a remote embedding model + an embeddings table + index
#    (run these once via `bq query` or run_sql; shown here for reference)
# --------------------------------------------------------------------------- #
def setup_sql() -> str:
    p, ds, tbl = config.GCP_PROJECT, config.BQ_DATASET, config.BQ_TABLE
    return f"""
-- a) Register a remote embedding model backed by Vertex AI (needs a BQ connection):
CREATE OR REPLACE MODEL `{p}.{ds}.embed_model`
  REMOTE WITH CONNECTION `{config.GCP_LOCATION}.bq_vertex_conn`
  OPTIONS (ENDPOINT = 'text-embedding-004');

-- b) Materialize embeddings for every fact card:
CREATE OR REPLACE TABLE `{p}.{ds}.{tbl}_emb` AS
SELECT doc_id, text, pmid, biomarker, biofluid, biosensor_principle, application,
       ml_generate_embedding_result AS embedding
FROM ML.GENERATE_EMBEDDING(
       MODEL `{p}.{ds}.embed_model`,
       (SELECT doc_id, text, pmid, biomarker, biofluid, biosensor_principle, application
          FROM `{p}.{ds}.{tbl}`),
       STRUCT(TRUE AS flatten_json_output));

-- c) (optional, for scale) build an ANN vector index:
CREATE OR REPLACE VECTOR INDEX `{tbl}_idx`
  ON `{p}.{ds}.{tbl}_emb`(embedding)
  OPTIONS (index_type = 'IVF', distance_type = 'COSINE');
""".strip()


# --------------------------------------------------------------------------- #
# 3) Serverless semantic retrieval entirely in SQL
# --------------------------------------------------------------------------- #
def semantic_search_sql(question: str, k: int = 6) -> str:
    p, ds, tbl = config.GCP_PROJECT, config.BQ_DATASET, config.BQ_TABLE
    q = question.replace("'", "\\'")
    return f"""
SELECT base.doc_id, base.pmid, base.biomarker, base.biofluid,
       base.biosensor_principle, base.text, distance
FROM VECTOR_SEARCH(
       TABLE `{p}.{ds}.{tbl}_emb`, 'embedding',
       (SELECT ml_generate_embedding_result AS embedding
          FROM ML.GENERATE_EMBEDDING(
                 MODEL `{p}.{ds}.embed_model`,
                 (SELECT '{q}' AS content),
                 STRUCT(TRUE AS flatten_json_output))),
       top_k => {k}, distance_type => 'COSINE');
""".strip()


def bq_semantic_search(question: str, k: int = 6) -> List[Dict[str, Any]]:
    """Run retrieval inside BigQuery and return rows (no local vector DB needed)."""
    rows = run_sql(semantic_search_sql(question, k))
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Plain-SQL analyst view (no ML) — the "replace manual processes" angle
# --------------------------------------------------------------------------- #
def biomarker_rollup_sql() -> str:
    p, ds, tbl = config.GCP_PROJECT, config.BQ_DATASET, config.BQ_TABLE
    return f"""
SELECT biomarker, biofluid, COUNT(*) AS n_studies,
       ARRAY_AGG(DISTINCT biosensor_principle IGNORE NULLS LIMIT 5) AS example_sensors
FROM `{p}.{ds}.{tbl}`
WHERE biomarker IS NOT NULL
GROUP BY biomarker, biofluid
ORDER BY n_studies DESC
LIMIT 50;
""".strip()


if __name__ == "__main__":
    print("gcp_bigquery.py import OK. Example SQL it would run:\n")
    print(setup_sql())
    print("\n--- semantic search ---\n")
    print(semantic_search_sql("wearable sensors that measure cortisol in sweat"))
