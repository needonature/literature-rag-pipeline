#!/usr/bin/env bash
# Live-GCP run for the literature RAG over BigQuery + Vertex embeddings.
# Runs the cloud (BigQuery + Vertex) path end to end.
#
# What this exercises, end to end on a live project:
#   - hands-on with a major cloud platform (GCP)
#   - data warehousing + production SQL (BigQuery: load + GROUP BY rollup)
#   - cloud-native vector search (BigQuery ML.GENERATE_EMBEDDING + VECTOR_SEARCH)
#
# Uses the `bq` CLI throughout (ships with gcloud — self-contained auth, no extra
# Python packages). Fact cards are built with the repo's own ingest logic (stdlib).
#
# Prereqs you do first (browser/account — cannot be automated):
#   1. A GCP project with billing enabled (new accounts get $300 free credit).
#   2. gcloud auth login
#
# Usage:
#   PY=/path/to/python3 \
#   bash cloud/run_on_gcp.sh <PROJECT_ID> [REGION]
#
# Cost: ~640 rows + a few queries + ~640 tiny embeddings => effectively $0,
# well inside the free tier / trial credit.
set -euo pipefail

PROJECT="${1:?usage: bash cloud/run_on_gcp.sh <PROJECT_ID> [REGION]}"
REGION="${2:-us-central1}"
DATASET="biosensor_rag"
TABLE="fact_cards"
CONN="bq_vertex_conn"
PY="${PY:-python3}"            # any python3 (stdlib only — used to build NDJSON + parse JSON)
NDJSON="/tmp/fact_cards.ndjson"
SCHEMA="doc_id:STRING,text:STRING,pmid:STRING,biomarker:STRING,biofluid:STRING,biosensor_principle:STRING,application:STRING,experiment_type:STRING"
BQ="bq --project_id=$PROJECT --location=$REGION query --use_legacy_sql=false"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "##### 0. project + APIs #####"
gcloud config set project "$PROJECT"
gcloud services enable bigquery.googleapis.com bigqueryconnection.googleapis.com aiplatform.googleapis.com

echo "##### A. data warehousing — build fact cards (repo ingest logic) -> load into BigQuery #####"
( cd "$REPO" && "$PY" - "$NDJSON" <<'PYEOF'
import json, sys
from ingest import load_fact_card_documents
with open(sys.argv[1], "w", encoding="utf-8") as f:
    for d in load_fact_card_documents():
        m = d.metadata
        f.write(json.dumps({
            "doc_id": d.doc_id, "text": d.text, "pmid": m.get("pmid"),
            "biomarker": m.get("biomarker"), "biofluid": m.get("biofluid"),
            "biosensor_principle": m.get("biosensor_principle"),
            "application": m.get("application"), "experiment_type": m.get("experiment_type"),
        }) + "\n")
print("[ndjson] wrote", sys.argv[1])
PYEOF
)
bq --location="$REGION" mk --dataset --force "$PROJECT:$DATASET" 2>/dev/null || echo "  (dataset exists)"
bq --project_id="$PROJECT" --location="$REGION" load \
   --source_format=NEWLINE_DELIMITED_JSON --replace \
   "$PROJECT:$DATASET.$TABLE" "$NDJSON" "$SCHEMA"
echo "  loaded fact_cards. row count:"
$BQ "SELECT COUNT(*) AS rows FROM \`$PROJECT.$DATASET.$TABLE\`"

echo "##### A2. production SQL — analyst rollup (GROUP BY over the warehouse) #####"
$BQ "$( cd "$REPO" && GOOGLE_CLOUD_PROJECT=$PROJECT GOOGLE_CLOUD_LOCATION=$REGION RAG_BQ_DATASET=$DATASET RAG_BQ_TABLE=$TABLE \
        "$PY" -c 'from cloud.gcp_bigquery import biomarker_rollup_sql; print(biomarker_rollup_sql())' )"

echo "##### B. cloud-native vector search — wire BigQuery to Vertex embeddings #####"
bq mk --connection --location="$REGION" --project_id="$PROJECT" \
   --connection_type=CLOUD_RESOURCE "$CONN" 2>/dev/null || echo "  (connection exists)"
SA=$(bq show --format=prettyjson --connection "$PROJECT.$REGION.$CONN" \
     | "$PY" -c "import sys,json;print(json.load(sys.stdin)['cloudResource']['serviceAccountId'])")
echo "  connection service account: $SA"
gcloud projects add-iam-policy-binding "$PROJECT" \
   --member="serviceAccount:$SA" --role="roles/aiplatform.user" --condition=None >/dev/null
echo "  granted roles/aiplatform.user; waiting 90s for IAM to propagate..."
sleep 90

$BQ "CREATE OR REPLACE MODEL \`$PROJECT.$DATASET.embed_model\`
       REMOTE WITH CONNECTION \`$PROJECT.$REGION.$CONN\`
       OPTIONS (ENDPOINT = 'text-embedding-004');"

$BQ "CREATE OR REPLACE TABLE \`$PROJECT.$DATASET.${TABLE}_emb\` AS
     SELECT doc_id, text, pmid, biomarker, biofluid, biosensor_principle, application,
            ml_generate_embedding_result AS embedding
     FROM ML.GENERATE_EMBEDDING(
            MODEL \`$PROJECT.$DATASET.embed_model\`,
            (SELECT doc_id, text, pmid, biomarker, biofluid, biosensor_principle, application
               FROM \`$PROJECT.$DATASET.$TABLE\`),
            STRUCT(TRUE AS flatten_json_output));"
# NB: CREATE VECTOR INDEX needs >=5000 rows; with ~640 cards VECTOR_SEARCH runs
# brute-force (no index) — correct and fast at this size.

echo "##### B2. semantic VECTOR_SEARCH entirely in SQL #####"
$BQ "SELECT base.pmid, base.biomarker, base.biofluid, base.biosensor_principle,
            SUBSTR(base.text,1,80) AS snippet, distance
     FROM VECTOR_SEARCH(
            TABLE \`$PROJECT.$DATASET.${TABLE}_emb\`, 'embedding',
            (SELECT ml_generate_embedding_result AS embedding
               FROM ML.GENERATE_EMBEDDING(
                      MODEL \`$PROJECT.$DATASET.embed_model\`,
                      (SELECT 'wearable sensors that measure cortisol in sweat' AS content),
                      STRUCT(TRUE AS flatten_json_output))),
            top_k => 6, distance_type => 'COSINE');"

echo "##### DONE #####"
echo "  - BigQuery console: dataset '$DATASET', tables '$TABLE' + '${TABLE}_emb'"
echo "  - the rollup result (A2) and the VECTOR_SEARCH result (B2) above"
