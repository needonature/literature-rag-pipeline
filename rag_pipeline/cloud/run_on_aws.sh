#!/usr/bin/env bash
# AWS run for the biosensor RAG — the twin of run_on_gcp.sh.
#
# Part A runs FULLY LOCAL ($0, no AWS account): an open-source pgvector database
# in Docker + vector search in SQL. Parts B/C use real AWS (Amazon Bedrock) and
# need `aws configure` + Bedrock model access.
#
# What this demonstrates:
#   - vector databases incl. open-source (pgvector) + cloud-native (OpenSearch)
#   - hands-on with a major cloud platform (AWS: Bedrock)
#   - LLM APIs (Claude on Bedrock); production SQL (the pgvector k-NN query)
#
# Usage:
#   pip install -r requirements.txt -r requirements-aws.txt
#   PY=python3 bash cloud/run_on_aws.sh
set -euo pipefail

PY="${PY:-python3}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export RAG_PG_DSN="${RAG_PG_DSN:-postgresql://rag:rag@localhost:5432/rag}"

echo "##### A. open-source vector DB (pgvector) — local Docker, \$0, vector search in SQL #####"
if ! docker exec ragpg pg_isready -U rag >/dev/null 2>&1; then
  docker rm -f ragpg >/dev/null 2>&1 || true
  docker run -d --name ragpg -p 5432:5432 \
    -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=rag \
    pgvector/pgvector:pg16 >/dev/null
  echo "  started pgvector Postgres; waiting for readiness..."
  for i in $(seq 1 40); do docker exec ragpg pg_isready -U rag >/dev/null 2>&1 && break; sleep 1; done
fi
( cd "$REPO" && "$PY" cloud/aws_pgvector.py "wearable sensors that measure cortisol in sweat" )

echo
echo "##### B. Amazon Bedrock — Titan embeddings into the SAME pgvector table #####"
echo "#   needs: aws configure + Bedrock model access to amazon.titan-embed-text-v2:0"
echo "#   run:   RAG_EMBED_BACKEND=bedrock $PY cloud/aws_pgvector.py \"cortisol in sweat\""
if [ "${RUN_BEDROCK:-0}" = "1" ]; then
  ( cd "$REPO" && RAG_EMBED_BACKEND=bedrock "$PY" cloud/aws_pgvector.py "cortisol in sweat" )
fi

echo
echo "##### C. Claude on Amazon Bedrock — grounded, PMID-cited answer #####"
echo "#   needs: Bedrock model access to your Claude model (RAG_BEDROCK_GEN_MODEL)"
echo "#   run:   $PY build_index.py --no-fact-cards"
echo "#          RAG_GEN_BACKEND=bedrock $PY ask.py \"Which wearable sensors measure cortisol in sweat?\""
if [ "${RUN_BEDROCK:-0}" = "1" ]; then
  ( cd "$REPO" && "$PY" build_index.py --no-fact-cards \
      && RAG_GEN_BACKEND=bedrock "$PY" ask.py "Which wearable sensors measure cortisol in sweat?" )
fi

echo
echo "##### DONE. Part A is the live, runnable open-source vector DB demo."
echo "#####       Set RUN_BEDROCK=1 (after aws configure + model access) to run B & C."
echo "#####       Stop the local DB with:  docker rm -f ragpg"
