# Container for the RAG assistant (Flask app) -> Google Cloud Run.
# Build context is the rag_pipeline/ folder:
#   docker build -f cloud/Dockerfile -t biosensor-rag .
#
# By default the image runs the FREE backends (local embeddings + extractive
# generation). Set RAG_GEN_BACKEND / RAG_EMBED_BACKEND env vars on the Cloud Run
# service to switch to Vertex AI + Gemini.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    PORT=8080

WORKDIR /app

# System deps kept minimal; CPU-only torch wheel is pulled via requirements.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt gunicorn

COPY . .

# Bake a prebuilt index into the image: run `python build_index.py` locally
# BEFORE `docker build`, so the index/ folder is copied in above. The serving
# side loads whatever embedder produced that index (see load_embedder), so the
# image is self-contained. (The source data lives OUTSIDE this folder and isn't
# in the build context, so we don't rebuild in-container. To rebuild here
# instead, COPY the data files in + set RAG_DATA_DIR, or mount them at runtime.)

EXPOSE 8080
# 1 worker keeps the in-memory index resident; threads handle concurrency.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 120 app:app
