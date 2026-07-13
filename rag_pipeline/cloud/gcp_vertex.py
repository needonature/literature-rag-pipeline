"""GCP path #1 — Vertex AI: cloud embeddings, cloud-native vector search, Gemini.

Cloud embeddings, cloud-native vector search, and Gemini generation as
real, runnable code. It mirrors the
local interfaces (embeddings.py / vector_store.py / generator.py) so the rest of
the pipeline is unchanged — you swap backends with env vars, not code.

Nothing here runs until you call it AND set:
    GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION
    RAG_VERTEX_INDEX_ENDPOINT, RAG_VERTEX_DEPLOYED_INDEX_ID   (for vector search)
plus Application Default Credentials (`gcloud auth application-default login`).

Pieces
------
  VertexVectorSearch  - upsert vectors to / query a Vertex AI Vector Search index
                        (Google's managed ANN service = cloud-native vector DB).
  VertexGeminiGenerator - grounded generation with Gemini on Vertex AI.
  push_local_index_to_vertex() - take the locally-built embeddings and stream
                        them into a Vertex index (so you can demo the cloud path
                        without re-embedding / re-paying).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from generator import BaseGenerator, SYSTEM_PROMPT, build_prompt, parse_cited_tags
from schema import RetrievedChunk


def _require_gcp() -> None:
    if not config.GCP_PROJECT:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT is not set. Configure GCP env + run "
            "`gcloud auth application-default login` before using the Vertex backends."
        )


# --------------------------------------------------------------------------- #
# Cloud-native vector search (Vertex AI Vector Search, a.k.a. Matching Engine)
# --------------------------------------------------------------------------- #
class VertexVectorSearch:
    """Thin wrapper over a deployed Vertex AI Vector Search index endpoint."""

    backend = "vertex_vector_search"

    def __init__(
        self,
        index_endpoint: Optional[str] = None,
        deployed_index_id: Optional[str] = None,
    ):
        _require_gcp()
        from google.cloud import aiplatform

        aiplatform.init(project=config.GCP_PROJECT, location=config.GCP_LOCATION)
        self._aiplatform = aiplatform
        self.index_endpoint_name = index_endpoint or config.VERTEX_INDEX_ENDPOINT
        self.deployed_index_id = deployed_index_id or config.VERTEX_DEPLOYED_INDEX_ID
        self._endpoint = aiplatform.MatchingEngineIndexEndpoint(self.index_endpoint_name)

    def query(self, query_vec: np.ndarray, k: int = 6) -> List[Tuple[str, float]]:
        resp = self._endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[query_vec.reshape(-1).tolist()],
            num_neighbors=k,
        )
        out: List[Tuple[str, float]] = []
        for neighbor in resp[0]:
            # Vertex returns distance; convert to a similarity-ish score for ranking.
            out.append((neighbor.id, 1.0 - float(neighbor.distance)))
        return out


def create_index(display_name: str, dimensions: int, gcs_uri: Optional[str] = None) -> str:
    """Create a Vertex AI Vector Search index (tree-AH ANN). Returns the index id.

    For a few thousand vectors you can also use brute-force; tree-AH scales to
    millions, which is the point of going cloud-native for the full 14M-corpus.
    """
    _require_gcp()
    from google.cloud import aiplatform

    aiplatform.init(project=config.GCP_PROJECT, location=config.GCP_LOCATION)
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=display_name,
        dimensions=dimensions,
        approximate_neighbors_count=150,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
        contents_delta_uri=gcs_uri,  # None => create empty, then stream-upsert
        index_update_method="STREAM_UPDATE",
    )
    print(f"[vertex] created index: {index.resource_name}")
    return index.resource_name


def push_local_index_to_vertex(index_resource_name: str, local_index_dir: Optional[Path] = None) -> None:
    """Stream the locally-built embeddings into a Vertex index (reuse, no re-embed)."""
    _require_gcp()
    from google.cloud import aiplatform

    local_index_dir = Path(local_index_dir or config.INDEX_DIR)
    embeddings = np.load(local_index_dir / "embeddings.npy")
    doc_ids: List[str] = []
    with (local_index_dir / "documents.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                doc_ids.append(json.loads(line)["doc_id"])

    aiplatform.init(project=config.GCP_PROJECT, location=config.GCP_LOCATION)
    index = aiplatform.MatchingEngineIndex(index_resource_name)
    datapoints = [
        {"datapoint_id": doc_ids[i], "feature_vector": embeddings[i].tolist()}
        for i in range(len(doc_ids))
    ]
    # batch the stream upsert
    BATCH = 1000
    for start in range(0, len(datapoints), BATCH):
        index.upsert_datapoints(datapoints=datapoints[start : start + BATCH])
    print(f"[vertex] upserted {len(datapoints)} datapoints into {index_resource_name}")


# --------------------------------------------------------------------------- #
# Gemini generation on Vertex AI
# --------------------------------------------------------------------------- #
class VertexGeminiGenerator(BaseGenerator):
    name = "vertex_gemini"

    def __init__(self, model: Optional[str] = None):
        _require_gcp()
        from google import genai  # new Google Gen AI SDK (replaces deprecated vertexai)

        self.model = model or config.GEMINI_MODEL
        self._client = genai.Client(
            vertexai=True, project=config.GCP_PROJECT, location=config.GCP_LOCATION
        )

    def generate(self, question: str, contexts: List[RetrievedChunk]) -> Dict[str, Any]:
        from google.genai import types

        prompt = build_prompt(question, contexts, config.MAX_CONTEXT_CHARS)
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.0),
        )
        answer = (resp.text or "").strip()
        usage = self._usage(prompt, answer)
        try:  # Vertex returns usage_metadata
            um = resp.usage_metadata
            usage["input_tokens"] = um.prompt_token_count
            usage["output_tokens"] = um.candidates_token_count
        except Exception:  # noqa: BLE001
            pass
        return {"answer": answer, "citations": parse_cited_tags(answer), "usage": usage}


if __name__ == "__main__":
    # Smoke test that the module imports cleanly even with no GCP creds.
    print("gcp_vertex.py import OK. Set GOOGLE_CLOUD_PROJECT to use the Vertex backends.")
    print("config:", config.GCP_PROJECT or "(no project set)", config.GCP_LOCATION)
