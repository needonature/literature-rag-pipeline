"""AWS path #1 — Amazon Bedrock backends (mirror of cloud/gcp_vertex.py).

  BedrockEmbedder        -> Titan Text Embeddings v2  (RAG_EMBED_BACKEND=bedrock)
  BedrockClaudeGenerator -> Claude on Bedrock          (RAG_GEN_BACKEND=bedrock)

Both plug into the existing factories in embeddings.py / generator.py, so the
rest of the pipeline (chunk -> retrieve -> answer) is unchanged — only the env
var flips. boto3 is imported lazily, so importing this module never needs AWS.

Prereqs to actually run:
  pip install -r requirements-aws.txt          # boto3
  aws configure                                # creds + region
  # enable model access in the Bedrock console for Titan + your Claude model
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import config
from embeddings import BaseEmbedder
from generator import (
    SYSTEM_PROMPT,
    BaseGenerator,
    build_prompt,
    cited_pmids,
    estimate_cost,
)
from schema import RetrievedChunk


def _runtime(region: Optional[str] = None):
    import boto3  # lazy

    return boto3.client("bedrock-runtime", region_name=region or config.AWS_REGION)


# --------------------------------------------------------------------------- #
# Embeddings — Amazon Titan Text Embeddings v2
# --------------------------------------------------------------------------- #
class BedrockEmbedder(BaseEmbedder):
    def __init__(self, model: Optional[str] = None, dim: Optional[int] = None, region: Optional[str] = None):
        self.model = model or config.BEDROCK_EMBED_MODEL
        self.dim = int(dim or config.BEDROCK_EMBED_DIM)
        self.name = f"bedrock:{self.model}"
        self._client = _runtime(region)

    def encode(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        vecs: List[List[float]] = []
        for t in texts:
            body = json.dumps({"inputText": (t or " ")[:8000], "dimensions": self.dim, "normalize": bool(normalize)})
            resp = self._client.invoke_model(modelId=self.model, body=body)
            payload = json.loads(resp["body"].read())
            vecs.append(payload["embedding"])
        emb = np.asarray(vecs, dtype="float32")
        if normalize and emb.size:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            emb = emb / norms
        if emb.size:
            self.dim = emb.shape[1]
        return emb

    def save(self, dir_path: Path) -> None:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        (Path(dir_path) / "embedder.json").write_text(
            json.dumps({"name": self.name, "dim": self.dim, "kind": "bedrock", "model_name": self.model}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, dir_path: Path) -> "BedrockEmbedder":
        meta = BaseEmbedder._meta(dir_path)
        return cls(model=meta.get("model_name"), dim=meta.get("dim"))


# --------------------------------------------------------------------------- #
# Generation — Claude on Amazon Bedrock (Anthropic messages API via invoke_model)
# --------------------------------------------------------------------------- #
class BedrockClaudeGenerator(BaseGenerator):
    name = "bedrock"

    def __init__(self, model: Optional[str] = None, region: Optional[str] = None):
        self.model = model or config.BEDROCK_GEN_MODEL
        self._client = _runtime(region)

    def generate(self, question: str, contexts: List[RetrievedChunk]) -> Dict[str, Any]:
        prompt = build_prompt(question, contexts, config.MAX_CONTEXT_CHARS)
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 700,
                "temperature": 0,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        resp = self._client.invoke_model(modelId=self.model, body=body)
        payload = json.loads(resp["body"].read())
        answer = "".join(
            b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text"
        ).strip()
        usage = self._usage(prompt, answer)
        u = payload.get("usage") or {}
        if u:
            usage["input_tokens"] = u.get("input_tokens")
            usage["output_tokens"] = u.get("output_tokens")
            usage["est_cost_usd"] = round(
                estimate_cost(self.model, u.get("input_tokens", 0) or 0, u.get("output_tokens", 0) or 0), 6
            )
        return {"answer": answer, "citations": cited_pmids(contexts), "usage": usage}


if __name__ == "__main__":
    print("aws_bedrock.py import OK.")
    print("  embed model:", config.BEDROCK_EMBED_MODEL, "| gen model:", config.BEDROCK_GEN_MODEL)
    print("  region:", config.AWS_REGION)
    print("  usage:  RAG_EMBED_BACKEND=bedrock RAG_GEN_BACKEND=bedrock python ask.py \"...\"")
