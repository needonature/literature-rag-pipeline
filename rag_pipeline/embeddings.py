"""Pluggable embedding backends.

Default = LocalHFEmbedder (sentence-transformers/all-MiniLM-L6-v2 run through the
already-installed `transformers` + `torch`): runs on a laptop, costs $0.

If the model can't be loaded (e.g. offline, no torch), we fall back to a classic
TF-IDF + LSA embedder via scikit-learn so the pipeline NEVER hard-fails (graceful degradation).

OpenAIEmbedder / VertexEmbedder are real cloud backends, constructed only when
explicitly selected (so importing this module never costs anything).

All backends share one interface:
    fit(texts)                      -> needed only by TF-IDF; no-op otherwise
    encode(texts, normalize=True)   -> np.ndarray [n, dim] float32
    save(dir) / load(dir)           -> persist any fitted state next to the index
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

import config


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
class BaseEmbedder:
    name: str = "base"
    dim: int = 0

    def fit(self, texts: Sequence[str]) -> "BaseEmbedder":
        return self

    def encode(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        raise NotImplementedError

    # default persistence just records which backend produced the index
    def save(self, dir_path: Path) -> None:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        (Path(dir_path) / "embedder.json").write_text(
            json.dumps({"name": self.name, "dim": self.dim}), encoding="utf-8"
        )

    @classmethod
    def _meta(cls, dir_path: Path) -> dict:
        p = Path(dir_path) / "embedder.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# --------------------------------------------------------------------------- #
# Local Hugging Face sentence embedder (default, free)
# --------------------------------------------------------------------------- #
class LocalHFEmbedder(BaseEmbedder):
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None, max_length: int = 256):
        import torch  # noqa: WPS433 - lazy import keeps cold start cheap
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name or config.LOCAL_EMBED_MODEL
        self.name = f"local:{self.model_name}"
        self.max_length = max_length
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dim = int(self.model.config.hidden_size)

    @staticmethod
    def _mean_pool(last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: Sequence[str], normalize: bool = True, batch_size: int = 32) -> np.ndarray:
        torch = self._torch
        vectors: List[np.ndarray] = []
        texts = [t if t else " " for t in texts]
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                enc = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
                ).to(self.device)
                out = self.model(**enc)
                emb = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
                if normalize:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                vectors.append(emb.cpu().numpy())
        return np.vstack(vectors).astype("float32")

    def save(self, dir_path: Path) -> None:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        (Path(dir_path) / "embedder.json").write_text(
            json.dumps({"name": self.name, "dim": self.dim, "kind": "local", "model_name": self.model_name}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, dir_path: Path) -> "LocalHFEmbedder":
        meta = cls._meta(dir_path)
        return cls(model_name=meta.get("model_name"))


# --------------------------------------------------------------------------- #
# TF-IDF + LSA fallback (offline, no downloads, no torch)
# --------------------------------------------------------------------------- #
class TfidfEmbedder(BaseEmbedder):
    def __init__(self, n_components: int = 256):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.name = "tfidf-lsa"
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1, max_features=50000
        )
        self.svd: Optional["TruncatedSVD"] = None  # type: ignore[name-defined]
        self.dim = n_components
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> "TfidfEmbedder":
        from sklearn.decomposition import TruncatedSVD

        tfidf = self.vectorizer.fit_transform(list(texts))
        n_comp = min(self.n_components, max(2, min(tfidf.shape) - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf)
        self.dim = n_comp
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        if not self._fitted or self.svd is None:
            raise RuntimeError("TfidfEmbedder must be .fit() on the corpus before encode().")
        tfidf = self.vectorizer.transform([t if t else " " for t in texts])
        emb = self.svd.transform(tfidf).astype("float32")
        if normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            emb = emb / norms
        return emb

    def save(self, dir_path: Path) -> None:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        with (Path(dir_path) / "tfidf_embedder.pkl").open("wb") as fh:
            pickle.dump({"vectorizer": self.vectorizer, "svd": self.svd, "dim": self.dim}, fh)
        (Path(dir_path) / "embedder.json").write_text(
            json.dumps({"name": self.name, "dim": self.dim, "kind": "tfidf"}), encoding="utf-8"
        )

    @classmethod
    def load(cls, dir_path: Path) -> "TfidfEmbedder":
        obj = cls()
        with (Path(dir_path) / "tfidf_embedder.pkl").open("rb") as fh:
            state = pickle.load(fh)
        obj.vectorizer = state["vectorizer"]
        obj.svd = state["svd"]
        obj.dim = state["dim"]
        obj._fitted = True
        return obj


# --------------------------------------------------------------------------- #
# OpenAI embeddings (cloud, optional) — reuses the OpenAI account you already use
# --------------------------------------------------------------------------- #
class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: Optional[str] = None):
        from openai import OpenAI  # lazy

        self.model = model or config.OPENAI_EMBED_MODEL
        self.name = f"openai:{self.model}"
        self.client = OpenAI()  # reads OPENAI_API_KEY
        self.dim = 1536 if "small" in self.model else 3072

    def encode(self, texts: Sequence[str], normalize: bool = True, batch_size: int = 256) -> np.ndarray:
        vectors: List[List[float]] = []
        texts = [t if t else " " for t in texts]
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            resp = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend([d.embedding for d in resp.data])
        emb = np.asarray(vectors, dtype="float32")
        if normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            emb = emb / norms
        self.dim = emb.shape[1]
        return emb


# --------------------------------------------------------------------------- #
# Vertex AI embeddings (GCP, optional) — see cloud/gcp_vertex.py for the full path
# --------------------------------------------------------------------------- #
class VertexEmbedder(BaseEmbedder):
    def __init__(self, model: Optional[str] = None):
        from google import genai  # lazy; new Google Gen AI SDK (replaces deprecated vertexai)

        self.model_name = model or config.VERTEX_EMBED_MODEL
        self.name = f"vertex:{self.model_name}"
        self.output_dim = config.VERTEX_EMBED_DIM
        self.client = genai.Client(
            vertexai=True, project=config.GCP_PROJECT or None, location=config.GCP_LOCATION
        )
        self.dim = self.output_dim

    def encode(self, texts: Sequence[str], normalize: bool = True, batch_size: int = 1) -> np.ndarray:
        # gemini-embedding-001 accepts only ONE input text per request on Vertex,
        # so we send sequentially (fine at this corpus size). output_dimensionality
        # applies Matryoshka truncation.
        from google.genai import types

        cfg = types.EmbedContentConfig(output_dimensionality=self.output_dim)
        vectors: List[List[float]] = []
        texts = [t if t else " " for t in texts]
        for t in texts:
            resp = self.client.models.embed_content(model=self.model_name, contents=t, config=cfg)
            vectors.append(resp.embeddings[0].values)
        emb = np.asarray(vectors, dtype="float32")
        if normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            emb = emb / norms
        self.dim = emb.shape[1]
        return emb


# --------------------------------------------------------------------------- #
# Factory + robust default with graceful fallback
# --------------------------------------------------------------------------- #
def get_embedder(backend: Optional[str] = None) -> BaseEmbedder:
    backend = (backend or config.EMBED_BACKEND).lower()
    if backend == "openai":
        return OpenAIEmbedder()
    if backend == "vertex":
        return VertexEmbedder()
    if backend == "bedrock":
        from cloud.aws_bedrock import BedrockEmbedder  # lazy; see cloud/aws_bedrock.py

        return BedrockEmbedder()
    if backend == "tfidf":
        return TfidfEmbedder()
    # default: local HF, but degrade gracefully to TF-IDF if it can't load
    try:
        return LocalHFEmbedder()
    except Exception as exc:  # noqa: BLE001
        print(f"[embeddings] local HF model unavailable ({exc}); falling back to TF-IDF/LSA.")
        return TfidfEmbedder()


def load_embedder(dir_path: Path) -> BaseEmbedder:
    """Reconstruct the SAME embedder that built the index (so query vectors match)."""
    meta = BaseEmbedder._meta(dir_path)
    kind = meta.get("kind")
    if kind == "tfidf":
        return TfidfEmbedder.load(dir_path)
    if kind == "local":
        return LocalHFEmbedder.load(dir_path)
    if kind == "openai" or (meta.get("name", "").startswith("openai")):
        return OpenAIEmbedder(model=meta.get("model_name"))
    if kind == "vertex" or (meta.get("name", "").startswith("vertex")):
        return VertexEmbedder(model=meta.get("model_name"))
    if kind == "bedrock" or (meta.get("name", "").startswith("bedrock")):
        from cloud.aws_bedrock import BedrockEmbedder

        return BedrockEmbedder(model=meta.get("model_name"), dim=meta.get("dim"))
    # last resort: rebuild from current config
    return get_embedder()
