"""A tiny vector store with two interchangeable backends.

  * FaissStore  - uses faiss IndexFlatIP if faiss is installed (cosine over
                  L2-normalized vectors).
  * NumpyStore  - pure-numpy brute-force search; no extra dependency, identical
                  results for our corpus size (a few thousand docs).

Both persist to INDEX_DIR so build/query are separate steps. For the
cloud-native equivalent (Vertex AI Vector Search / BigQuery VECTOR_SEARCH) see
cloud/gcp_vertex.py and cloud/gcp_bigquery.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from schema import Document

try:  # optional, swapped in automatically when present
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # noqa: BLE001
    _HAS_FAISS = False


class _BaseStore:
    backend = "base"

    def __init__(self) -> None:
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self.dim: int = 0

    # -- build -------------------------------------------------------------- #
    def build(self, documents: List[Document], embeddings: np.ndarray) -> None:
        if len(documents) != embeddings.shape[0]:
            raise ValueError("documents and embeddings length mismatch")
        self.documents = documents
        self.embeddings = embeddings.astype("float32")
        self.dim = int(embeddings.shape[1])
        self._post_build()

    def _post_build(self) -> None:  # subclass hook
        pass

    # -- search ------------------------------------------------------------- #
    def search(self, query_vec: np.ndarray, k: int = 6) -> List[Tuple[int, float]]:
        raise NotImplementedError

    # -- persistence -------------------------------------------------------- #
    def save(self, dir_path: Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        np.save(dir_path / "embeddings.npy", self.embeddings)
        with (dir_path / "documents.jsonl").open("w", encoding="utf-8") as fh:
            for d in self.documents:
                fh.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
        (dir_path / "store.json").write_text(
            json.dumps({"backend": self.backend, "dim": self.dim, "n": len(self.documents)}),
            encoding="utf-8",
        )

    @classmethod
    def _load_common(cls, dir_path: Path) -> Tuple[List[Document], np.ndarray]:
        dir_path = Path(dir_path)
        embeddings = np.load(dir_path / "embeddings.npy")
        documents: List[Document] = []
        with (dir_path / "documents.jsonl").open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    documents.append(Document.from_dict(json.loads(line)))
        return documents, embeddings


class NumpyStore(_BaseStore):
    backend = "numpy"

    def search(self, query_vec: np.ndarray, k: int = 6) -> List[Tuple[int, float]]:
        if self.embeddings is None or len(self.documents) == 0:
            return []
        q = np.asarray(query_vec, dtype="float32").reshape(-1)
        scores = self.embeddings @ q  # cosine if both normalized
        k = min(k, scores.shape[0])
        # argpartition for top-k, then sort those k
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    @classmethod
    def load(cls, dir_path: Path) -> "NumpyStore":
        documents, embeddings = cls._load_common(dir_path)
        store = cls()
        store.documents = documents
        store.embeddings = embeddings.astype("float32")
        store.dim = int(embeddings.shape[1])
        return store


class FaissStore(_BaseStore):
    backend = "faiss"

    def __init__(self) -> None:
        super().__init__()
        self._index = None

    def _post_build(self) -> None:
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(self.embeddings)

    def search(self, query_vec: np.ndarray, k: int = 6) -> List[Tuple[int, float]]:
        if self._index is None:
            return []
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        k = min(k, len(self.documents))
        scores, idxs = self._index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]

    def save(self, dir_path: Path) -> None:
        super().save(dir_path)
        faiss.write_index(self._index, str(Path(dir_path) / "faiss.index"))

    @classmethod
    def load(cls, dir_path: Path) -> "FaissStore":
        documents, embeddings = cls._load_common(dir_path)
        store = cls()
        store.documents = documents
        store.embeddings = embeddings.astype("float32")
        store.dim = int(embeddings.shape[1])
        idx_path = Path(dir_path) / "faiss.index"
        if idx_path.exists():
            store._index = faiss.read_index(str(idx_path))
        else:
            store._post_build()
        return store


def get_store() -> _BaseStore:
    """Pick the best available backend automatically."""
    return FaissStore() if _HAS_FAISS else NumpyStore()


def load_store(dir_path: Path) -> _BaseStore:
    meta = json.loads((Path(dir_path) / "store.json").read_text(encoding="utf-8"))
    if meta.get("backend") == "faiss" and _HAS_FAISS:
        return FaissStore.load(dir_path)
    return NumpyStore.load(dir_path)
