"""Hybrid retriever: dense semantic search + lexical overlap, with metadata filters.

Why hybrid: pure vector search can miss exact tokens (an analyte name, a PMID),
while pure keyword search misses paraphrases. We blend a normalized semantic
score with a normalized keyword-overlap score (HYBRID_ALPHA controls the mix) —
a standard, defensible RAG retrieval pattern that's easy to explain and tune.

Metadata filtering (e.g. biofluid="Sweat") lets the same index back both broad
Q&A and precise "structured lookups" over the distilled fact cards.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import config
from embeddings import BaseEmbedder, load_embedder
from schema import RetrievedChunk
from text_utils import tokenize
from vector_store import _BaseStore, load_store


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class Retriever:
    def __init__(self, store: _BaseStore, embedder: BaseEmbedder, hybrid_alpha: Optional[float] = None):
        self.store = store
        self.embedder = embedder
        self.hybrid_alpha = config.HYBRID_ALPHA if hybrid_alpha is None else hybrid_alpha
        # Precompute lexical token sets once for the whole corpus.
        self._doc_tokens: List[set] = [set(tokenize(d.text)) for d in store.documents]

    @classmethod
    def from_index(cls, index_dir: Optional[Path] = None) -> "Retriever":
        index_dir = Path(index_dir or config.INDEX_DIR)
        store = load_store(index_dir)
        embedder = load_embedder(index_dir)
        return cls(store, embedder)

    # ---------------------------------------------------------------- #
    def _lexical_scores(self, query: str) -> np.ndarray:
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return np.zeros(len(self.store.documents), dtype="float32")
        scores = np.fromiter(
            (len(q_tokens & dt) / len(q_tokens) for dt in self._doc_tokens),
            dtype="float32",
            count=len(self._doc_tokens),
        )
        return scores

    @staticmethod
    def _passes(meta: Dict[str, Any], filters: Optional[Dict[str, str]]) -> bool:
        if not filters:
            return True
        for key, want in filters.items():
            have = meta.get(key)
            if have is None:
                return False
            if str(want).lower() not in str(have).lower():
                return False
        return True

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, str]] = None,
        candidate_pool: int = 50,
    ) -> List[RetrievedChunk]:
        k = k or config.TOP_K
        docs = self.store.documents
        if not docs:
            return []

        # 1) dense semantic candidates
        q_vec = self.embedder.encode([query], normalize=True)[0]
        sem_hits = self.store.search(q_vec, k=min(candidate_pool, len(docs)))
        sem_score_by_idx = {idx: s for idx, s in sem_hits}

        # 2) lexical scores over the same candidate set (plus any strong lexical-only hits)
        lex_all = self._lexical_scores(query)
        lex_top = np.argsort(-lex_all)[: min(candidate_pool, len(docs))]
        cand = set(sem_score_by_idx) | {int(i) for i in lex_top if lex_all[i] > 0}
        cand_idx = [i for i in cand if self._passes(docs[i].metadata, filters)]
        if not cand_idx:
            # filters too strict -> fall back to semantic-only over all docs honoring filters
            cand_idx = [i for i, _ in sem_hits if self._passes(docs[i].metadata, filters)]
        if not cand_idx:
            return []

        sem_list = [sem_score_by_idx.get(i, 0.0) for i in cand_idx]
        lex_list = [float(lex_all[i]) for i in cand_idx]
        sem_n = _minmax(sem_list)
        lex_n = _minmax(lex_list)
        alpha = self.hybrid_alpha

        scored: List[RetrievedChunk] = []
        for pos, i in enumerate(cand_idx):
            blended = alpha * sem_n[pos] + (1 - alpha) * lex_n[pos]
            d = docs[i]
            scored.append(
                RetrievedChunk(
                    doc_id=d.doc_id,
                    text=d.text,
                    score=float(blended),
                    metadata=d.metadata,
                    semantic_score=float(sem_list[pos]),
                    lexical_score=float(lex_list[pos]),
                )
            )
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:k]
