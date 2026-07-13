"""Shared data structures for the RAG pipeline.

Kept deliberately tiny and dependency-free so every other module (ingest,
embeddings, vector store, retriever, generator) can import from one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """A unit of knowledge in the corpus (an abstract, or a distilled fact card)."""
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Document":
        return Document(doc_id=d["doc_id"], text=d["text"], metadata=d.get("metadata", {}) or {})


@dataclass
class RetrievedChunk:
    """A document returned by the retriever, with its relevance score."""
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    semantic_score: Optional[float] = None
    lexical_score: Optional[float] = None

    @property
    def pmid(self) -> Optional[str]:
        return self.metadata.get("pmid")

    @property
    def citation(self) -> str:
        pmid = self.metadata.get("pmid")
        if pmid:
            return f"PMID:{pmid}"
        return f"DOC:{self.doc_id}"


@dataclass
class RagAnswer:
    """The full result of a RAG query: the answer plus everything needed to audit it."""
    question: str
    answer: str
    citations: List[str] = field(default_factory=list)
    contexts: List[RetrievedChunk] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "contexts": [
                {
                    "doc_id": c.doc_id,
                    "score": round(c.score, 4),
                    "pmid": c.metadata.get("pmid"),
                    "title": c.metadata.get("title"),
                    "biomarker": c.metadata.get("biomarker"),
                    "biofluid": c.metadata.get("biofluid"),
                    "text": c.text[:400],
                }
                for c in self.contexts
            ],
            "usage": self.usage,
        }
