"""Small text helpers shared by chunking and hybrid (lexical) retrieval."""
from __future__ import annotations

import re
from typing import List

from schema import Document

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "was", "were", "be", "by", "as", "at", "from", "that", "this", "it",
    "we", "using", "used", "use", "based", "study", "via", "which",
}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


def chunk_document(doc: Document, max_words: int, overlap_words: int) -> List[Document]:
    """Split a long document into overlapping word windows.

    Abstracts and fact cards are short, so most documents stay whole; this only
    kicks in for long records, and preserves metadata + a chunk suffix on the id.
    """
    words = (doc.text or "").split()
    if len(words) <= max_words:
        return [doc]
    chunks: List[Document] = []
    step = max(1, max_words - overlap_words)
    for ci, start in enumerate(range(0, len(words), step)):
        window = words[start : start + max_words]
        if not window:
            break
        meta = dict(doc.metadata)
        meta["parent_id"] = doc.doc_id
        meta["chunk"] = ci
        chunks.append(Document(doc_id=f"{doc.doc_id}#c{ci}", text=" ".join(window), metadata=meta))
        if start + max_words >= len(words):
            break
    return chunks
