"""Turn the EXISTING pipeline outputs into a retrieval corpus.

This is where we reuse the ~$100 OpenAI Batch extraction instead of paying again:

  1. Abstract documents  - clean rows (PMID, Title, Abstract, Keywords) from the
     PubMed extract CSV. These are the rich, unstructured text we retrieve over.

  2. Fact-card documents  - one short, self-contained sentence per row of the
     distilled Biomarker<->Biosensor table that the extraction pipeline produced
     (combined_output_*.json). These make the curated knowledge directly
     retrievable and give every answer a crisp, grounded "fact" to cite.

Both kinds become `Document` objects with normalized metadata so the rest of the
pipeline never has to care which source a hit came from.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config import ABSTRACTS_CSV, DATA_DIR, STRUCTURED_JSONS
from schema import Document

# Allow very wide CSV fields (PubMed abstracts can be long).
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


def _clean(value: Any) -> str:
    """Coerce a value (which may be a list, None, or number) into a clean string."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(_clean(v) for v in value if v not in (None, ""))
    return str(value).strip()


def _norm_key(value: str) -> str:
    return _clean(value).lower()


# --------------------------------------------------------------------------- #
# 1. Abstract documents
# --------------------------------------------------------------------------- #
def load_abstract_documents(csv_name: Optional[str] = None, limit: Optional[int] = None) -> List[Document]:
    csv_name = csv_name or ABSTRACTS_CSV
    path = Path(csv_name)
    if not path.is_absolute():
        path = DATA_DIR / csv_name
    if not path.exists():
        print(f"[ingest] WARNING: abstracts CSV not found: {path}")
        return []

    docs: List[Document] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            pmid = _clean(row.get("PMID"))
            title = _clean(row.get("Title"))
            abstract = _clean(row.get("Abstract"))
            if not (title or abstract):
                continue
            keywords = _clean(row.get("Keywords"))
            text_parts = [title, abstract]
            if keywords:
                text_parts.append(f"Keywords: {keywords}")
            doc = Document(
                doc_id=f"abs::{pmid or i}",
                text="\n".join(p for p in text_parts if p),
                metadata={
                    "source": "abstract",
                    "pmid": pmid or None,
                    "title": title or None,
                    "year": _clean(row.get("PubDate")) or None,
                    "journal": _clean(row.get("JournalTitle")) or None,
                    "keywords": keywords or None,
                },
            )
            docs.append(doc)
    print(f"[ingest] {len(docs):>5} abstract documents from {path.name}")
    return docs


# --------------------------------------------------------------------------- #
# 2. Fact-card documents (the distilled Biomarker<->Biosensor table)
# --------------------------------------------------------------------------- #
def _fact_card_text(rec: Dict[str, Any]) -> str:
    principle = _clean(rec.get("Biosensor_Principle"))
    biofluid = _clean(rec.get("Biofluid"))
    biomarker = _clean(rec.get("Biomarker"))
    application = _clean(rec.get("Primary_Application"))
    product = _clean(rec.get("Example_Product"))
    exp = _clean(rec.get("Experiment_Type"))

    sent = []
    if principle and biomarker:
        s = f"A {principle} is used to detect/measure {biomarker}"
        if biofluid:
            s += f" in {biofluid}"
        s += "."
        sent.append(s)
    elif biomarker:
        sent.append(f"Biomarker measured: {biomarker} (biofluid: {biofluid or 'n/a'}).")
    if application:
        sent.append(f"Primary application: {application}.")
    if product:
        sent.append(f"Example product/platform: {product}.")
    if exp:
        sent.append(f"Evidence level / experiment type: {exp}.")
    return " ".join(sent).strip()


def load_fact_card_documents(json_names: Optional[Iterable[str]] = None) -> List[Document]:
    json_names = list(json_names) if json_names else list(STRUCTURED_JSONS)
    docs: List[Document] = []
    seen = set()
    for name in json_names:
        path = DATA_DIR / name
        if not path.exists():
            print(f"[ingest] WARNING: structured JSON not found: {path}")
            continue
        try:
            records = json.load(path.open("r", encoding="utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001 - resilient ingest
            print(f"[ingest] WARNING: could not parse {name}: {exc}")
            continue
        if isinstance(records, dict):
            records = [records]
        n = 0
        for rec in records:
            if not isinstance(rec, dict):
                continue
            text = _fact_card_text(rec)
            if not text:
                continue
            pmid = _clean(rec.get("PMID") or rec.get("pmid"))
            rid = _clean(rec.get("id"))
            key = (pmid, rid, text)
            if key in seen:
                continue
            seen.add(key)
            uid = pmid or rid or f"{name}-{n}"
            docs.append(
                Document(
                    doc_id=f"fact::{name}::{uid}",
                    text=text,
                    metadata={
                        "source": "fact_card",
                        "pmid": pmid or None,
                        "record_id": rid or None,
                        "biomarker": _clean(rec.get("Biomarker")) or None,
                        "biofluid": _clean(rec.get("Biofluid")) or None,
                        "biosensor_principle": _clean(rec.get("Biosensor_Principle")) or None,
                        "application": _clean(rec.get("Primary_Application")) or None,
                        "experiment_type": _clean(rec.get("Experiment_Type")) or None,
                        "origin_file": name,
                    },
                )
            )
            n += 1
        print(f"[ingest] {n:>5} fact-card documents from {name}")
    return docs


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def build_corpus(
    abstracts_csv: Optional[object] = None,
    include_fact_cards: bool = True,
    abstract_limit: Optional[int] = None,
) -> List[Document]:
    # Accept one CSV, a comma-separated string, or a list -> load + concatenate.
    sources = abstracts_csv if abstracts_csv is not None else ABSTRACTS_CSV
    if isinstance(sources, str):
        sources = [s.strip() for s in sources.split(",") if s.strip()]
    docs: List[Document] = []
    for src in sources:
        docs += load_abstract_documents(src, limit=abstract_limit)
    if include_fact_cards:
        docs += load_fact_card_documents()
    # De-dup on (doc_id) just in case.
    uniq: Dict[str, Document] = {}
    for d in docs:
        uniq[d.doc_id] = d
    out = list(uniq.values())
    print(f"[ingest] total corpus: {len(out)} documents")
    return out


if __name__ == "__main__":
    corpus = build_corpus()
    for d in corpus[:3]:
        print("-" * 60)
        print(d.doc_id, "|", {k: v for k, v in d.metadata.items() if v})
        print(d.text[:300])
