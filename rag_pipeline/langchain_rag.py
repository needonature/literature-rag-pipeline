"""LangChain implementation of the biosensor-literature RAG.

This is an ALTERNATIVE to the hand-rolled pipeline in this repo — same corpus,
same idea (retrieve abstracts -> grounded, PMID-cited answer) — but assembled
from off-the-shelf LangChain components wherever possible:

    document loading .... csv -> langchain_core.documents.Document
    chunking ............ RecursiveCharacterTextSplitter        (langchain_text_splitters)
    retrieval ........... TFIDFRetriever (default, $0)           (langchain_community.retrievers)
                          or Chroma + OpenAIEmbeddings           (open-source vector DB, opt-in)
    RAG chain ........... create_retrieval_chain +
                          create_stuff_documents_chain           (langchain.chains)
    generation .......... ChatOpenAI / ChatAnthropic / ChatOllama (pluggable LLM)

Default path runs at $0 / offline (no API key, no torch): LangChain's TF-IDF
retriever + a no-LLM citation formatter. Set LCRAG_LLM to synthesize a real
answer with an LLM.

Run:
    pip install -r requirements-langchain.txt
    python langchain_rag.py "Which wearable sensors measure cortisol in sweat?"

Env switches:
    LCRAG_RETRIEVER = tfidf (default) | chroma     # chroma needs OPENAI_API_KEY + langchain-chroma
    LCRAG_LLM       = none  (default) | openai | anthropic | ollama
    LCRAG_TOP_K     = 6
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

csv.field_size_limit(2_147_483_647)

# Same corpus the hand-rolled pipeline uses (the two NER-confirmed abstract sets,
# both time windows). They live one level up, in the pipeline folder.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("RAG_DATA_DIR", os.path.dirname(HERE))
ABSTRACT_CSVS = [
    "biosensor_ner_2014to2019_A_P_T_A.csv",
    "biosensor_ner_2021plus_A_P_T_A.csv",
]
TOP_K = int(os.environ.get("LCRAG_TOP_K", "6"))

PROMPT = ChatPromptTemplate.from_template(
    "You are a biomedical assistant for wearable-biosensor literature. "
    "Answer the question using ONLY the context below. After each claim, cite the "
    "source as [PMID:xxxxx]. If the context does not contain the answer, say so.\n\n"
    "Context:\n{context}\n\nQuestion: {input}\n\nGrounded answer:"
)
# Injects the PMID into each retrieved chunk so the LLM can cite it.
DOC_PROMPT = PromptTemplate.from_template("[PMID:{pmid}] {page_content}")


def load_documents() -> list[Document]:
    docs: list[Document] = []
    for name in ABSTRACT_CSVS:
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            print(f"[lc-rag] WARNING: corpus file not found: {path}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                title = (row.get("Title") or "").strip()
                abstract = (row.get("Abstract") or "").strip()
                pmid = (row.get("PMID") or "").strip()
                content = (title + "\n" + abstract).strip()
                if not content:
                    continue
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"pmid": pmid or "n/a", "title": title, "source": name},
                    )
                )
    return docs


def build_retriever(docs: list[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    backend = os.environ.get("LCRAG_RETRIEVER", "tfidf").lower()

    if backend == "chroma":
        # Open-source embedded vector DB (Chroma) + OpenAI embeddings. Needs a key.
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        store = Chroma.from_documents(chunks, OpenAIEmbeddings())
        return store.as_retriever(search_kwargs={"k": TOP_K}), f"Chroma + OpenAI embeddings ({len(chunks)} chunks)"

    # Default: LangChain's TF-IDF retriever — pure scikit-learn, no model, no key, $0.
    from langchain_community.retrievers import TFIDFRetriever

    retriever = TFIDFRetriever.from_documents(chunks)
    retriever.k = TOP_K
    return retriever, f"TFIDFRetriever ({len(chunks)} chunks, scikit-learn, $0)"


def get_llm():
    which = os.environ.get("LCRAG_LLM", "none").lower()
    if which == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.environ.get("LCRAG_OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    if which == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=os.environ.get("LCRAG_CLAUDE_MODEL", "claude-sonnet-4-6"), temperature=0)
    if which == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=os.environ.get("LCRAG_OLLAMA_MODEL", "llama3.1:8b"), temperature=0)
    return None


def answer(question: str):
    docs = load_documents()
    retriever, retr_desc = build_retriever(docs)
    llm = get_llm()

    if llm is not None:
        # Canonical LangChain RAG: retriever -> stuff docs into prompt -> LLM.
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain

        combine = create_stuff_documents_chain(llm, PROMPT, document_prompt=DOC_PROMPT)
        chain = create_retrieval_chain(retriever, combine)
        out = chain.invoke({"input": question})
        return out["answer"], out["context"], retr_desc

    # $0 fallback (no LLM key): return the retrieved evidence with PMID citations.
    hits = retriever.invoke(question)
    lines = [
        "(no LLM configured — set LCRAG_LLM=openai|anthropic|ollama for a synthesized answer)",
        "",
        "Top retrieved evidence:",
    ]
    for d in hits:
        title = d.metadata.get("title") or d.page_content[:80]
        lines.append(f"  - [PMID:{d.metadata.get('pmid','n/a')}] {title[:90]}")
    return "\n".join(lines), hits, retr_desc


def main() -> None:
    ap = argparse.ArgumentParser(description="LangChain RAG over the biosensor-literature corpus.")
    ap.add_argument("question", help="natural-language question")
    args = ap.parse_args()

    ans, ctx, retr_desc = answer(args.question)
    print("=" * 78)
    print(f"retriever : {retr_desc}")
    print(f"llm       : {os.environ.get('LCRAG_LLM', 'none')}")
    print("=" * 78)
    print(f"Q: {args.question}\n")
    print(ans)
    pmids = sorted({c.metadata.get("pmid", "n/a") for c in ctx})
    print("\nSources (PMID):", ", ".join(pmids))


if __name__ == "__main__":
    main()
