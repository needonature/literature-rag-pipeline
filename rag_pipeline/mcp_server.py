"""MCP server — exposes the biosensor RAG as agent-callable tools (Model Context Protocol).

Turns the retrieval + RAG pipeline into Model Context Protocol tools, so any MCP
client (Claude Desktop, an agent loop, etc.) can call literature search and
grounded Q&A as a step inside a multi-step workflow — function-calling over your
own corpus, via the open MCP standard.

Setup:
    pip install -r requirements.txt -r requirements-mcp.txt   # needs Python >= 3.10
    python build_index.py --no-fact-cards                     # build the index once
    python mcp_server.py                                      # run the stdio MCP server

Register with an MCP client, e.g. Claude Desktop (claude_desktop_config.json):
    "mcpServers": {
      "biosensor-rag": { "command": "python", "args": ["/ABS/PATH/rag_pipeline/mcp_server.py"] }
    }

Tools exposed:
    search_biosensor_literature(query, k)   -> top-k abstracts + PMID citations + scores
    answer_biosensor_question(question, k)  -> grounded, PMID-cited answer + usage
"""
from __future__ import annotations

import os
import sys

# Make the server runnable from any cwd (an MCP client launches it by absolute path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP  # noqa: E402

import config  # noqa: E402,F401  (kept for env/config side effects + discoverability)
from rag import RagPipeline  # noqa: E402

mcp = FastMCP("biosensor-rag")

_PIPE: "RagPipeline | None" = None


def _pipeline() -> RagPipeline:
    global _PIPE
    if _PIPE is None:
        _PIPE = RagPipeline.from_index()  # loads the vector index + embedder + LLM backend
    return _PIPE


@mcp.tool()
def search_biosensor_literature(query: str, k: int = 6) -> str:
    """Semantic search over the curated wearable-biosensor PubMed corpus.

    Returns the top-k most relevant abstracts, each with its PMID citation, a
    relevance score, and a title/snippet — for grounding an answer or choosing sources.
    """
    contexts = _pipeline().retriever.retrieve(query, k=k)
    if not contexts:
        return "No matching passages in the indexed corpus."
    lines = []
    for c in contexts:
        title = c.metadata.get("title") or c.text[:90].replace("\n", " ")
        lines.append(f"[{c.citation}] (score {c.score:.3f}) {title}")
    return "\n".join(lines)


@mcp.tool()
def answer_biosensor_question(question: str, k: int = 6) -> str:
    """Answer a wearable-biosensor question with retrieval-augmented generation.

    Retrieves supporting abstracts and returns a grounded answer with inline
    [PMID:...] citations, plus a usage line (retrieval / generation latency).
    """
    res = _pipeline().answer(question, k=k)
    sources = ", ".join(res.citations) if res.citations else "(none)"
    u = res.usage
    usage = (
        f"retrieval={u.get('retrieval_ms')}ms generation={u.get('generation_ms')}ms "
        f"n_ctx={u.get('n_contexts')}"
    )
    return f"{res.answer}\n\nSources: {sources}\nUsage: {usage}"


if __name__ == "__main__":
    mcp.run()
