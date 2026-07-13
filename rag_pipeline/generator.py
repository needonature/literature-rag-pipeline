"""Pluggable generation backends — the "G" in RAG.

Default = ExtractiveGenerator: NO LLM, NO cost. It stitches a grounded answer
from the retrieved sentences and cites PMIDs, so the whole pipeline runs for $0
(important: you asked not to spend new API money — this honors that, and is also
perfect for CI / eval).

Flip RAG_GEN_BACKEND to get real generation:
    ollama  -> local LLM, free, genuine generation (recommended for a live demo)
    claude  -> Anthropic Claude  (needs ANTHROPIC_API_KEY)
    openai  -> OpenAI            (needs OPENAI_API_KEY; the account you already use)
    gemini  -> Google Gemini     (needs GOOGLE_API_KEY)  [GCP/Vertex variant in cloud/]

Every backend shares the same grounded prompt and returns the same dict shape,
so swapping models is a one-line config change.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import config
from schema import RetrievedChunk
from text_utils import tokenize

SYSTEM_PROMPT = (
    "You are a careful biomedical literature assistant for a public-health team. "
    "Answer the QUESTION using ONLY the numbered CONTEXT passages. "
    "Cite every claim inline with its source tag, e.g. [PMID:36705589]. "
    "If the context does not contain the answer, say so plainly instead of guessing. "
    "Be concise and factual; prefer the curated biomarker/biosensor facts when present."
)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _source_tag(c: RetrievedChunk) -> str:
    return c.citation  # PMID:xxx or DOC:xxx


def build_context_block(contexts: List[RetrievedChunk], max_chars: int) -> str:
    lines, total = [], 0
    for i, c in enumerate(contexts, 1):
        head_bits = [f"[{_source_tag(c)}]"]
        if c.metadata.get("title"):
            head_bits.append(c.metadata["title"])
        elif c.metadata.get("biomarker"):
            head_bits.append(f"{c.metadata.get('biosensor_principle','')} / {c.metadata.get('biomarker','')}")
        block = f"({i}) {' '.join(b for b in head_bits if b)}\n{c.text.strip()}"
        if total + len(block) > max_chars and lines:
            break
        lines.append(block)
        total += len(block)
    return "\n\n".join(lines)


def build_prompt(question: str, contexts: List[RetrievedChunk], max_chars: int) -> str:
    ctx = build_context_block(contexts, max_chars)
    return f"CONTEXT:\n{ctx}\n\nQUESTION: {question}\n\nGrounded answer with inline [PMID:...] citations:"


def count_tokens(text: str, model: str = "") -> int:
    try:
        import tiktoken  # lazy

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001
        return max(1, len(text) // 4)  # rough heuristic


def estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    p = config.PRICING.get(model)
    if not p:
        return 0.0
    return (in_tok / 1_000_000) * p["in"] + (out_tok / 1_000_000) * p["out"]


def cited_pmids(contexts: List[RetrievedChunk]) -> List[str]:
    out, seen = [], set()
    for c in contexts:
        tag = _source_tag(c)
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


_CITE_RE = re.compile(r"PMID:\s*(\d+)", re.IGNORECASE)


def parse_cited_tags(answer: str) -> List[str]:
    """Extract the citation tags the model ACTUALLY wrote in its answer (in order,
    deduped). Using this instead of cited_pmids(contexts) makes
    citation_faithfulness a real test: it can now catch a model that cites a
    PMID it was never given (hallucinated / from training memory)."""
    seen, out = set(), []
    for m in _CITE_RE.finditer(answer or ""):
        tag = f"PMID:{m.group(1)}"
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
class BaseGenerator:
    name = "base"
    model = "_local"

    def generate(self, question: str, contexts: List[RetrievedChunk]) -> Dict[str, Any]:
        raise NotImplementedError

    def _usage(self, prompt: str, answer: str) -> Dict[str, Any]:
        in_tok = count_tokens(prompt, self.model)
        out_tok = count_tokens(answer, self.model)
        return {
            "backend": self.name,
            "model": self.model,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "est_cost_usd": round(estimate_cost(self.model, in_tok, out_tok), 6),
        }


# --------------------------------------------------------------------------- #
# Extractive (default, free)
# --------------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


class ExtractiveGenerator(BaseGenerator):
    name = "extractive"
    model = "_extractive"

    def generate(self, question: str, contexts: List[RetrievedChunk]) -> Dict[str, Any]:
        q_tokens = set(tokenize(question))
        scored = []
        for c in contexts:
            tag = _source_tag(c)
            for sent in _SENT_SPLIT.split(c.text.strip()):
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                overlap = len(q_tokens & set(tokenize(sent)))
                scored.append((overlap + 0.25 * c.score, sent, tag))
        scored.sort(key=lambda x: x[0], reverse=True)

        picked, used_tags, seen_sent = [], [], set()
        for _, sent, tag in scored:
            if sent in seen_sent:
                continue
            seen_sent.add(sent)
            picked.append(f"{sent} [{tag}]")
            if tag not in used_tags:
                used_tags.append(tag)
            if len(picked) >= 4:
                break

        if not picked:
            answer = "The retrieved passages do not contain enough information to answer this question."
            used_tags = []
        else:
            answer = (
                "Based on the retrieved literature:\n- "
                + "\n- ".join(picked)
            )
        prompt = build_prompt(question, contexts, config.MAX_CONTEXT_CHARS)
        return {"answer": answer, "citations": used_tags or cited_pmids(contexts), "usage": self._usage(prompt, answer)}


# --------------------------------------------------------------------------- #
# Ollama (local LLM, free, real generation)
# --------------------------------------------------------------------------- #
class OllamaGenerator(BaseGenerator):
    name = "ollama"

    def __init__(self, model: Optional[str] = None):
        self.model = model or config.OLLAMA_MODEL

    def generate(self, question: str, contexts: List[RetrievedChunk]) -> Dict[str, Any]:
        import json
        import urllib.request

        prompt = build_prompt(question, contexts, config.MAX_CONTEXT_CHARS)
        body = json.dumps(
            {
                "model": self.model,
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode()
        url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/generate"
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        answer = data.get("response", "").strip()
        usage = self._usage(prompt, answer)
        usage["model"] = self.model
        return {"answer": answer, "citations": parse_cited_tags(answer), "usage": usage}


# --------------------------------------------------------------------------- #
# Anthropic Claude
# --------------------------------------------------------------------------- #
class ClaudeGenerator(BaseGenerator):
    name = "claude"

    def __init__(self, model: Optional[str] = None):
        from anthropic import Anthropic  # lazy

        self.model = model or config.CLAUDE_MODEL
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY

    def generate(self, question: str, contexts: List[RetrievedChunk]) -> Dict[str, Any]:
        prompt = build_prompt(question, contexts, config.MAX_CONTEXT_CHARS)
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=700,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        usage = self._usage(prompt, answer)
        if getattr(msg, "usage", None):  # prefer real provider counts
            usage["input_tokens"] = msg.usage.input_tokens
            usage["output_tokens"] = msg.usage.output_tokens
            usage["est_cost_usd"] = round(
                estimate_cost(self.model, msg.usage.input_tokens, msg.usage.output_tokens), 6
            )
        return {"answer": answer, "citations": parse_cited_tags(answer), "usage": usage}


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
class OpenAIGenerator(BaseGenerator):
    name = "openai"

    def __init__(self, model: Optional[str] = None):
        from openai import OpenAI  # lazy

        self.model = model or config.OPENAI_GEN_MODEL
        self.client = OpenAI()  # reads OPENAI_API_KEY

    def generate(self, question: str, contexts: List[RetrievedChunk]) -> Dict[str, Any]:
        prompt = build_prompt(question, contexts, config.MAX_CONTEXT_CHARS)
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        answer = (resp.choices[0].message.content or "").strip()
        usage = self._usage(prompt, answer)
        if getattr(resp, "usage", None):
            usage["input_tokens"] = resp.usage.prompt_tokens
            usage["output_tokens"] = resp.usage.completion_tokens
            usage["est_cost_usd"] = round(
                estimate_cost(self.model, resp.usage.prompt_tokens, resp.usage.completion_tokens), 6
            )
        return {"answer": answer, "citations": parse_cited_tags(answer), "usage": usage}


# --------------------------------------------------------------------------- #
# Gemini (Google AI Studio key; the Vertex variant lives in cloud/gcp_vertex.py)
# --------------------------------------------------------------------------- #
class GeminiGenerator(BaseGenerator):
    name = "gemini"

    def __init__(self, model: Optional[str] = None):
        import google.generativeai as genai  # lazy

        self.model_name = model or config.GEMINI_MODEL
        self.model = self.model_name
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self._client = genai.GenerativeModel(self.model_name, system_instruction=SYSTEM_PROMPT)

    def generate(self, question: str, contexts: List[RetrievedChunk]) -> Dict[str, Any]:
        prompt = build_prompt(question, contexts, config.MAX_CONTEXT_CHARS)
        resp = self._client.generate_content(prompt, generation_config={"temperature": 0})
        answer = (resp.text or "").strip()
        return {"answer": answer, "citations": parse_cited_tags(answer), "usage": self._usage(prompt, answer)}


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def get_generator(backend: Optional[str] = None) -> BaseGenerator:
    backend = (backend or config.GEN_BACKEND).lower()
    if backend in ("extractive", "none", "free"):
        return ExtractiveGenerator()
    if backend == "ollama":
        return OllamaGenerator()
    if backend == "claude":
        return ClaudeGenerator()
    if backend == "openai":
        return OpenAIGenerator()
    if backend == "gemini":
        return GeminiGenerator()
    if backend in ("vertex", "vertex_gemini"):
        from cloud.gcp_vertex import VertexGeminiGenerator  # lazy; Gemini on Vertex AI (uses ADC, no API key)

        return VertexGeminiGenerator()
    if backend == "bedrock":
        from cloud.aws_bedrock import BedrockClaudeGenerator  # lazy; see cloud/aws_bedrock.py

        return BedrockClaudeGenerator()
    print(f"[generator] unknown backend '{backend}', using free extractive generator.")
    return ExtractiveGenerator()
