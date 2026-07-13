"""Flask web app — the 'internal AI-powered tool' front door.

    python app.py            # http://localhost:8080
    POST /ask  {"question": "...", "k": 6, "filters": {"biofluid": "Sweat"}}
    GET  /healthz

Containerize with cloud/Dockerfile and deploy to Cloud Run (see cloud/DEPLOY_GCP.md).
The pipeline is loaded once at startup; the generation backend is whatever
RAG_GEN_BACKEND points to (default: free extractive).
"""
from __future__ import annotations

import os

from flask import Flask, jsonify, request

import config
from rag import RagPipeline

app = Flask(__name__)
_PIPE = None


def pipe() -> RagPipeline:
    global _PIPE
    if _PIPE is None:
        _PIPE = RagPipeline.from_index()
    return _PIPE


INDEX_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>Biosensor Literature RAG</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:820px;margin:40px auto;padding:0 16px;color:#1a1a2e}
 h1{font-size:20px} .sub{color:#667;margin-bottom:20px}
 textarea{width:100%;height:64px;padding:10px;font-size:15px;border:1px solid #ccd;border-radius:8px}
 button{margin-top:10px;padding:9px 18px;font-size:15px;background:#2d6cdf;color:#fff;border:0;border-radius:8px;cursor:pointer}
 .ans{white-space:pre-wrap;background:#f6f8fc;border:1px solid #e3e8f5;border-radius:8px;padding:14px;margin-top:18px}
 .cite{color:#2d6cdf} .ev{font-size:13px;color:#445;margin-top:6px;border-left:3px solid #cdd;padding-left:8px}
 .u{font-size:12px;color:#889;margin-top:10px}
</style></head><body>
<h1>Wearable Biosensor Literature — RAG Assistant</h1>
<div class="sub">Grounded answers over the curated PubMed biosensor corpus, with PMID citations.</div>
<textarea id="q" placeholder="e.g. Which wearable sensors measure cortisol in sweat?"></textarea>
<div><button onclick="ask()">Ask</button></div>
<div id="out"></div>
<script>
async function ask(){
 const q=document.getElementById('q').value; const out=document.getElementById('out');
 out.innerHTML='<div class="u">thinking…</div>';
 const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
 const d=await r.json();
 let ev=(d.contexts||[]).map(c=>`<div class="ev">[${c.pmid?('PMID:'+c.pmid):c.doc_id}] ${c.title||c.biomarker||''} — ${c.text}</div>`).join('');
 out.innerHTML=`<div class="ans">${(d.answer||'').replace(/\\[(PMID:[0-9]+)\\]/g,'<span class=cite>[$1]</span>')}</div>`
  +`<div class="u">citations: ${(d.citations||[]).join(', ')||'none'} · retrieval ${d.usage?.retrieval_ms||0}ms · backend ${d.usage?.backend||'extractive'} · est_cost $${d.usage?.est_cost_usd||0}</div>`
  +ev;
}
</script></body></html>
"""


@app.route("/")
def home():
    return INDEX_HTML


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "embed_backend": config.EMBED_BACKEND, "gen_backend": config.GEN_BACKEND})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    k = data.get("k")
    filters = data.get("filters")
    result = pipe().answer(question, k=k, filters=filters)
    return jsonify(result.to_dict())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
