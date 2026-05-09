"""FastAPI service with non-streaming JSON endpoint and SSE streaming endpoint.

Endpoints:
  GET  /                        → static frontend (frontend/index.html)
  GET  /health                  → liveness
  POST /api/ask                 → JSON answer (cached + structured)
  GET  /api/ask/stream?q=...    → text/event-stream of incremental events
  GET  /api/sample-questions    → sample question bank for the UI
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from src.models import AnswerResponse, AskRequest
from src.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

app = FastAPI(
    title="Evidence-Grounded Clinical Chatbot",
    description="Answers clinical questions using PubMed, ClinicalTrials.gov, and Europe PMC, synthesized by Claude.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = Pipeline()
ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend" / "index.html"
SAMPLES = ROOT / "examples" / "sample_questions.json"


@app.get("/health")
async def health():
    """Lightweight liveness check — returns 200 if the process is up.
    For dependency status (Anthropic / Supabase / retrievers) hit /health/deep."""
    return {"status": "ok", "model": pipeline.llm.model}


@app.get("/health/deep")
async def health_deep():
    """Active probe of all upstream dependencies. Useful for uptime monitors
    that want to know not just 'process alive' but 'system functional'.
    Each probe has a 4s timeout so total response stays <12s in worst case."""
    import asyncio as _a
    import httpx as _h
    from src.cache import _get_client
    from src.config import settings as _s

    async def probe_pubmed() -> dict:
        try:
            async with _h.AsyncClient(timeout=4) as c:
                r = await c.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi",
                                params={"retmode": "json", "tool": "egcc",
                                        "email": _s.pubmed_email})
                return {"ok": r.status_code == 200, "status": r.status_code}
        except Exception as e:
            return {"ok": False, "error": str(e)[:80]}

    async def probe_clinicaltrials() -> dict:
        try:
            async with _h.AsyncClient(timeout=4) as c:
                r = await c.get("https://clinicaltrials.gov/api/v2/studies",
                                params={"pageSize": 1, "format": "json"})
                return {"ok": r.status_code == 200, "status": r.status_code}
        except Exception as e:
            return {"ok": False, "error": str(e)[:80]}

    async def probe_paperclip() -> dict:
        if not _s.paperclip_api_key:
            return {"ok": False, "error": "no api key"}
        try:
            async with _h.AsyncClient(timeout=4) as c:
                r = await c.post("https://paperclip.gxl.ai/mcp",
                                 headers={"X-API-Key": _s.paperclip_api_key,
                                          "Accept": "application/json, text/event-stream",
                                          "Content-Type": "application/json"},
                                 json={"jsonrpc": "2.0", "id": 1,
                                       "method": "tools/list", "params": {}})
                return {"ok": r.status_code == 200, "status": r.status_code}
        except Exception as e:
            return {"ok": False, "error": str(e)[:80]}

    async def probe_anthropic() -> dict:
        try:
            async with _h.AsyncClient(timeout=4) as c:
                r = await c.get("https://api.anthropic.com/v1/messages",
                                headers={"x-api-key": _s.anthropic_api_key,
                                         "anthropic-version": "2023-06-01"})
                # 401/405 means reachable + auth working but wrong method/no body — both fine for liveness
                return {"ok": r.status_code in (200, 400, 401, 405), "status": r.status_code}
        except Exception as e:
            return {"ok": False, "error": str(e)[:80]}

    def probe_supabase() -> dict:
        sb = _get_client()
        if not sb:
            return {"ok": False, "error": "not configured"}
        try:
            sb.table("query_cache").select("question_hash").limit(1).execute()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)[:80]}

    pm, ct, pc, an = await _a.gather(
        probe_pubmed(), probe_clinicaltrials(), probe_paperclip(), probe_anthropic(),
    )
    sb = await _a.to_thread(probe_supabase)

    deps = {"pubmed": pm, "clinicaltrials": ct, "paperclip": pc, "anthropic": an, "supabase": sb}
    overall_ok = all(d.get("ok", False) for k, d in deps.items() if k in ("pubmed", "clinicaltrials", "anthropic"))
    return {
        "status": "ok" if overall_ok else "degraded",
        "model": pipeline.llm.model,
        "deps": deps,
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Favicon is inlined as data: URL in index.html. This endpoint exists
    only to silence the 204-vs-404 noise when browsers request /favicon.ico."""
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/robots.txt", include_in_schema=False)
async def robots():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("User-agent: *\nAllow: /\n")


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def index():
    if FRONTEND.exists():
        return HTMLResponse(FRONTEND.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Evidence-Grounded Clinical Chatbot</h1><p>POST /api/ask</p>")


@app.get("/api/sample-questions")
async def sample_questions():
    if SAMPLES.exists():
        return JSONResponse(json.loads(SAMPLES.read_text()))
    return JSONResponse([])


@app.post("/api/ask", response_model=AnswerResponse)
async def ask(req: AskRequest):
    try:
        return await pipeline.ask(
            question=req.question,
            max_pubmed=req.max_pubmed,
            max_trials=req.max_trials,
            max_preprints=req.max_preprints,
            max_paperclip=req.max_paperclip,
            use_web_search=req.use_web_search,
            rerank=req.rerank,
            deepen=req.deepen,
            use_cache=req.use_cache,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logging.exception("ask failed")
        raise HTTPException(500, f"internal error: {e}")


@app.get("/api/ask/stream")
async def ask_stream(
    request: Request,
    q: str = Query(..., min_length=3, description="Clinical question"),
    max_pubmed: int | None = Query(None, ge=1, le=20),
    max_trials: int | None = Query(None, ge=0, le=20),
    max_preprints: int | None = Query(None, ge=0, le=10),
    max_paperclip: int | None = Query(None, ge=0, le=10),
    use_web_search: bool | None = Query(None),
    rerank: bool | None = Query(None),
    deepen: bool | None = Query(None),
    use_cache: bool = Query(True),
):
    """Server-Sent Events stream. Each event has shape: {type, ...payload}."""

    async def event_generator():
        try:
            async for ev in pipeline.stream_ask(
                question=q,
                max_pubmed=max_pubmed,
                max_trials=max_trials,
                max_preprints=max_preprints,
                max_paperclip=max_paperclip,
                use_web_search=use_web_search,
                rerank=rerank,
                deepen=deepen,
                use_cache=use_cache,
            ):
                if await request.is_disconnected():
                    break
                yield {"event": ev["type"], "data": json.dumps(ev, ensure_ascii=False)}
        except Exception as e:
            logging.exception("stream_ask failed")
            yield {"event": "error", "data": json.dumps({"type": "error", "message": str(e)})}

    # Streaming notes:
    # - Locally and on hosts with true streaming (Render / Fly / EC2),
    #   answer_delta events arrive as Claude generates them.
    # - On Vercel Python serverless, the response body is buffered until the
    #   function returns. Status / queries / evidence events still stream
    #   (separated by 100ms+ gaps), but answer_delta bursts at the end.
    # - X-Accel-Buffering / ping / padding tricks don't override this; only
    #   migrating to Vercel Edge (TypeScript) or another host fixes it.
    # - We keep the streaming endpoint anyway because:
    #   (a) early status updates DO stream and show progress
    #   (b) the JSON endpoint /api/ask is the canonical synchronous interface
    #   (c) self-hosted deployments get the full streaming UX for free
    return EventSourceResponse(
        event_generator(),
        ping=15,
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
        },
    )


# --- Vercel Python serverless adapter ---
# Vercel discovers an ASGI `app` symbol in api/main.py; nothing else needed.
