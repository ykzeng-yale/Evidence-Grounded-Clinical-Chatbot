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
    return {"status": "ok", "model": pipeline.llm.model}


@app.get("/", response_class=HTMLResponse)
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

    return EventSourceResponse(event_generator())


# --- Vercel Python serverless adapter ---
# Vercel discovers an ASGI `app` symbol in api/main.py; nothing else needed.
