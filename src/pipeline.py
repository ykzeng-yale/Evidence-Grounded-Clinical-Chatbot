"""End-to-end orchestration:

   question
     → reformulate (LLM)
     → parallel retrieve (PubMed + ClinicalTrials.gov + EuropePMC + optional Tavily)
     → aggregate + dedupe
     → synthesize (Claude, structured tool_use)
     → AnswerResponse

Streaming variant `stream_ask` yields SSE-friendly events for the web UI:
   {"type": "status", "stage": "retrieval_start"}
   {"type": "evidence", "evidence": [...]}
   {"type": "answer_delta", "text_partial": "..."}
   {"type": "done", "response": {...}}
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator

from src.cache import get_cached, set_cached
from src.config import settings
from src.models import AnswerResponse, Citation, Evidence
from src.retrievers.clinicaltrials import search_clinicaltrials
from src.retrievers.europepmc import search_europepmc
from src.retrievers.pubmed import search_pubmed
from src.retrievers.web import search_web
from src.safety import DISCLAIMER, needs_individual_advice_refusal
from src.synthesis.llm import ClaudeSynthesizer

log = logging.getLogger(__name__)


def _dedupe(evidence: list[Evidence]) -> list[Evidence]:
    """Keep first occurrence of each citation id; preserves source ordering."""
    seen: set[str] = set()
    out: list[Evidence] = []
    for ev in evidence:
        if ev.id in seen:
            continue
        seen.add(ev.id)
        out.append(ev)
    return out


def _build_citations(evidence: list[Evidence], key_evidence: list[dict]) -> list[Citation]:
    """Promote LLM-cited evidence to top, then append the rest."""
    cited_ids = {ke.get("citation_id") for ke in key_evidence if ke.get("citation_id")}
    summary_by_id = {ke["citation_id"]: ke.get("summary") for ke in key_evidence if ke.get("citation_id")}
    head: list[Citation] = []
    tail: list[Citation] = []
    for ev in evidence:
        c = Citation(
            id=ev.id, title=ev.title, url=ev.url, source=ev.source,
            summary=summary_by_id.get(ev.id),
        )
        (head if ev.id in cited_ids else tail).append(c)
    return head + tail


class Pipeline:
    def __init__(self):
        self.llm = ClaudeSynthesizer(settings.anthropic_api_key, settings.anthropic_model)

    # ---------- Non-streaming (CLI / API JSON) ----------
    async def ask(
        self,
        question: str,
        max_pubmed: int | None = None,
        max_trials: int | None = None,
        max_preprints: int | None = None,
        use_web_search: bool | None = None,
        use_cache: bool = True,
    ) -> AnswerResponse:
        question = question.strip()
        if not question:
            raise ValueError("question is required")

        # Safety: refuse individualized medical advice early.
        if needs_individual_advice_refusal(question):
            return AnswerResponse(
                question=question,
                answer=(
                    "I can't provide individualized medical recommendations. "
                    "Please consult a qualified clinician about your specific situation. "
                    "I can summarize published evidence on related topics if you reframe "
                    "the question (e.g., 'What is the evidence for X in condition Y?')."
                ),
                limitations="Refused due to individualized-advice request.",
                confidence="low",
                disclaimer=DISCLAIMER,
            )

        if use_cache:
            cached = get_cached(question)
            if cached:
                cached.metadata["from_cache"] = True
                return cached

        max_pubmed = max_pubmed or settings.max_pubmed
        max_trials = max_trials or settings.max_trials
        max_preprints = max_preprints if max_preprints is not None else settings.max_preprints
        use_web = settings.use_web_search if use_web_search is None else use_web_search

        t0 = time.perf_counter()
        # Step 1: reformulate
        reform = await self.llm.reformulate(question)
        pubmed_q = reform.get("pubmed_query") or question
        trials_q = reform.get("trials_query") or question

        # Step 2: parallel retrieve
        tasks = [
            search_pubmed(pubmed_q, max_pubmed, settings.ncbi_api_key, settings.pubmed_email),
            search_clinicaltrials(trials_q, max_trials),
            search_europepmc(pubmed_q, max_preprints, prefer_preprints=True),
        ]
        if use_web and settings.tavily_api_key:
            tasks.append(search_web(question, settings.tavily_api_key, max_results=3))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        evidence: list[Evidence] = []
        for r in results:
            if isinstance(r, Exception):
                log.warning(f"retriever failed: {r}")
                continue
            evidence.extend(r)
        evidence = _dedupe(evidence)
        t_retrieval = time.perf_counter() - t0

        # Step 3: synthesize
        t1 = time.perf_counter()
        synth = await self.llm.synthesize(question, evidence)
        t_synth = time.perf_counter() - t1

        response = AnswerResponse(
            question=question,
            answer=synth.get("answer", ""),
            citations=_build_citations(evidence, synth.get("key_evidence", [])),
            evidence=evidence,
            limitations=synth.get("limitations", ""),
            confidence=synth.get("confidence", "moderate"),
            disclaimer=DISCLAIMER,
            metadata={
                "intent": reform.get("intent"),
                "pubmed_query": pubmed_q,
                "trials_query": trials_q,
                "n_pubmed": sum(1 for e in evidence if e.source == "PubMed"),
                "n_trials": sum(1 for e in evidence if e.source == "ClinicalTrials.gov"),
                "n_preprints": sum(1 for e in evidence if e.source == "EuropePMC"),
                "n_web": sum(1 for e in evidence if e.source == "WebSearch"),
                "retrieval_seconds": round(t_retrieval, 2),
                "synthesis_seconds": round(t_synth, 2),
                "total_seconds": round(t_retrieval + t_synth, 2),
                "model": settings.anthropic_model,
                "from_cache": False,
            },
        )
        if use_cache:
            set_cached(question, response)
        return response

    # ---------- Streaming variant for SSE ----------
    async def stream_ask(
        self,
        question: str,
        max_pubmed: int | None = None,
        max_trials: int | None = None,
        max_preprints: int | None = None,
        use_web_search: bool | None = None,
        use_cache: bool = True,
    ) -> AsyncIterator[dict]:
        question = question.strip()
        if not question:
            yield {"type": "error", "message": "question is required"}
            return

        if needs_individual_advice_refusal(question):
            refusal = AnswerResponse(
                question=question,
                answer=(
                    "I can't provide individualized medical recommendations. "
                    "Please consult a qualified clinician. I can summarize the published "
                    "evidence on related topics if you reframe the question."
                ),
                limitations="Refused due to individualized-advice request.",
                confidence="low",
                disclaimer=DISCLAIMER,
            )
            yield {"type": "done", "response": refusal.model_dump()}
            return

        if use_cache:
            cached = get_cached(question)
            if cached:
                cached.metadata["from_cache"] = True
                yield {"type": "status", "stage": "cache_hit"}
                yield {"type": "evidence", "evidence": [e.model_dump() for e in cached.evidence]}
                # stream the cached answer in small chunks for UX consistency
                txt = cached.answer
                step = max(20, len(txt) // 60)
                for i in range(0, len(txt), step):
                    yield {"type": "answer_delta", "text_partial": txt[: i + step]}
                yield {"type": "done", "response": cached.model_dump()}
                return

        yield {"type": "status", "stage": "reformulating"}
        reform = await self.llm.reformulate(question)
        pubmed_q = reform.get("pubmed_query") or question
        trials_q = reform.get("trials_query") or question
        yield {"type": "queries", "pubmed_query": pubmed_q, "trials_query": trials_q,
               "intent": reform.get("intent")}

        max_pubmed = max_pubmed or settings.max_pubmed
        max_trials = max_trials or settings.max_trials
        max_preprints = max_preprints if max_preprints is not None else settings.max_preprints
        use_web = settings.use_web_search if use_web_search is None else use_web_search

        yield {"type": "status", "stage": "retrieving",
               "sources": ["PubMed", "ClinicalTrials.gov", "EuropePMC"] +
                          (["WebSearch"] if use_web else [])}

        t0 = time.perf_counter()
        tasks = [
            search_pubmed(pubmed_q, max_pubmed, settings.ncbi_api_key, settings.pubmed_email),
            search_clinicaltrials(trials_q, max_trials),
            search_europepmc(pubmed_q, max_preprints, prefer_preprints=True),
        ]
        if use_web and settings.tavily_api_key:
            tasks.append(search_web(question, settings.tavily_api_key, max_results=3))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        evidence: list[Evidence] = []
        for r in results:
            if isinstance(r, Exception):
                log.warning(f"retriever failed: {r}")
                continue
            evidence.extend(r)
        evidence = _dedupe(evidence)
        t_retrieval = time.perf_counter() - t0

        yield {"type": "evidence", "evidence": [e.model_dump() for e in evidence],
               "retrieval_seconds": round(t_retrieval, 2)}

        if not evidence:
            empty = AnswerResponse(
                question=question,
                answer=(
                    "No evidence was retrieved from PubMed, ClinicalTrials.gov, or "
                    "Europe PMC for this question. Please try a more specific term."
                ),
                limitations="Empty evidence set; cannot answer.",
                confidence="low",
                disclaimer=DISCLAIMER,
            )
            yield {"type": "done", "response": empty.model_dump()}
            return

        yield {"type": "status", "stage": "synthesizing"}
        t1 = time.perf_counter()
        structured: dict | None = None
        async for ev in self.llm.synthesize_stream(question, evidence):
            if ev["type"] == "text_delta":
                if "text_partial" in ev:
                    yield {"type": "answer_delta", "text_partial": ev["text_partial"]}
                else:
                    yield {"type": "answer_delta", "text": ev.get("text", "")}
            elif ev["type"] == "structured":
                structured = ev["data"]
        t_synth = time.perf_counter() - t1

        if structured is None:
            structured = {
                "answer": "(synthesis returned no structured output)",
                "key_evidence": [],
                "limitations": "Model failed to return structured output.",
                "confidence": "low",
            }

        response = AnswerResponse(
            question=question,
            answer=structured.get("answer", ""),
            citations=_build_citations(evidence, structured.get("key_evidence", [])),
            evidence=evidence,
            limitations=structured.get("limitations", ""),
            confidence=structured.get("confidence", "moderate"),
            disclaimer=DISCLAIMER,
            metadata={
                "intent": reform.get("intent"),
                "pubmed_query": pubmed_q,
                "trials_query": trials_q,
                "n_pubmed": sum(1 for e in evidence if e.source == "PubMed"),
                "n_trials": sum(1 for e in evidence if e.source == "ClinicalTrials.gov"),
                "n_preprints": sum(1 for e in evidence if e.source == "EuropePMC"),
                "n_web": sum(1 for e in evidence if e.source == "WebSearch"),
                "retrieval_seconds": round(t_retrieval, 2),
                "synthesis_seconds": round(t_synth, 2),
                "total_seconds": round(t_retrieval + t_synth, 2),
                "model": settings.anthropic_model,
                "from_cache": False,
            },
        )
        if use_cache:
            set_cached(question, response)
        yield {"type": "done", "response": response.model_dump()}
