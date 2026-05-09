"""End-to-end orchestration with full RAG stack:

   question
     ─► [exact + semantic cache check]            (Supabase pgvector)
     ─► reformulate (Claude)
     ─► parallel retrieve:
            PubMed (E-utilities)
            ClinicalTrials.gov (v2)
            Europe PMC  (preprints)
            Paperclip   (8M+ corpus, MCP)
            Tavily      (optional, FDA / guidelines)
     ─► dedupe
     ─► rerank (Claude Haiku LLM-rerank)          ← Tier A
     ─► synthesize-pass-1 (Claude tool_use)
     ─► [if deepen_enabled] Firecrawl PMC full-text  ← Tier B
     ─► synthesize-pass-2 with deepened context
     ─► AnswerResponse
     ─► write to cache (exact + semantic)

The streaming variant `stream_ask` yields SSE-friendly events for the web UI.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator

from src.cache import get_cached, set_cached
from src.config import settings
from src.embeddings import configured_provider
from src.models import AnswerResponse, Citation, Evidence
from src.retrievers.clinicaltrials import search_clinicaltrials
from src.retrievers.europepmc import search_europepmc
from src.retrievers.firecrawl_deepen import deepen_top_citations
from src.retrievers.paperclip import search_paperclip
from src.retrievers.pubmed import search_pubmed
from src.retrievers.web import search_web
from src.safety import DISCLAIMER, needs_clinical_question_refusal, needs_individual_advice_refusal
from src.synthesis.llm import ClaudeSynthesizer
from src.synthesis.rerank import rerank_evidence

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

    # ---------- Non-streaming ----------
    async def ask(
        self,
        question: str,
        max_pubmed: int | None = None,
        max_trials: int | None = None,
        max_preprints: int | None = None,
        max_paperclip: int | None = None,
        use_web_search: bool | None = None,
        rerank: bool | None = None,
        deepen: bool | None = None,
        use_cache: bool = True,
    ) -> AnswerResponse:
        question = question.strip()
        if not question:
            raise ValueError("question is required")

        if needs_individual_advice_refusal(question):
            return _refusal_response(question, kind="individual_advice")
        if needs_clinical_question_refusal(question):
            return _refusal_response(question, kind="non_clinical")

        if use_cache:
            hit = await get_cached(question)
            if hit:
                cached, meta = hit
                cached.metadata = {**(cached.metadata or {}), "cache_hit": meta}
                return cached

        max_pubmed = max_pubmed or settings.max_pubmed
        max_trials = max_trials or settings.max_trials
        max_preprints = max_preprints if max_preprints is not None else settings.max_preprints
        max_paperclip = max_paperclip if max_paperclip is not None else settings.max_paperclip
        use_web = settings.use_web_search if use_web_search is None else use_web_search
        do_rerank = settings.rerank_enabled if rerank is None else rerank
        do_deepen = settings.deepen_enabled if deepen is None else deepen

        # ----- Reformulate -----
        t0 = time.perf_counter()
        reform = await self.llm.reformulate(question)
        pubmed_q = reform.get("pubmed_query") or question
        trials_q = reform.get("trials_query") or question

        # ----- Parallel retrieve -----
        evidence = await self._retrieve(
            pubmed_q, trials_q, question,
            max_pubmed=max_pubmed, max_trials=max_trials,
            max_preprints=max_preprints, max_paperclip=max_paperclip,
            use_web=use_web,
        )
        t_retrieval = time.perf_counter() - t0

        # ----- Rerank (Tier A) -----
        t1 = time.perf_counter()
        if do_rerank and len(evidence) > 1:
            evidence = await rerank_evidence(
                question, evidence,
                api_key=settings.anthropic_api_key,
                keep_top_k=None,  # keep all visible to user; synthesis trims via prompt
            )
        t_rerank = time.perf_counter() - t1

        # ----- Synthesis pass 1 -----
        t2 = time.perf_counter()
        synth_input = evidence[: settings.rerank_keep_top_k] if do_rerank else evidence
        synth = await self.llm.synthesize(question, synth_input)
        t_synth1 = time.perf_counter() - t2

        # ----- Tier B: full-text deepening + synth pass 2 -----
        t_deepen = 0.0
        t_synth2 = 0.0
        deepened_ids: list[str] = []
        if do_deepen and settings.firecrawl_api_key and synth.get("key_evidence"):
            t3 = time.perf_counter()
            deepened = await deepen_top_citations(
                citations=synth.get("key_evidence", []),
                evidence=evidence,
                api_key=settings.firecrawl_api_key,
                max_deepen=settings.deepen_top_k,
                pubmed_email=settings.pubmed_email,
            )
            t_deepen = time.perf_counter() - t3
            if deepened:
                deepened_ids = [d.id for d in deepened]
                # Replace original evidence entries with deepened versions for synthesis-2
                deepened_by_id = {d.id: d for d in deepened}
                synth_input2 = [deepened_by_id.get(e.id, e) for e in synth_input]
                t4 = time.perf_counter()
                synth = await self.llm.synthesize(question, synth_input2)
                t_synth2 = time.perf_counter() - t4

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
                "n_paperclip": sum(1 for e in evidence if e.source == "Paperclip"),
                "n_web": sum(1 for e in evidence if e.source == "WebSearch"),
                "retrieval_seconds": round(t_retrieval, 2),
                "rerank_seconds": round(t_rerank, 2),
                "synthesis_seconds": round(t_synth1 + t_synth2, 2),
                "deepen_seconds": round(t_deepen, 2),
                "total_seconds": round(t_retrieval + t_rerank + t_synth1 + t_deepen + t_synth2, 2),
                "deepened_citations": deepened_ids,
                "rerank_used": bool(do_rerank),
                "deepen_used": bool(do_deepen and deepened_ids),
                "model": settings.anthropic_model,
                "embedding_provider": configured_provider(),
                "cache_hit": None,
            },
        )
        if use_cache:
            await set_cached(question, response)
        return response

    # ---------- Retrieval helper ----------
    async def _retrieve(
        self, pubmed_q: str, trials_q: str, raw_q: str,
        *, max_pubmed: int, max_trials: int, max_preprints: int,
        max_paperclip: int, use_web: bool,
    ) -> list[Evidence]:
        tasks = [
            search_pubmed(pubmed_q, max_pubmed, settings.ncbi_api_key, settings.pubmed_email),
            search_clinicaltrials(trials_q, max_trials),
            search_europepmc(pubmed_q, max_preprints, prefer_preprints=True),
        ]
        if max_paperclip and settings.paperclip_api_key:
            tasks.append(search_paperclip(pubmed_q, max_paperclip, settings.paperclip_api_key))
        if use_web and settings.tavily_api_key:
            tasks.append(search_web(raw_q, settings.tavily_api_key, max_results=3))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        evidence: list[Evidence] = []
        for r in results:
            if isinstance(r, Exception):
                log.warning(f"retriever failed: {r}")
                continue
            evidence.extend(r)
        return _dedupe(evidence)

    # ---------- Streaming variant ----------
    async def stream_ask(
        self,
        question: str,
        max_pubmed: int | None = None,
        max_trials: int | None = None,
        max_preprints: int | None = None,
        max_paperclip: int | None = None,
        use_web_search: bool | None = None,
        rerank: bool | None = None,
        deepen: bool | None = None,
        use_cache: bool = True,
    ) -> AsyncIterator[dict]:
        question = question.strip()
        if not question:
            yield {"type": "error", "message": "question is required"}
            return

        if needs_individual_advice_refusal(question):
            yield {"type": "done", "response": _refusal_response(question, "individual_advice").model_dump()}
            return
        if needs_clinical_question_refusal(question):
            yield {"type": "done", "response": _refusal_response(question, "non_clinical").model_dump()}
            return

        if use_cache:
            hit = await get_cached(question)
            if hit:
                cached, meta = hit
                cached.metadata = {**(cached.metadata or {}), "cache_hit": meta}
                yield {"type": "status", "stage": f"cache_hit_{meta['tier']}",
                       "similarity": meta.get("similarity")}
                yield {"type": "evidence", "evidence": [e.model_dump() for e in cached.evidence]}
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
        max_paperclip = max_paperclip if max_paperclip is not None else settings.max_paperclip
        use_web = settings.use_web_search if use_web_search is None else use_web_search
        do_rerank = settings.rerank_enabled if rerank is None else rerank
        do_deepen = settings.deepen_enabled if deepen is None else deepen

        sources = ["PubMed", "ClinicalTrials.gov", "EuropePMC"]
        if max_paperclip and settings.paperclip_api_key:
            sources.append("Paperclip")
        if use_web and settings.tavily_api_key:
            sources.append("WebSearch")
        yield {"type": "status", "stage": "retrieving", "sources": sources}

        t0 = time.perf_counter()
        evidence = await self._retrieve(
            pubmed_q, trials_q, question,
            max_pubmed=max_pubmed, max_trials=max_trials,
            max_preprints=max_preprints, max_paperclip=max_paperclip,
            use_web=use_web,
        )
        t_retrieval = time.perf_counter() - t0

        yield {"type": "evidence", "evidence": [e.model_dump() for e in evidence],
               "retrieval_seconds": round(t_retrieval, 2)}

        if not evidence:
            empty = AnswerResponse(
                question=question,
                answer="No evidence retrieved. Try a more specific term.",
                limitations="Empty evidence set; cannot answer.",
                confidence="low",
                disclaimer=DISCLAIMER,
            )
            yield {"type": "done", "response": empty.model_dump()}
            return

        if do_rerank and len(evidence) > 1:
            yield {"type": "status", "stage": "reranking"}
            t1 = time.perf_counter()
            evidence = await rerank_evidence(
                question, evidence,
                api_key=settings.anthropic_api_key,
                keep_top_k=None,
            )
            yield {"type": "rerank_done", "rerank_seconds": round(time.perf_counter() - t1, 2),
                   "evidence_reordered": [e.model_dump() for e in evidence]}

        yield {"type": "status", "stage": "synthesizing"}
        t2 = time.perf_counter()
        synth_input = evidence[: settings.rerank_keep_top_k] if do_rerank else evidence
        structured: dict | None = None
        async for ev in self.llm.synthesize_stream(question, synth_input):
            if ev["type"] == "text_delta":
                if "text_partial" in ev:
                    yield {"type": "answer_delta", "text_partial": ev["text_partial"]}
                else:
                    yield {"type": "answer_delta", "text": ev.get("text", "")}
            elif ev["type"] == "structured":
                structured = ev["data"]
        t_synth1 = time.perf_counter() - t2
        if structured is None:
            structured = {"answer": "(no output)", "key_evidence": [], "limitations": "", "confidence": "low"}

        # ----- Tier B: deepen + synthesize-2 -----
        deepened_ids: list[str] = []
        t_deepen = 0.0
        t_synth2 = 0.0
        if do_deepen and settings.firecrawl_api_key and structured.get("key_evidence"):
            yield {"type": "status", "stage": "deepening_full_text"}
            t3 = time.perf_counter()
            deepened = await deepen_top_citations(
                citations=structured.get("key_evidence", []),
                evidence=evidence,
                api_key=settings.firecrawl_api_key,
                max_deepen=settings.deepen_top_k,
                pubmed_email=settings.pubmed_email,
            )
            t_deepen = time.perf_counter() - t3
            if deepened:
                deepened_ids = [d.id for d in deepened]
                yield {"type": "deepened", "deepened_citations": deepened_ids,
                       "deepen_seconds": round(t_deepen, 2)}
                deepened_by_id = {d.id: d for d in deepened}
                synth_input2 = [deepened_by_id.get(e.id, e) for e in synth_input]
                yield {"type": "status", "stage": "refining_with_full_text"}
                t4 = time.perf_counter()
                async for ev in self.llm.synthesize_stream(question, synth_input2):
                    if ev["type"] == "text_delta":
                        if "text_partial" in ev:
                            yield {"type": "answer_delta", "text_partial": ev["text_partial"]}
                        else:
                            yield {"type": "answer_delta", "text": ev.get("text", "")}
                    elif ev["type"] == "structured":
                        structured = ev["data"]
                t_synth2 = time.perf_counter() - t4

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
                "n_paperclip": sum(1 for e in evidence if e.source == "Paperclip"),
                "n_web": sum(1 for e in evidence if e.source == "WebSearch"),
                "retrieval_seconds": round(t_retrieval, 2),
                "synthesis_seconds": round(t_synth1 + t_synth2, 2),
                "deepen_seconds": round(t_deepen, 2),
                "total_seconds": round(t_retrieval + t_synth1 + t_deepen + t_synth2, 2),
                "deepened_citations": deepened_ids,
                "rerank_used": bool(do_rerank),
                "deepen_used": bool(do_deepen and deepened_ids),
                "model": settings.anthropic_model,
                "embedding_provider": configured_provider(),
                "cache_hit": None,
            },
        )
        if use_cache:
            await set_cached(question, response)
        yield {"type": "done", "response": response.model_dump()}


def _refusal_response(question: str, kind: str = "individual_advice") -> AnswerResponse:
    if kind == "non_clinical":
        msg = (
            "This looks like a greeting or non-clinical input. Please ask a clinical "
            "question — for example: \"What is the evidence for GLP-1 receptor agonists "
            "in obesity management?\" or \"Are there active CAR-T trials for lupus?\""
        )
        lim = "Input did not appear to be a clinical question."
    else:
        msg = (
            "I can't provide individualized medical recommendations. Please consult "
            "a qualified clinician about your specific situation. I can summarize "
            "published evidence on related topics if you reframe the question."
        )
        lim = "Refused due to individualized-advice request."
    return AnswerResponse(
        question=question,
        answer=msg,
        limitations=lim,
        confidence="low",
        disclaimer=DISCLAIMER,
        metadata={"refusal": kind},
    )
