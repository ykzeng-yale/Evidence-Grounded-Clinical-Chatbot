"""LLM-as-reranker (Claude Haiku 4.5).

Why LLM-rerank instead of cross-encoder / sentence-transformers / cohere-rerank?
  - PubMed/CT.gov return BM25-relevance results that mostly look on-topic but
    often include tangential papers (e.g., a different drug class, wrong
    population). A small LLM is good at *clinical* relevance (does this paper
    actually answer the user's question?) where a generic re-ranker is not.
  - Uses our existing Anthropic key — no new dep, no new credential.
  - Haiku 4.5 is fast (~1s for 15 abstracts) and cheap.

Returns the same Evidence list, reordered by relevance score, optionally
truncated to top-K. The pipeline keeps all evidence visible to the user;
only the *synthesis context* uses the rerank top-K.
"""
from __future__ import annotations

import json
import logging
from typing import List

from anthropic import AsyncAnthropic

from src.models import Evidence

log = logging.getLogger(__name__)

RERANK_SYSTEM = (
    "You are a clinical evidence reranker. Given a user question and a numbered list "
    "of candidate papers/trials, score each by RELEVANCE to the question on a 0–10 "
    "scale (10 = directly answers the question; 0 = unrelated). Always return the "
    "rerank_evidence tool call with one entry per candidate."
)

RERANK_TOOL = {
    "name": "rerank_evidence",
    "description": "Score every candidate by relevance to the user question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "description": "One entry per candidate, in input order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "1-based candidate index"},
                        "score": {"type": "number", "description": "0–10 relevance score"},
                        "reason": {"type": "string", "description": "≤15 words"},
                    },
                    "required": ["index", "score"],
                },
            }
        },
        "required": ["scores"],
    },
}


def _build_rerank_prompt(question: str, evidence: list[Evidence]) -> str:
    parts = [f"USER QUESTION:\n{question}\n", "CANDIDATES:"]
    for i, ev in enumerate(evidence, 1):
        snippet = (ev.content or "").strip().replace("\n", " ")
        if len(snippet) > 350:
            snippet = snippet[:350] + "…"
        meta = []
        if ev.metadata.get("year"):
            meta.append(str(ev.metadata["year"]))
        if ev.metadata.get("status"):
            meta.append(ev.metadata["status"])
        if ev.metadata.get("phases"):
            meta.append("/".join(ev.metadata["phases"]))
        meta_str = f" ({' · '.join(meta)})" if meta else ""
        parts.append(f"{i}. [{ev.source} {ev.id}] {ev.title}{meta_str}\n   {snippet}")
    parts.append(
        "\nScore each candidate 0–10 on direct relevance to the user question. "
        "Penalize: wrong drug class, wrong population, wrong outcome. "
        "Reward: meta-analyses and Phase 3/4 RCTs, recent (≤5y), exact-topic matches."
    )
    return "\n".join(parts)


async def rerank_evidence(
    question: str,
    evidence: list[Evidence],
    api_key: str,
    keep_top_k: int | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> list[Evidence]:
    """Reorder `evidence` by Claude-judged relevance. Optionally keep only top K.
    Falls back to original order on error.
    """
    if not evidence:
        return evidence
    if len(evidence) == 1:
        return evidence

    client = AsyncAnthropic(api_key=api_key)
    try:
        msg = await client.messages.create(
            model=model,
            max_tokens=1500,
            system=RERANK_SYSTEM,
            tools=[RERANK_TOOL],
            tool_choice={"type": "tool", "name": "rerank_evidence"},
            messages=[{"role": "user", "content": _build_rerank_prompt(question, evidence)}],
        )
    except Exception as e:
        log.warning(f"rerank failed, keeping original order: {e}")
        return evidence[:keep_top_k] if keep_top_k else evidence

    scores: list[dict] = []
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "rerank_evidence":
            scores = list(block.input.get("scores", []))
            break

    if not scores:
        return evidence[:keep_top_k] if keep_top_k else evidence

    # Map index -> score; 1-based
    score_map = {s["index"]: float(s.get("score", 0)) for s in scores if isinstance(s.get("index"), int)}
    paired = [(score_map.get(i + 1, 0.0), ev) for i, ev in enumerate(evidence)]
    paired.sort(key=lambda x: x[0], reverse=True)
    reordered = [ev for _, ev in paired]

    # Stash scores into metadata for transparency in the UI
    for s in scores:
        idx = s.get("index", 0) - 1
        if 0 <= idx < len(evidence):
            evidence[idx].metadata["rerank_score"] = float(s.get("score", 0))
            if s.get("reason"):
                evidence[idx].metadata["rerank_reason"] = s["reason"]

    log.info(f"reranked {len(evidence)} items; top score {paired[0][0]:.1f}")
    return reordered[:keep_top_k] if keep_top_k else reordered
