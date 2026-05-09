"""Two-tier cache:
  - Exact-match  (query_cache table, sha256 hash) — always available
  - Semantic     (query_embeddings + pgvector RPC `match_cached_query`)
                 — enabled when an embedding API key is configured AND
                 USE_SEMANTIC_CACHE=true.

Semantic cache returns a hit when the cosine similarity ≥ SEMANTIC_THRESHOLD
(default 0.92). This catches paraphrases like "GLP-1 in obesity" vs
"semaglutide for weight management" without re-running the full pipeline.

Both tiers gracefully no-op on errors so the pipeline never blocks.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging

from src.config import settings
from src.embeddings import configured_provider, embed_query
from src.models import AnswerResponse

log = logging.getLogger(__name__)
_client = None
_initialized = False
SEMANTIC_THRESHOLD = 0.92


def _get_client():
    global _client, _initialized
    if _initialized:
        return _client
    _initialized = True
    if not settings.use_supabase_cache or not settings.supabase_url or not settings.supabase_service_key:
        return None
    try:
        from supabase import create_client
        _client = create_client(settings.supabase_url, settings.supabase_service_key)
        log.info("Supabase cache enabled (exact-match)")
        if settings.use_semantic_cache and configured_provider():
            log.info(f"Semantic cache enabled (embedding provider: {configured_provider()})")
    except Exception as e:
        log.warning(f"Supabase init failed; cache disabled: {e}")
        _client = None
    return _client


def _hash(question: str) -> str:
    return hashlib.sha256(question.lower().strip().encode()).hexdigest()


# ---------- Exact-match cache (sha256) ----------
def get_cached_exact(question: str) -> AnswerResponse | None:
    sb = _get_client()
    if not sb:
        return None
    try:
        h = _hash(question)
        res = sb.table("query_cache").select("response").eq("question_hash", h).limit(1).execute()
        rows = res.data or []
        if rows:
            return AnswerResponse(**rows[0]["response"])
    except Exception as e:
        log.debug(f"exact cache get failed: {e}")
    return None


def set_cached_exact(question: str, response: AnswerResponse) -> None:
    sb = _get_client()
    if not sb:
        return
    try:
        h = _hash(question)
        sb.table("query_cache").upsert({
            "question_hash": h,
            "question": question,
            "response": response.model_dump(),
        }).execute()
    except Exception as e:
        log.debug(f"exact cache set failed: {e}")


# ---------- Semantic cache (pgvector RPC) ----------
async def get_cached_semantic(question: str, threshold: float = SEMANTIC_THRESHOLD) -> tuple[AnswerResponse, float, str] | None:
    sb = _get_client()
    if not sb or not settings.use_semantic_cache or not configured_provider():
        return None
    embedding = await embed_query(question)
    if not embedding:
        return None
    try:
        res = sb.rpc("match_cached_query", {
            "query_embedding": embedding,
            "match_threshold": threshold,
            "match_count": 1,
        }).execute()
        rows = res.data or []
        if rows:
            row = rows[0]
            return AnswerResponse(**row["response"]), float(row["similarity"]), row["question"]
    except Exception as e:
        log.debug(f"semantic cache get failed: {e}")
    return None


async def set_cached_semantic(question: str, response: AnswerResponse) -> None:
    sb = _get_client()
    if not sb or not settings.use_semantic_cache or not configured_provider():
        return
    embedding = await embed_query(question)
    if not embedding:
        return
    try:
        h = _hash(question)
        sb.table("query_embeddings").upsert({
            "question_hash": h,
            "question": question,
            "embedding": embedding,
            "response": response.model_dump(),
            "embed_model": configured_provider(),
        }, on_conflict="question_hash").execute()
    except Exception as e:
        log.debug(f"semantic cache set failed: {e}")


# ---------- Combined entry points used by pipeline ----------
async def get_cached(question: str) -> tuple[AnswerResponse, dict] | None:
    """Returns (response, cache_meta) on hit. cache_meta describes which tier hit."""
    # Tier 1: exact match (cheap)
    exact = get_cached_exact(question)
    if exact:
        return exact, {"tier": "exact"}
    # Tier 2: semantic match (one embedding call + RPC)
    sem = await get_cached_semantic(question)
    if sem:
        resp, sim, matched_question = sem
        return resp, {"tier": "semantic", "similarity": sim, "matched_question": matched_question}
    return None


async def set_cached(question: str, response: AnswerResponse) -> None:
    # Write to both tiers in parallel
    await asyncio.gather(
        asyncio.to_thread(set_cached_exact, question, response),
        set_cached_semantic(question, response),
        return_exceptions=True,
    )
