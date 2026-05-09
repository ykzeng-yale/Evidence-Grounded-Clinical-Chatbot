"""Supabase-backed query cache. Optional — gracefully no-ops if not configured."""
from __future__ import annotations

import hashlib
import logging

from src.config import settings
from src.models import AnswerResponse

log = logging.getLogger(__name__)
_client = None
_initialized = False


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
        log.info("Supabase cache enabled")
    except Exception as e:
        log.warning(f"Supabase init failed; cache disabled: {e}")
        _client = None
    return _client


def _hash(question: str) -> str:
    return hashlib.sha256(question.lower().strip().encode()).hexdigest()


def get_cached(question: str) -> AnswerResponse | None:
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
        log.debug(f"cache get failed: {e}")
    return None


def set_cached(question: str, response: AnswerResponse) -> None:
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
        log.debug(f"cache set failed: {e}")
