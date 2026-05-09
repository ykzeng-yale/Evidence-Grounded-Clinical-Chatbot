"""Embedding service abstraction.

Supports multiple providers so the user can drop in whichever key they have:
  - voyage     (Voyage AI: voyage-3-lite — Anthropic's recommended embedder)
  - openai     (text-embedding-3-small)
  - cohere     (embed-english-v3.0)

If no embedding provider is configured, embed() returns None and the semantic
cache gracefully no-ops (falling back to exact-hash cache only).

Why Voyage by default?
  - Voyage-3 is the embedding model Anthropic recommends for Claude pipelines.
  - voyage-3-lite is 512-dim, fast, free up to 50M tokens/month.
  - voyage-clinical-* exists for medical text but is paywalled; we keep this
    code provider-agnostic so it's a 1-line swap.
"""
from __future__ import annotations

import logging
import os
from typing import List

import httpx

from src.config import settings

log = logging.getLogger(__name__)

EMBED_DIM_BY_MODEL = {
    "voyage-3-lite":               512,
    "voyage-3":                    1024,
    "voyage-3-large":              1024,
    "text-embedding-3-small":      1536,
    "text-embedding-3-large":      3072,
    "embed-english-v3.0":          1024,
}


def configured_provider() -> str | None:
    """Return the active embedding provider, or None if no key is configured."""
    if os.getenv("VOYAGE_API_KEY"):    return "voyage"
    if os.getenv("OPENAI_API_KEY"):    return "openai"
    if os.getenv("COHERE_API_KEY"):    return "cohere"
    return None


def configured_dim() -> int | None:
    p = configured_provider()
    if p == "voyage":  return EMBED_DIM_BY_MODEL["voyage-3-lite"]
    if p == "openai":  return EMBED_DIM_BY_MODEL["text-embedding-3-small"]
    if p == "cohere":  return EMBED_DIM_BY_MODEL["embed-english-v3.0"]
    return None


async def embed_query(text: str, timeout: float = 12.0) -> List[float] | None:
    """Embed a single query string. Returns None if no provider configured."""
    provider = configured_provider()
    if not provider:
        return None
    text = (text or "").strip()
    if not text:
        return None

    try:
        if provider == "voyage":
            r = await _voyage_embed([text], "query", timeout)
        elif provider == "openai":
            r = await _openai_embed([text], "text-embedding-3-small", timeout)
        elif provider == "cohere":
            r = await _cohere_embed([text], "search_query", timeout)
        else:
            return None
        return r[0] if r else None
    except Exception as e:
        log.warning(f"embed_query failed ({provider}): {e}")
        return None


async def embed_documents(texts: List[str], timeout: float = 30.0) -> List[List[float]] | None:
    provider = configured_provider()
    if not provider:
        return None
    texts = [t for t in (texts or []) if t and t.strip()]
    if not texts:
        return []
    try:
        if provider == "voyage":
            return await _voyage_embed(texts, "document", timeout)
        if provider == "openai":
            return await _openai_embed(texts, "text-embedding-3-small", timeout)
        if provider == "cohere":
            return await _cohere_embed(texts, "search_document", timeout)
    except Exception as e:
        log.warning(f"embed_documents failed ({provider}): {e}")
    return None


# ---------- Provider implementations ----------

async def _voyage_embed(texts, input_type, timeout):
    key = os.environ["VOYAGE_API_KEY"]
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": "voyage-3-lite", "input": texts, "input_type": input_type}
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post("https://api.voyageai.com/v1/embeddings", headers=headers, json=payload)
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]


async def _openai_embed(texts, model, timeout):
    key = os.environ["OPENAI_API_KEY"]
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": texts}
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload)
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]


async def _cohere_embed(texts, input_type, timeout):
    key = os.environ["COHERE_API_KEY"]
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": "embed-english-v3.0", "texts": texts, "input_type": input_type}
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post("https://api.cohere.ai/v1/embed", headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["embeddings"]
