"""Optional Tavily web search — useful for very recent FDA actions or guidelines
that have not yet been indexed in PubMed.

Disabled by default; toggle USE_WEB_SEARCH=true in .env or per-request.
"""
from __future__ import annotations

import logging
from typing import List

import httpx

from src.models import Evidence

log = logging.getLogger(__name__)
TAVILY_URL = "https://api.tavily.com/search"


async def search_web(query: str, api_key: str, max_results: int = 3, timeout: float = 15.0) -> List[Evidence]:
    if not api_key:
        return []
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        # Bias toward authoritative biomedical/regulatory domains
        "include_domains": [
            "fda.gov",
            "ema.europa.eu",
            "nih.gov",
            "who.int",
            "cdc.gov",
            "uptodate.com",
            "nejm.org",
            "thelancet.com",
            "jamanetwork.com",
            "nature.com",
            "cochranelibrary.com",
        ],
        "include_raw_content": False,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(TAVILY_URL, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        log.warning(f"Tavily web search failed: {e}")
        return []

    out: list[Evidence] = []
    for i, hit in enumerate(data.get("results", []), 1):
        url = hit.get("url", "")
        title = hit.get("title", "(no title)")
        content = hit.get("content", "")
        out.append(Evidence(
            source="WebSearch",
            id=f"WEB:{i}",
            title=title,
            content=content,
            url=url,
            metadata={"score": hit.get("score"), "published": hit.get("published_date")},
        ))
    return out
