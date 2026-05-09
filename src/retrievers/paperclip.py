"""Paperclip MCP retriever — 8M+ biomedical paper corpus (PMC + bioRxiv +
medRxiv + arXiv) with hybrid semantic + keyword search.

Auth: X-API-Key header.
Endpoint: https://paperclip.gxl.ai/mcp (JSON-RPC 2.0).

Why include this alongside PubMed E-utils + Europe PMC?
  - Paperclip exposes a single hybrid (BM25 + dense) search across the
    pre-indexed corpus, which is much better at synonym handling than the raw
    NCBI BM25 endpoint.
  - It includes arXiv (statistical methods, ML-in-medicine) which the other
    retrievers don't cover.
  - The search ID returned (`s_xxx`) can be passed back to `map`/`reduce`
    later for parallel multi-paper reading — kept as a hook for future use.
"""
from __future__ import annotations

import logging
import re
from typing import List

import httpx

from src.models import Evidence

log = logging.getLogger(__name__)
PAPERCLIP_MCP = "https://paperclip.gxl.ai/mcp"


async def search_paperclip(query: str, max_results: int = 5, api_key: str = "", timeout: float = 30.0) -> List[Evidence]:
    if not api_key:
        return []

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "paperclip",
            "arguments": {"command": f'search "{query.replace(chr(34), chr(39))}" -n {max_results}'},
        },
    }
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(PAPERCLIP_MCP, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        log.warning(f"Paperclip search failed: {e}")
        return []

    if data.get("error"):
        log.warning(f"Paperclip MCP error: {data['error']}")
        return []
    content_blocks = (data.get("result") or {}).get("content") or []
    text_blob = "\n".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
    if not text_blob:
        return []
    return _parse_paperclip_text(text_blob, max_results)


# Paperclip plain-text result format (one block per paper):
#
#   1. Autoantibody origins in lupus and in relapse post CAR-T therapy
#      Amalie Grenov, ... et al.
#      bio_019b0aeb7479 · bioRxiv · 2025-10-20
#      https://doi.org/10.1101/2025.10.20.683393
#      "Snippet quote..."
#
# Some entries lack a URL line; some use PMC URL instead of DOI; some are arXiv.

_RESULT_RE = re.compile(
    r"""
    ^\s*\d+\.\s+(?P<title>.+?)\n        # 1. Title
    \s+(?P<authors>.+?)\n               # Authors line
    \s+(?P<id>\S+)\s+·\s+(?P<src>[^·\n]+?)\s+·\s+(?P<date>[^\n]+?)\n  # id · source · date
    (?:\s+(?P<url>https?://\S+)\n)?     # optional url
    (?:\s+["“](?P<snippet>[^"”]+)["”])? # optional snippet
    """,
    re.VERBOSE | re.MULTILINE,
)


def _parse_paperclip_text(text: str, max_results: int) -> list[Evidence]:
    out: list[Evidence] = []
    # Walk results greedily — the regex above is anchored on the numbered prefix.
    # Reset matching from "N. " marker for robustness.
    blocks = re.split(r"\n(?=\s*\d+\.\s)", text.strip())
    for block in blocks:
        m = _RESULT_RE.match(block + "\n")
        if not m:
            continue
        title = m.group("title").strip()
        authors = re.sub(r"\s+", " ", m.group("authors").strip())
        pid = m.group("id").strip()
        src_label = m.group("src").strip()  # 'bioRxiv', 'medRxiv', 'PMC', 'arXiv'
        date = m.group("date").strip()
        url = (m.group("url") or "").strip()
        snippet = (m.group("snippet") or "").strip()

        # Build canonical citation id + URL
        cid, fallback_url = _canonical_id(pid, src_label, url)
        if not url:
            url = fallback_url

        out.append(Evidence(
            source="Paperclip",
            id=cid,
            title=title,
            content=snippet,
            url=url or "",
            metadata={
                "year": date[:4] if date else "",
                "authors": authors,
                "paperclip_source": src_label,
                "paperclip_id": pid,
                "is_preprint": src_label.lower() in ("biorxiv", "medrxiv", "arxiv", "research square"),
            },
        ))
        if len(out) >= max_results:
            break
    return out


def _canonical_id(paperclip_id: str, source_label: str, url: str) -> tuple[str, str]:
    """Map (paperclip_id, source) to (citation_id, fallback_url)."""
    src_low = source_label.lower()
    if paperclip_id.startswith("PMC"):
        return f"PMC:{paperclip_id}", f"https://pmc.ncbi.nlm.nih.gov/articles/{paperclip_id}/"
    if paperclip_id.startswith("PMID:") or paperclip_id.isdigit():
        n = paperclip_id.replace("PMID:", "")
        return f"PMID:{n}", f"https://pubmed.ncbi.nlm.nih.gov/{n}/"
    if paperclip_id.startswith("bio_") and "doi.org" in url:
        # bioRxiv preprints have a DOI URL — surface as DOI
        doi = url.split("doi.org/", 1)[-1].rstrip("/")
        return f"DOI:{doi}", url
    if paperclip_id.startswith("med_") and "doi.org" in url:
        doi = url.split("doi.org/", 1)[-1].rstrip("/")
        return f"DOI:{doi}", url
    if paperclip_id.startswith("arx_"):
        arxiv = paperclip_id.replace("arx_", "")
        return f"ARXIV:{arxiv}", f"https://arxiv.org/abs/{arxiv}"
    # generic fallback
    return f"PCLIP:{paperclip_id}", url or "https://paperclip.gxl.ai/"
