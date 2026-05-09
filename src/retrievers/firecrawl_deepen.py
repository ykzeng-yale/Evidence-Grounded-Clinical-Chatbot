"""Firecrawl full-text deepening.

After the first synthesis pass, Claude marks 1–3 citations as `key_evidence`.
For PMC/PubMed papers (where full text is freely available), we use Firecrawl
to scrape the article HTML, extract the Methods + Results sections, and
append them to the next-pass context.

Why this matters: PubMed abstracts often omit the dose, sample size, p-values,
hazard ratios, and adverse-event tables that clinicians actually need. A
second synthesis pass with full-text Methods+Results produces materially
better quantitative answers — at the cost of ~2s of extra Firecrawl latency.
"""
from __future__ import annotations

import logging
import re
from typing import List

import httpx

from src.models import Evidence

log = logging.getLogger(__name__)
FIRECRAWL_URL = "https://api.firecrawl.dev/v2/scrape"

# Sections we care about (in order of priority)
SECTION_HEADERS = [
    "Methods", "METHODS", "Methodology", "Materials and Methods",
    "Results", "RESULTS", "Findings",
    "Discussion", "DISCUSSION",   # only kept if Results not found
    "Conclusion", "Conclusions",
]


def _pmc_url_from_id(citation_id: str) -> str | None:
    """Map PMID:### or PMC:PMC### to a freely-scrape-able PMC full-text URL."""
    if citation_id.startswith("PMC:"):
        pmcid = citation_id.split(":", 1)[1]
        if not pmcid.startswith("PMC"):
            pmcid = "PMC" + pmcid
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    if citation_id.startswith("PMID:"):
        pmid = citation_id.split(":", 1)[1]
        # PMID landing page works but PMC has full text. We try the PMC redirect.
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return None


async def _scrape_full_text(url: str, api_key: str, timeout: float = 25.0) -> str | None:
    """POST to Firecrawl /scrape, return the page markdown (or None on failure)."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
        "removeBase64Images": True,
        "blockAds": True,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(FIRECRAWL_URL, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return (data.get("data") or {}).get("markdown") or ""
    except Exception as e:
        log.warning(f"Firecrawl scrape failed for {url}: {e}")
        return None


def _extract_sections(markdown: str, max_chars: int = 4000) -> str:
    """Heuristic: pull out Methods + Results blocks (or their nearest equivalents)."""
    if not markdown:
        return ""
    # Split into header→body chunks. Treat lines like "## Methods" as headers.
    chunks: list[tuple[str, str]] = []
    current_header = "_intro"
    current_body: list[str] = []
    for line in markdown.splitlines():
        m = re.match(r"^\s*#{1,4}\s+(.+?)\s*$", line)
        if m:
            chunks.append((current_header, "\n".join(current_body).strip()))
            current_header = m.group(1).strip()
            current_body = []
        else:
            current_body.append(line)
    chunks.append((current_header, "\n".join(current_body).strip()))

    # Pick out Methods + Results (or fallback to Discussion if Results absent)
    picked: list[str] = []
    have_results = False
    for header, body in chunks:
        h_low = header.lower()
        if any(s.lower() in h_low for s in ("methods", "methodology", "materials")):
            picked.append(f"## {header}\n{body}")
        elif any(s.lower() in h_low for s in ("results", "findings")):
            picked.append(f"## {header}\n{body}")
            have_results = True
    if not have_results:
        for header, body in chunks:
            if any(s.lower() in header.lower() for s in ("discussion", "conclusion")):
                picked.append(f"## {header}\n{body}")
                break

    text = "\n\n".join(picked).strip()
    # Trim
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…[truncated]"
    return text


async def deepen_top_citations(
    citations: list[dict],   # key_evidence from first synthesis: [{citation_id, summary}, ...]
    evidence: list[Evidence],
    api_key: str,
    max_deepen: int = 2,
    timeout: float = 25.0,
) -> List[Evidence]:
    """For up to `max_deepen` PMC/PubMed citations, fetch full text and create
    Evidence objects of source='PubMed' with the Methods+Results appended to
    the existing abstract. Returns the *deepened* Evidence list (NOT a merge —
    caller decides where to insert).
    """
    if not api_key:
        return []
    by_id = {ev.id: ev for ev in evidence}
    targets: list[Evidence] = []
    for c in citations[:max_deepen * 3]:  # over-collect in case of non-deepen-able cites
        cid = c.get("citation_id")
        if not cid or cid not in by_id:
            continue
        ev = by_id[cid]
        if ev.source not in ("PubMed", "EuropePMC"):
            continue
        url = _pmc_url_from_id(cid)
        if not url:
            continue
        targets.append(ev)
        if len(targets) >= max_deepen:
            break

    if not targets:
        return []

    deepened: list[Evidence] = []
    for ev in targets:
        url = _pmc_url_from_id(ev.id)
        markdown = await _scrape_full_text(url, api_key, timeout=timeout)
        if not markdown:
            continue
        sections = _extract_sections(markdown)
        if not sections:
            continue
        # Mutate the existing Evidence's content to prepend the abstract + sections
        new_content = (ev.content or "") + "\n\n--- FULL-TEXT METHODS / RESULTS (Firecrawl) ---\n" + sections
        deepened.append(Evidence(
            source=ev.source,
            id=ev.id,
            title=ev.title,
            content=new_content,
            url=ev.url,
            metadata={**ev.metadata, "deepened": True, "deepen_url": url},
        ))
        log.info(f"deepened {ev.id}: +{len(sections)} chars")
    return deepened
