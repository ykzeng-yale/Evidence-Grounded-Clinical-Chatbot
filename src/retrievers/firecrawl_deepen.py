"""Firecrawl full-text deepening.

After the first synthesis pass, Claude marks 1–3 citations as `key_evidence`.
For PMC papers (where full text is freely available), we use Firecrawl to
scrape the article HTML, extract the Methods + Results sections, and append
them to the next-pass context.

Resolution flow for citation IDs:
  - PMC:PMC######  → use directly
  - PMID:#         → call NCBI ID converter to map to PMC:#, skip if not in PMC
  - Anything else  → skip (paywalled, not full-text accessible)

Why this matters: PubMed abstracts often omit the dose, sample size, p-values,
hazard ratios, and adverse-event tables that clinicians actually need. A
second synthesis pass with full-text Methods+Results produces materially
better quantitative answers — at the cost of ~2s of extra Firecrawl latency.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import List

import httpx

from src.models import Evidence

log = logging.getLogger(__name__)
FIRECRAWL_URL = "https://api.firecrawl.dev/v2/scrape"
NCBI_IDCONV_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"

# Sections we care about (in order of priority)
SECTION_HEADERS = [
    "Methods", "METHODS", "Methodology", "Materials and Methods",
    "Results", "RESULTS", "Findings",
    "Discussion", "DISCUSSION",   # only kept if Results not found
    "Conclusion", "Conclusions",
]


def _direct_pmc_url(citation_id: str) -> str | None:
    """Map PMC:PMC### directly. Returns None for PMID (caller must resolve via NCBI)."""
    if citation_id.startswith("PMC:"):
        pmcid = citation_id.split(":", 1)[1]
        if not pmcid.startswith("PMC"):
            pmcid = "PMC" + pmcid
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    return None


async def _resolve_pmid_to_pmc(pmids: list[str], email: str | None = None, timeout: float = 8.0) -> dict[str, str]:
    """Batch-call NCBI ID converter to map PMID -> PMCID. Returns {pmid: pmcid}."""
    if not pmids:
        return {}
    params = {"ids": ",".join(pmids), "format": "json", "tool": "egcc"}
    if email:
        params["email"] = email
    out: dict[str, str] = {}
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
            r = await c.get(NCBI_IDCONV_URL, params=params)
            r.raise_for_status()
            data = r.json()
            for rec in data.get("records", []):
                pmid = rec.get("pmid")
                pmcid = rec.get("pmcid")
                if pmid and pmcid:
                    out[str(pmid)] = pmcid
    except Exception as e:
        log.debug(f"NCBI ID converter failed: {e}")
    return out


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
    citations: list[dict],
    evidence: list[Evidence],
    api_key: str,
    max_deepen: int = 2,
    timeout: float = 25.0,
    pubmed_email: str | None = None,
) -> List[Evidence]:
    """For up to `max_deepen` cited papers that have PMC full text available,
    fetch via Firecrawl and return Evidence objects with Methods+Results
    appended. Resolves PMID→PMC via NCBI ID converter.

    Sources considered: PubMed, EuropePMC, Paperclip. (CT.gov + WebSearch skipped.)
    """
    if not api_key:
        return []
    by_id = {ev.id: ev for ev in evidence}

    # 1. Identify candidate citations and split into direct-PMC and needs-resolve PMID groups
    candidates: list[tuple[Evidence, str | None]] = []  # (evidence, direct_pmc_url or None)
    pmids_to_resolve: list[tuple[str, Evidence]] = []   # [(pmid, evidence)]
    for c in citations[: max_deepen * 5]:  # over-collect; many won't have PMC
        cid = c.get("citation_id")
        if not cid or cid not in by_id:
            continue
        ev = by_id[cid]
        if ev.source not in ("PubMed", "EuropePMC", "Paperclip"):
            continue
        direct = _direct_pmc_url(cid)
        if direct:
            candidates.append((ev, direct))
        elif cid.startswith("PMID:"):
            pmid = cid.split(":", 1)[1]
            pmids_to_resolve.append((pmid, ev))

    # 2. Batch-resolve PMIDs to PMCIDs
    if pmids_to_resolve:
        pmid_map = await _resolve_pmid_to_pmc([p for p, _ in pmids_to_resolve], email=pubmed_email)
        for pmid, ev in pmids_to_resolve:
            pmcid = pmid_map.get(pmid)
            if pmcid:
                if not pmcid.startswith("PMC"):
                    pmcid = "PMC" + pmcid
                candidates.append((ev, f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"))

    if not candidates:
        log.info("deepen: no PMC-accessible candidates among top citations")
        return []

    # 3. Scrape up to max_deepen in parallel (faster than sequential)
    targets = candidates[:max_deepen]
    log.info(f"deepen: scraping {len(targets)} PMC URLs in parallel")
    scrape_tasks = [_scrape_full_text(url, api_key, timeout=timeout) for _, url in targets]
    markdowns = await asyncio.gather(*scrape_tasks, return_exceptions=True)

    deepened: list[Evidence] = []
    for (ev, url), md in zip(targets, markdowns):
        if isinstance(md, Exception) or not md:
            log.debug(f"deepen scrape failed for {ev.id}: {md if isinstance(md, Exception) else 'empty'}")
            continue
        sections = _extract_sections(md)
        if not sections:
            log.debug(f"deepen: no Methods/Results sections found in {url}")
            continue
        new_content = (ev.content or "") + "\n\n--- FULL-TEXT METHODS / RESULTS (Firecrawl) ---\n" + sections
        deepened.append(Evidence(
            source=ev.source,
            id=ev.id,
            title=ev.title,
            content=new_content,
            url=ev.url,
            metadata={**ev.metadata, "deepened": True, "deepen_url": url},
        ))
        log.info(f"deepened {ev.id}: +{len(sections)} chars from {url}")
    return deepened
