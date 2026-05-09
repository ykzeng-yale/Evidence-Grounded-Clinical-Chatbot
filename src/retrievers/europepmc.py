"""Europe PMC retriever — covers PubMed + PMC + bioRxiv + medRxiv preprints.

Why include Europe PMC alongside PubMed?
  - Europe PMC indexes preprints (bioRxiv/medRxiv) that PubMed does not.
  - This is the same corpus as the open-source `paperclip` tool, but via a
    free public REST API (no OAuth), so it works in serverless environments.

Docs: https://europepmc.org/RestfulWebService
"""
from __future__ import annotations

import logging
from typing import List

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.models import Evidence

log = logging.getLogger(__name__)
EPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    reraise=True,
)
async def search_europepmc(query: str, max_results: int = 3, prefer_preprints: bool = True, timeout: float = 15.0) -> List[Evidence]:
    """Search Europe PMC; bias toward preprints to complement PubMed coverage."""
    # SRC:PPR limits to preprint corpora (bioRxiv, medRxiv, Research Square, etc.)
    if prefer_preprints:
        q = f"({query}) AND (SRC:PPR)"
    else:
        q = query

    params = {
        "query": q,
        "format": "json",
        "pageSize": max_results,
        "resultType": "core",
        # NOTE: do NOT pass sort=relevance — Europe PMC returns an empty body if you do.
        # Default ordering is relevance for free-text queries.
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(EPMC_BASE, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        log.warning(f"Europe PMC search failed: {e}")
        return []

    results = (data.get("resultList") or {}).get("result") or []
    out: list[Evidence] = []
    for item in results:
        # Build canonical citation ID by source
        src = (item.get("source") or "").upper()
        ext_id = item.get("id") or item.get("pmid") or ""
        pmid = item.get("pmid")
        pmcid = item.get("pmcid")
        doi = item.get("doi")

        if pmid:
            cid = f"PMID:{pmid}"
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        elif pmcid:
            cid = f"PMC:{pmcid}"
            url = f"https://europepmc.org/article/PMC/{pmcid}"
        elif doi:
            cid = f"DOI:{doi}"
            url = f"https://doi.org/{doi}"
        else:
            cid = f"EPMC:{src}:{ext_id}"
            url = f"https://europepmc.org/article/{src}/{ext_id}"

        title = (item.get("title") or "(no title)").strip().rstrip(".")
        abstract = item.get("abstractText") or ""
        journal = item.get("journalTitle") or item.get("bookOrReportDetails", {}).get("publisher") or src
        year = str(item.get("pubYear") or "")
        authors = item.get("authorString") or ""

        out.append(Evidence(
            source="EuropePMC",
            id=cid,
            title=title,
            content=abstract,
            url=url,
            metadata={
                "journal": journal,
                "year": year,
                "authors": authors,
                "epmc_source": src,
                "is_preprint": src == "PPR",
            },
        ))
    return out
