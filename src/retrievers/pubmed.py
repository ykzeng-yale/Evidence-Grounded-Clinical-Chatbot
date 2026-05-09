"""PubMed retriever via NCBI E-utilities (esearch + efetch).

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/

Two-step pattern:
  1) esearch  → list of PMIDs for query
  2) efetch   → full article XML (title, abstract, journal, year, authors)

Why XML over JSON for efetch: JSON mode does not return abstracts; XML does.
"""
from __future__ import annotations

import asyncio
import logging
from typing import List
from xml.etree import ElementTree as ET

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.models import Evidence

log = logging.getLogger(__name__)
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    reraise=True,
)
async def _get(client: httpx.AsyncClient, url: str, params: dict) -> httpx.Response:
    r = await client.get(url, params=params)
    r.raise_for_status()
    return r


async def search_pubmed(
    query: str,
    max_results: int = 5,
    api_key: str | None = None,
    email: str | None = None,
    timeout: float = 15.0,
) -> List[Evidence]:
    """Search PubMed and return parsed Evidence objects with abstracts."""
    common = {"tool": "evidence-grounded-clinical-chatbot"}
    if email:
        common["email"] = email
    if api_key:
        common["api_key"] = api_key

    async with httpx.AsyncClient(timeout=timeout) as client:
        # Step 1: esearch
        search_params = {
            **common,
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        try:
            r = await _get(client, f"{PUBMED_BASE}/esearch.fcgi", search_params)
            ids = r.json().get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            log.warning(f"PubMed esearch failed: {e}")
            return []

        if not ids:
            return []

        # Step 2: efetch — XML so we get abstracts
        fetch_params = {
            **common,
            "db": "pubmed",
            "id": ",".join(ids),
            "rettype": "abstract",
            "retmode": "xml",
        }
        try:
            r2 = await _get(client, f"{PUBMED_BASE}/efetch.fcgi", fetch_params)
            return _parse_pubmed_xml(r2.text)
        except Exception as e:
            log.warning(f"PubMed efetch failed: {e}")
            return []


def _parse_pubmed_xml(xml_str: str) -> List[Evidence]:
    out: list[Evidence] = []
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        log.warning(f"PubMed XML parse error: {e}")
        return []

    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""
        if not pmid:
            continue

        title_el = art.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        # Abstract may have multiple labelled sections (Background, Methods, etc.)
        abs_parts: list[str] = []
        for at in art.findall(".//Abstract/AbstractText"):
            txt = "".join(at.itertext()).strip()
            label = at.get("Label")
            abs_parts.append(f"{label}: {txt}" if label else txt)
        abstract = "\n".join(p for p in abs_parts if p)

        journal_el = art.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else ""

        year = ""
        year_el = art.find(".//PubDate/Year")
        if year_el is not None and year_el.text:
            year = year_el.text
        else:
            ml = art.find(".//PubDate/MedlineDate")
            if ml is not None and ml.text:
                year = ml.text[:4]

        authors: list[str] = []
        all_authors = art.findall(".//Author")
        for au in all_authors[:3]:
            ln = au.find("LastName")
            if ln is not None and ln.text:
                fn = au.find("ForeName")
                init = f" {fn.text[0]}." if fn is not None and fn.text else ""
                authors.append(f"{ln.text}{init}")
        author_str = ", ".join(authors)
        if len(all_authors) > 3:
            author_str += " et al."

        # Publication type (helps the LLM weight RCTs vs reviews vs case reports)
        pub_types = [pt.text for pt in art.findall(".//PublicationType") if pt.text]

        out.append(Evidence(
            source="PubMed",
            id=f"PMID:{pmid}",
            title=title or "(no title)",
            content=abstract,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            metadata={
                "journal": journal,
                "year": year,
                "authors": author_str,
                "publication_types": pub_types,
            },
        ))
    return out
