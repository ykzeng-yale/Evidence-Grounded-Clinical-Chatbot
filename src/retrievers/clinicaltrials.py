"""ClinicalTrials.gov v2 API retriever.

Docs: https://clinicaltrials.gov/data-api/api

We hit /studies with query.term + selected fields to keep the payload small.
The v2 API returns nested protocolSection objects; we flatten the bits
that the LLM actually needs (title, summary, status, phase, conditions, interventions).
"""
from __future__ import annotations

import logging
from typing import List

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.models import Evidence

log = logging.getLogger(__name__)
CT_BASE = "https://clinicaltrials.gov/api/v2/studies"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    reraise=True,
)
async def search_clinicaltrials(query: str, max_results: int = 5, timeout: float = 15.0) -> List[Evidence]:
    """Search ClinicalTrials.gov by free-text term and return Evidence list."""
    params = {
        "query.term": query,
        "pageSize": max_results,
        "format": "json",
        # countTotal is useful but slow; skip
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(CT_BASE, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        log.warning(f"ClinicalTrials.gov search failed: {e}")
        return []

    out: list[Evidence] = []
    for study in data.get("studies", []):
        ps = study.get("protocolSection", {}) or {}

        idm = ps.get("identificationModule", {}) or {}
        nct = idm.get("nctId") or ""
        if not nct:
            continue
        title = idm.get("briefTitle") or idm.get("officialTitle") or "(no title)"

        descm = ps.get("descriptionModule", {}) or {}
        summary = descm.get("briefSummary") or ""
        # detailedDescription can be very long; only keep first 1500 chars
        if len(summary) < 200:
            dd = descm.get("detailedDescription") or ""
            if dd:
                summary = (summary + "\n\n" + dd).strip()
        summary = summary[:2000]

        statusm = ps.get("statusModule", {}) or {}
        status = statusm.get("overallStatus") or ""
        start_date = (statusm.get("startDateStruct") or {}).get("date") or ""
        completion_date = (statusm.get("completionDateStruct") or {}).get("date") or ""

        designm = ps.get("designModule", {}) or {}
        phases = designm.get("phases") or []
        enrollment = (designm.get("enrollmentInfo") or {}).get("count")

        condm = ps.get("conditionsModule", {}) or {}
        conditions = condm.get("conditions") or []

        armsm = ps.get("armsInterventionsModule", {}) or {}
        interventions = [
            f"{i.get('type', '')}: {i.get('name', '')}".strip(": ").strip()
            for i in (armsm.get("interventions") or [])
            if i.get("name")
        ]

        out.append(Evidence(
            source="ClinicalTrials.gov",
            id=f"NCT:{nct}",
            title=title,
            content=summary,
            url=f"https://clinicaltrials.gov/study/{nct}",
            metadata={
                "status": status,
                "phases": phases,
                "conditions": conditions[:5],
                "interventions": interventions[:5],
                "enrollment": enrollment,
                "start_date": start_date,
                "completion_date": completion_date,
            },
        ))
    return out
