"""Smoke tests — hit live APIs with small queries to verify wiring.

Run:  pytest tests/ -v
These cost a few NCBI/CT.gov requests (no LLM calls) so they're cheap.
"""
import os
import pytest

from src.retrievers.clinicaltrials import search_clinicaltrials
from src.retrievers.europepmc import search_europepmc
from src.retrievers.pubmed import search_pubmed


@pytest.mark.asyncio
async def test_pubmed_returns_evidence():
    res = await search_pubmed("semaglutide obesity", max_results=2,
                              email=os.environ.get("PUBMED_EMAIL", "test@example.com"),
                              api_key=os.environ.get("NCBI_API_KEY"))
    assert isinstance(res, list)
    if res:  # network may be flaky
        ev = res[0]
        assert ev.id.startswith("PMID:")
        assert ev.url.startswith("https://pubmed.ncbi.nlm.nih.gov/")
        assert ev.title


@pytest.mark.asyncio
async def test_clinicaltrials_returns_evidence():
    res = await search_clinicaltrials("CAR-T lupus", max_results=2)
    assert isinstance(res, list)
    if res:
        ev = res[0]
        assert ev.id.startswith("NCT:")
        assert ev.url.startswith("https://clinicaltrials.gov/study/")
        assert ev.title


@pytest.mark.asyncio
async def test_europepmc_preprints():
    res = await search_europepmc("CAR-T lupus", max_results=2, prefer_preprints=True)
    assert isinstance(res, list)
    if res:
        ev = res[0]
        assert ev.title
        assert ev.url
