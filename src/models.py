"""Pydantic data models shared across retrievers, pipeline, and API."""
from typing import Literal, Optional
from pydantic import BaseModel, Field


SourceType = Literal["PubMed", "ClinicalTrials.gov", "EuropePMC", "Paperclip", "WebSearch"]


class Evidence(BaseModel):
    source: SourceType
    id: str                          # e.g. "PMID:12345" / "NCT01234567" / "PMC:..."
    title: str
    content: str = ""                # abstract / summary
    url: str
    metadata: dict = Field(default_factory=dict)


class Citation(BaseModel):
    id: str
    title: str
    url: str
    source: SourceType
    summary: Optional[str] = None    # short reason this citation supports the answer


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10_000)
    max_pubmed: Optional[int] = Field(default=None, ge=0, le=20)
    max_trials: Optional[int] = Field(default=None, ge=0, le=20)
    max_preprints: Optional[int] = Field(default=None, ge=0, le=10)
    max_paperclip: Optional[int] = Field(default=None, ge=0, le=10)
    use_web_search: Optional[bool] = None
    rerank: Optional[bool] = None
    deepen: Optional[bool] = None
    use_cache: bool = True


class AnswerResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    limitations: str = ""
    confidence: Literal["high", "moderate", "low"] = "moderate"
    disclaimer: str = ""
    metadata: dict = Field(default_factory=dict)
