"""Prompt templates for grounded clinical synthesis.

Design notes:
- The system prompt forces grounding: every claim must cite a CONTEXT id.
- We distinguish published evidence (PubMed/EuropePMC) from registered trials
  (ClinicalTrials.gov) and from web sources (FDA/guidelines/news).
- We require a `confidence` self-rating to discourage overclaiming.
"""
from __future__ import annotations

from src.models import Evidence


SYSTEM_PROMPT = """You are a clinical evidence synthesis assistant. You answer clinical \
questions by SYNTHESIZING the CONTEXT provided to you (and ONLY that context).

GROUNDING RULES (strict):
1. Every factual claim in the ANSWER must be followed by an inline citation tag.
   - PubMed:           [PMID:12345678]
   - Europe PMC:       [PMID:...] / [PMC:PMC123456] / [DOI:...]
   - ClinicalTrials:   [NCT01234567]
   - Web sources:      [WEB:1]
2. If the CONTEXT does not contain enough evidence to answer, SAY SO in LIMITATIONS.
   Do NOT invent citations. Do NOT use prior knowledge as a substitute for cited evidence.
3. Distinguish PUBLISHED EVIDENCE (PubMed / EuropePMC) from ONGOING/COMPLETED TRIALS
   (ClinicalTrials.gov), and from REGULATORY/GUIDELINE sources (WebSearch).
4. Weight evidence by study type when visible: meta-analyses & RCTs > observational > \
case reports; completed Phase 3/4 trials > Phase 1/2 > preprints.
5. Note conflicting findings explicitly; do not paper over them.
6. NEVER provide individualized medical advice (dosing, "you should take X"). Educate \
about the evidence, then recommend consulting a clinician.

OUTPUT: Use the `submit_clinical_answer` tool. Populate every field. Keep the answer \
2-4 paragraphs. Be precise and quantitative when possible (effect sizes, p-values, \
hazard ratios, CIs)."""


SYNTHESIS_TOOL = {
    "name": "submit_clinical_answer",
    "description": "Submit the synthesized, evidence-grounded clinical answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": (
                    "2-4 paragraph synthesized answer with inline citations like "
                    "[PMID:12345], [NCT01234567], [WEB:1]. Quantitative when possible."
                ),
            },
            "key_evidence": {
                "type": "array",
                "description": "The most load-bearing citations and why each matters.",
                "items": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "string", "description": "e.g. PMID:12345 or NCT01234567"},
                        "summary": {"type": "string", "description": "One sentence: why this evidence supports the answer."},
                    },
                    "required": ["citation_id", "summary"],
                },
            },
            "limitations": {
                "type": "string",
                "description": (
                    "What is missing, conflicting, outdated, or low-quality. If the "
                    "evidence is sparse, say so plainly. 2-4 sentences."
                ),
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "moderate", "low"],
                "description": (
                    "high = consistent multi-RCT/meta-analysis evidence; "
                    "moderate = some RCTs or observational consensus; "
                    "low = sparse, preprint-only, or conflicting."
                ),
            },
        },
        "required": ["answer", "key_evidence", "limitations", "confidence"],
    },
}


REFORMULATE_TOOL = {
    "name": "reformulate_query",
    "description": "Convert a user clinical question into precise retrieval queries.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pubmed_query": {
                "type": "string",
                "description": "PubMed-style query (Boolean/MeSH if helpful), max ~150 chars.",
            },
            "trials_query": {
                "type": "string",
                "description": "ClinicalTrials.gov free-text term — drug + condition.",
            },
            "intent": {
                "type": "string",
                "enum": ["efficacy", "safety", "trials_landscape", "mechanism", "guidelines", "other"],
            },
        },
        "required": ["pubmed_query", "trials_query", "intent"],
    },
}


def build_user_prompt(question: str, evidence: list[Evidence]) -> str:
    """Render the question + evidence into a single user message."""
    parts = [f"# QUESTION\n{question}\n", "# CONTEXT (your sole source of truth)\n"]
    if not evidence:
        parts.append("(No evidence retrieved.)")
    for ev in evidence:
        parts.append(f"---\n[{ev.id}]  source={ev.source}")
        parts.append(f"Title: {ev.title}")
        meta_bits = []
        if ev.metadata:
            if ev.metadata.get("year"):
                meta_bits.append(f"Year: {ev.metadata['year']}")
            if ev.metadata.get("journal"):
                meta_bits.append(f"Journal: {ev.metadata['journal']}")
            if ev.metadata.get("publication_types"):
                meta_bits.append(f"Type: {', '.join(ev.metadata['publication_types'][:2])}")
            if ev.metadata.get("status"):
                meta_bits.append(f"Status: {ev.metadata['status']}")
            if ev.metadata.get("phases"):
                meta_bits.append(f"Phase: {', '.join(ev.metadata['phases'])}")
            if ev.metadata.get("conditions"):
                meta_bits.append(f"Conditions: {', '.join(ev.metadata['conditions'][:3])}")
            if ev.metadata.get("interventions"):
                meta_bits.append(f"Interventions: {', '.join(ev.metadata['interventions'][:3])}")
            if ev.metadata.get("is_preprint"):
                meta_bits.append("PREPRINT (not peer-reviewed)")
        if meta_bits:
            parts.append(" | ".join(meta_bits))
        if ev.content:
            parts.append(f"Content: {ev.content[:1800]}")
        parts.append("")  # blank line
    parts.append(
        "---\nNow synthesize an evidence-grounded answer using ONLY the CONTEXT above. "
        "Cite every claim with the bracketed IDs shown."
    )
    return "\n".join(parts)


def build_reformulate_prompt(question: str) -> str:
    return (
        f"User question: {question}\n\n"
        "Convert this into precise PubMed and ClinicalTrials.gov queries. "
        "Strip filler words. Keep medical terms exact (e.g., 'GLP-1 receptor agonist' "
        "not 'weight loss drugs'). For PubMed, prefer key entity terms over Boolean "
        "operators unless clearly needed."
    )
