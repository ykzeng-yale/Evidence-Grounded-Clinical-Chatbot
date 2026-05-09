"""Safety disclaimers and guardrails."""

DISCLAIMER = (
    "This information is generated from public biomedical literature (PubMed, "
    "Europe PMC) and registered clinical trials (ClinicalTrials.gov). It is provided "
    "for educational purposes only and is NOT medical advice. Evidence may be "
    "incomplete, outdated, or not applicable to your situation. Always consult a "
    "qualified healthcare provider for diagnosis or treatment decisions."
)

# Refusal patterns: questions we should decline rather than answer
REFUSAL_KEYWORDS = (
    "what should i take",
    "what should i do",
    "should i stop my",
    "what dose should i",
)


def needs_individual_advice_refusal(question: str) -> bool:
    q = question.lower().strip()
    return any(k in q for k in REFUSAL_KEYWORDS)
