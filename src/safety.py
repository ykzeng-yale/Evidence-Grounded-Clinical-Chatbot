"""Safety disclaimers and guardrails."""
import re

DISCLAIMER = (
    "This information is generated from public biomedical literature (PubMed, "
    "Europe PMC) and registered clinical trials (ClinicalTrials.gov). It is provided "
    "for educational purposes only and is NOT medical advice. Evidence may be "
    "incomplete, outdated, or not applicable to your situation. Always consult a "
    "qualified healthcare provider for diagnosis or treatment decisions."
)

REFUSAL_KEYWORDS = (
    "what should i take",
    "what should i do",
    "should i stop my",
    "what dose should i",
)


def needs_individual_advice_refusal(question: str) -> bool:
    q = question.lower().strip()
    return any(k in q for k in REFUSAL_KEYWORDS)


# Trivial / non-clinical inputs we should reject before spending API budget.
# We accept anything ≥3 words; below that we require at least one biomedical token.
_TRIVIAL = {"hi", "hello", "hey", "test", "ok", "thanks", "thank you", "ping",
            "yo", "?", "??", "...", "yes", "no"}
_BIOMED_HINTS = re.compile(
    r"\b(drug|medication|patient|trial|disease|cancer|diabet|treatment|therapy|"
    r"symptom|diagnos|surgery|vaccine|infection|syndrome|adverse|efficacy|dose|"
    r"clinical|fda|guideline|risk|prognos|biomarker|gene|antibod|inhibitor|"
    r"agonist|antagonist|receptor|enzyme|protein|cell|tissue|organ|blood|"
    r"hormone|metabol|pharm|toxic|carcinoma|lymphoma|leukemia|tumor|tumour|"
    r"hypertension|cardio|hepat|renal|pulmon|neuro|psych|immune|allerg)\w*",
    re.IGNORECASE,
)


def needs_clinical_question_refusal(question: str) -> bool:
    """Return True for input that's clearly not a clinical question."""
    q = question.strip().lower()
    if not q:
        return True
    if q in _TRIVIAL:
        return True
    word_count = len(re.findall(r"\w+", q))
    if word_count < 3 and not _BIOMED_HINTS.search(q):
        return True
    return False
