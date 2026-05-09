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
    """Return True for input that's clearly not a clinical question.

    Heuristics (English-centric, but lenient toward CJK / other scripts):
      - empty / pure punctuation → refuse
      - exact match in trivial greetings set → refuse
      - very short (<3 ASCII words) without any biomedical hint → refuse
      - non-ASCII content (>=8 chars) → trust the LLM, don't refuse here
    """
    q = question.strip()
    if not q:
        return True
    q_low = q.lower()
    if q_low in _TRIVIAL:
        return True
    # If the input contains any non-ASCII (e.g., Chinese, Arabic, etc.) we
    # don't have a regex vocabulary to validate it; defer to the LLM gate.
    if any(ord(c) > 127 for c in q) and len(q) >= 8:
        return False
    word_count = len(re.findall(r"[A-Za-z][A-Za-z0-9'-]*", q))
    if word_count < 3 and not _BIOMED_HINTS.search(q):
        return True
    return False
