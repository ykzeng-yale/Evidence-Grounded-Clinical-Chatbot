# Evidence-Grounded Clinical Chatbot

> A Python service that answers clinical questions by retrieving evidence from
> **PubMed**, **ClinicalTrials.gov**, and **Europe PMC** (preprints), then
> synthesizing a grounded answer with **Claude** — every claim cites a source ID.

**Live demo:** *(deployed to Vercel — link in repo description)*
**Take-home assignment:** *Evidence-Grounded Clinical Chatbot, May 2026*

---

## What it does

```
                    ┌─────────────────────────────────────────────┐
   user question ─► │ 1. Reformulate (Claude tool_use)            │
                    │    → pubmed_query, trials_query, intent     │
                    └────────────────────┬────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │ 2. Parallel async retrieval                 │
                    │    ├ PubMed E-utilities (esearch + efetch)  │
                    │    ├ ClinicalTrials.gov v2 (/studies)       │
                    │    ├ Europe PMC (preprints — bioRxiv etc.)  │
                    │    └ Tavily (optional, FDA / guidelines)    │
                    └────────────────────┬────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │ 3. Synthesize (Claude, tool_use, streaming) │
                    │    submit_clinical_answer({                 │
                    │      answer, key_evidence[], limitations,   │
                    │      confidence: high|moderate|low })       │
                    └────────────────────┬────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │ 4. AnswerResponse + Supabase cache          │
                    └─────────────────────────────────────────────┘
```

The web UI streams Server-Sent Events so the user sees:

1. **`reformulating`** → the LLM-derived retrieval queries
2. **`retrieving`** → live count of evidence items as they land
3. **`synthesizing`** → Claude's answer typed token-by-token
4. **`done`** → final structured response with citation cards

## Why these choices

| Decision | Rationale |
|---|---|
| **Direct API retrieval, no vector store** | PubMed and ClinicalTrials.gov are the *ground truth*. A vector cache would go stale and add a hallucination surface. The "RAG" is the API call itself. |
| **Europe PMC for preprints** | The assignment requires PubMed + CT.gov; preprints from bioRxiv/medRxiv (the same corpus the open-source [paperclip](https://github.com/GXL-ai/paperclip) tool indexes) add timely signal. Europe PMC exposes them via a free public REST API — no OAuth, works in serverless. |
| **Async parallel retrieval** | All three retrievers run simultaneously via `asyncio.gather` — total latency = max(retriever), not sum. |
| **Tool-use for structured output** | Claude's `tool_choice` guarantees a JSON object with `answer / key_evidence / limitations / confidence`. No regex parsing, no malformed responses. |
| **Inline citation tags `[PMID:#] / [NCT:#]`** | Cheap to enforce, easy for the UI to linkify, and trivially auditable — the eval script flags any cite ID that isn't in the retrieved evidence. |
| **Mandatory `limitations` + `confidence`** | Clinical context demands honesty. If evidence is sparse or conflicting, the model must say so rather than overclaim. |
| **Query reformulation** | A first Claude call rewrites natural questions into precise PubMed/CT.gov terms, improving recall (e.g., "weight loss drugs" → "GLP-1 receptor agonist obesity"). |
| **Supabase cache** | Same question hits → 0 API cost, sub-100ms response. Optional and gracefully degrades. |
| **Streaming via SSE (not WebSocket)** | One-way server→client is sufficient; SSE works through every CDN, no upgrade dance. |
| **Refusal layer for individualized advice** | Patterns like "what should I take" trip an early refusal — we educate, we don't prescribe. |

## Repository layout

```
.
├── api/main.py              # FastAPI app: /api/ask, /api/ask/stream, /api/sample-questions
├── src/
│   ├── config.py            # pydantic-settings; reads .env
│   ├── models.py            # Evidence, Citation, AnswerResponse
│   ├── safety.py            # disclaimer + refusal patterns
│   ├── cache.py             # Supabase cache (optional)
│   ├── pipeline.py          # orchestration: ask() and stream_ask()
│   ├── retrievers/
│   │   ├── pubmed.py        # NCBI E-utilities (esearch + efetch XML)
│   │   ├── clinicaltrials.py# CT.gov v2 (/studies)
│   │   ├── europepmc.py     # Europe PMC (preprints)
│   │   └── web.py           # Tavily web search (optional)
│   └── synthesis/
│       ├── prompts.py       # system prompt + tool schemas
│       └── llm.py           # Claude client (sync + streaming)
├── frontend/index.html      # Tailwind-CDN single-page UI with SSE streaming
├── cli.py                   # python cli.py "your question"
├── scripts/evaluate.py      # automated eval over examples/sample_questions.json
├── tests/test_smoke.py      # live API smoke tests (no LLM cost)
├── examples/sample_questions.json
├── vercel.json              # Vercel Python serverless config
├── requirements.txt
└── .env.example
```

## Quickstart

```bash
git clone https://github.com/ykzeng-yale/Evidence-Grounded-Clinical-Chatbot
cd Evidence-Grounded-Clinical-Chatbot

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env, fill in ANTHROPIC_API_KEY (required) and PUBMED_EMAIL (NCBI policy)

# CLI
python cli.py "What is the current evidence for GLP-1 receptor agonists in obesity management?"

# Web UI + API server
uvicorn api.main:app --reload --port 8000
# open http://localhost:8000

# Smoke tests
pytest tests/ -v

# Eval against sample questions
python scripts/evaluate.py --limit 3
```

## API

### `POST /api/ask`  — JSON answer

```bash
curl -s http://localhost:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Are there active clinical trials for CAR-T therapy in lupus?"}' | jq
```

Returns:

```jsonc
{
  "question": "Are there active clinical trials for CAR-T therapy in lupus?",
  "answer": "Several Phase 1/2 trials of CD19-directed CAR-T are recruiting in refractory SLE [NCT05798117]…",
  "citations": [
    { "id": "NCT:NCT05798117", "title": "...", "url": "https://clinicaltrials.gov/study/NCT05798117",
      "source": "ClinicalTrials.gov", "summary": "Open-label Phase 1/2 of CD19 CAR-T in refractory SLE." }
  ],
  "evidence": [ /* full retrieved set with abstracts */ ],
  "limitations": "All current trials are early-phase with small cohorts; long-term safety data are limited.",
  "confidence": "moderate",
  "disclaimer": "...",
  "metadata": { "retrieval_seconds": 1.2, "synthesis_seconds": 3.4, "total_seconds": 4.6,
                "n_pubmed": 5, "n_trials": 5, "n_preprints": 3 }
}
```

### `GET /api/ask/stream?q=...`  — SSE stream

Events emitted (in order):

| `event:` | payload |
|---|---|
| `status`     | `{stage: "reformulating" \| "retrieving" \| "synthesizing"}` |
| `queries`    | `{pubmed_query, trials_query, intent}` |
| `evidence`   | `{evidence: [...], retrieval_seconds}` |
| `answer_delta` | `{text_partial: "..."}`  *(growing answer string)* |
| `done`       | `{response: AnswerResponse}` |
| `error`      | `{message}` |

## Evaluation

`scripts/evaluate.py` runs the pipeline against the sample questions and reports
**latency**, **citation validity** (every inline `[PMID:#]` must appear in the
retrieved evidence — catches hallucinated citations), and **confidence
distribution**. A real production eval would add clinician-graded answer quality
on a held-out gold set; this script is the lightweight automated proxy.

```
$ python scripts/evaluate.py
▶ GLP-1 in obesity: ...
  ✓ 4.7s · 5P+5T+3pre · cited 4/4 valid · conf=high
...
========================================================================
RESULTS: 6/6 succeeded
  latency p50/p95: 4.6s / 7.1s
  citation validity: 100.0%
  runs with any hallucinated citations: 0/6
  confidence distribution: {'high': 3, 'moderate': 3}
========================================================================
```

## Safety

- **Disclaimer** appended to every response.
- **Refusal layer** in `src/safety.py` declines individualized recommendations
  (e.g., "what dose should I take") and reframes toward evidence summaries.
- Model is instructed to **never substitute prior knowledge for cited evidence**.
- **Limitations** section is mandatory and rated by self-reported `confidence`.

## Known limitations & next steps

- **Eval is automated-only.** Real clinical correctness needs clinician adjudication
  on a held-out gold set; we surface latency + citation validity as proxies.
- **Recency.** PubMed indexing lags ~3–6 months; Europe PMC partially mitigates via
  preprints; the optional Tavily web search covers FDA/guidelines but is off by default.
- **Single-turn.** No conversational memory. Adding a `session_id` + Supabase
  `chat_history` table is a 30-line extension.
- **Multi-language.** Retrievers default to English. Non-English queries succeed
  but recall drops; reformulation helps but isn't a translation layer.
- **Trial activeness.** We pass `OverallStatus` to the LLM, but a "active trials"
  question could be sharpened with `query.term=AREA[OverallStatus]RECRUITING`.

## Required keys

| Key | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Claude synthesis |
| `PUBMED_EMAIL`      | ✅ (NCBI policy) | Identifying email tag for E-utilities |
| `NCBI_API_KEY`      | recommended | Lifts PubMed rate limit 3 → 10 req/s (free) |
| `TAVILY_API_KEY`    | optional | FDA/guideline web search |
| `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` | optional | Query cache |

See [`.env.example`](.env.example).

## License

MIT (assignment / portfolio use).
