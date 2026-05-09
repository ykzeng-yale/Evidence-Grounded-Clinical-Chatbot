# Evidence-Grounded Clinical Chatbot

> A Python service that answers clinical questions with a real, multi-stage RAG
> pipeline: live retrieval from **PubMed**, **ClinicalTrials.gov**, **Europe PMC**,
> and **Paperclip** (8M+ paper hybrid index), Claude-as-reranker, **Firecrawl**
> full-text deepening for the most-cited papers, and a **pgvector semantic cache**
> on Supabase — every claim cites a verifiable source ID.

**Live demo:** *(deployed to Vercel — link in repo description)*
**Take-home assignment:** *Evidence-Grounded Clinical Chatbot, May 2026*

---

## What it does

```
                    ┌─────────────────────────────────────────────┐
   user question ─► │ 0. Two-tier cache check                     │
                    │    a) sha256 exact-match (Supabase)         │
                    │    b) pgvector semantic match (sim ≥ 0.92)  │
                    └────────────────────┬────────────────────────┘
                                         ▼ miss
                    ┌─────────────────────────────────────────────┐
                    │ 1. Reformulate (Claude tool_use)            │
                    │    → pubmed_query, trials_query, intent     │
                    └────────────────────┬────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │ 2. Parallel async retrieval (5 sources)     │
                    │    ├ PubMed E-utilities (esearch + efetch)  │
                    │    ├ ClinicalTrials.gov v2 (/studies)       │
                    │    ├ Europe PMC (preprints)                 │
                    │    ├ Paperclip MCP (8M+ corpus, hybrid BM25 │
                    │    │  + dense; PMC + bioRxiv + medRxiv +    │
                    │    │  arXiv) — JSON-RPC over X-API-Key      │
                    │    └ Tavily (optional, FDA / guidelines)    │
                    └────────────────────┬────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │ 3. Rerank (Claude Haiku as reranker)        │
                    │    Score 0–10 per item, take top K          │
                    └────────────────────┬────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │ 4. Synthesize pass 1 (Claude, tool_use)     │
                    │    → answer + key_evidence[] + limitations  │
                    └────────────────────┬────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │ 5. Tier B deepen (top key_evidence only)    │
                    │    Firecrawl → PMC full text → Methods +    │
                    │    Results sections → re-synthesize         │
                    └────────────────────┬────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │ 6. Write to BOTH cache tiers + return       │
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
| **Live API retrieval as ground truth** | PubMed and ClinicalTrials.gov change daily. A pre-embedded vector store would silently serve stale trial statuses and superseded efficacy data — *dangerous* in clinical context. We treat live API calls as the canonical retrieval; vector tech is layered *on top* for rerank and cache, never as a replacement. |
| **Paperclip MCP as a 4th source** | Hybrid BM25+dense search across 8M+ pre-indexed papers (PMC + bioRxiv + medRxiv + arXiv). Better synonym handling than raw NCBI BM25 and includes arXiv (statistical methods, ML-in-medicine) that the others miss. Called via JSON-RPC with `X-API-Key` (no OAuth dance). |
| **Europe PMC for additional preprints** | Free public API — covers the same corpus as paperclip but without an API key, ensuring the system has at least 3 sources even if paperclip is unavailable. |
| **Async parallel retrieval** | All retrievers run simultaneously via `asyncio.gather` — total latency = max(retriever), not sum. |
| **Tier A: Claude-as-reranker (Haiku 4.5)** | After retrieval, score every candidate 0–10 on clinical relevance. Demotes wrong-population, wrong-drug-class, and (real example) astrophysics papers that slip in via keyword overlap. Cheap (~1s) and uses our existing Anthropic key. |
| **Tier B: Firecrawl full-text deepening** | After pass-1 synthesis, the top 1–2 cited PMC papers get scraped for their Methods + Results sections. Pass-2 re-synthesizes with the quantitative detail abstracts omit (effect sizes, CIs, AE rates). Adds ~3s; meaningfully sharper answers. |
| **Tier C: pgvector semantic cache** | Two-tier cache: sha256 exact-match (cheap) + pgvector cosine match (cosine ≥ 0.92). Catches paraphrases — "evidence for GLP-1 in obesity" and "semaglutide weight management" hit the same cached answer. Embeddings via Voyage AI (Anthropic-recommended), OpenAI, or Cohere. Falls back to exact-match cache when no embedding key is set. |
| **Tool-use for structured output** | Claude's `tool_choice` guarantees JSON: `answer / key_evidence / limitations / confidence`. No regex parsing, no malformed responses. |
| **Inline citation tags `[PMID:#] / [NCT:#] / [PMC:#] / [DOI:#] / [ARXIV:#]`** | Cheap to enforce, easy for the UI to linkify, trivially auditable — the eval script flags any cite ID that isn't in the retrieved evidence. |
| **Mandatory `limitations` + `confidence`** | Clinical context demands honesty. If evidence is sparse or conflicting, the model must say so rather than overclaim. |
| **Query reformulation (Claude)** | A first Claude call rewrites natural questions into precise PubMed/CT.gov terms ("weight loss drugs" → "GLP-1 receptor agonist obesity"). |
| **Streaming via SSE** | One-way server→client; works through every CDN, no upgrade dance. UI streams: status → queries → evidence → rerank_done → answer_delta → deepened → answer_delta (refined) → done. |
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
| `ANTHROPIC_API_KEY`  | ✅ | Claude synthesis + Claude-as-reranker (Haiku 4.5) |
| `PUBMED_EMAIL`       | ✅ (NCBI policy) | Identifying email tag for E-utilities |
| `NCBI_API_KEY`       | recommended | Lifts PubMed rate limit 3 → 10 req/s (free) |
| `PAPERCLIP_API_KEY`  | recommended | 8M+ paper hybrid search (PMC + bioRxiv + medRxiv + arXiv) |
| `FIRECRAWL_API_KEY`  | recommended | Tier B: full-text deepening for top citations |
| `TAVILY_API_KEY`     | optional | Web search for FDA/guidelines |
| `VOYAGE_API_KEY` *(or `OPENAI_API_KEY` / `COHERE_API_KEY`)* | optional | Tier C: pgvector semantic cache. Without one, semantic cache disables and exact-match cache still works. |
| `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` | recommended | Cache backend (pgvector). Apply `scripts/supabase_setup.sql` once. |

See [`.env.example`](.env.example).

## Ablation evaluation

`scripts/ablation_eval.py` runs each sample question through 4 configurations
(baseline / +rerank / +deepen / +rerank+deepen) and reports latency, evidence
counts, citation validity, and confidence per configuration — so the upgrades
are *demonstrated* not just claimed.

```
python scripts/ablation_eval.py --questions 1 2
```

## License

MIT (assignment / portfolio use).
