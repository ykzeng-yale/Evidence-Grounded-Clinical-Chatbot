"""Scientific ablation: compare baseline vs +rerank vs +deepen vs +all on the
sample question set. Reports latency, evidence diversity, citation validity,
and confidence per configuration so we can SHOW (not just claim) the upgrades
help.

Usage:
  python scripts/ablation_eval.py
  python scripts/ablation_eval.py --questions 1 2 4    # subset
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import Pipeline  # noqa: E402

CITE_RE = re.compile(r"\[(PMID|NCT|PMC|DOI|WEB|EPMC|ARXIV|PCLIP):([^\]\s]+)\]")

CONFIGS = [
    ("baseline",     {"rerank": False, "deepen": False}),
    ("+rerank",      {"rerank": True,  "deepen": False}),
    ("+deepen",      {"rerank": False, "deepen": True}),
    ("+rerank+deepen", {"rerank": True, "deepen": True}),
]


@dataclass
class Run:
    config: str
    question: str
    total_s: float
    retrieve_s: float
    rerank_s: float
    synth_s: float
    deepen_s: float
    n_evidence: int
    n_sources: int                 # how many distinct source buckets
    answer_len: int
    n_cites_in_answer: int
    cite_validity_pct: float
    confidence: str
    deepened: list = field(default_factory=list)
    error: str | None = None


def analyze(answer: str, evidence_ids: set[str]) -> tuple[int, float]:
    cites = CITE_RE.findall(answer)
    if not cites:
        return 0, 0.0
    norm = set()
    for kind, val in cites:
        if kind == "NCT" and not val.startswith("NCT"):
            val = "NCT" + val
        norm.add(f"{kind}:{val}")
    valid = norm & evidence_ids
    return len(norm), 100.0 * len(valid) / len(norm)


async def run_one(p: Pipeline, q: str, label: str, opts: dict) -> Run:
    try:
        t = time.perf_counter()
        r = await p.ask(q, use_cache=False, **opts)
        ev_ids = {e.id for e in r.evidence}
        n_cites, validity = analyze(r.answer, ev_ids)
        sources = {e.source for e in r.evidence}
        m = r.metadata or {}
        return Run(
            config=label,
            question=q,
            total_s=m.get("total_seconds") or round(time.perf_counter() - t, 2),
            retrieve_s=m.get("retrieval_seconds") or 0,
            rerank_s=m.get("rerank_seconds") or 0,
            synth_s=m.get("synthesis_seconds") or 0,
            deepen_s=m.get("deepen_seconds") or 0,
            n_evidence=len(r.evidence),
            n_sources=len(sources),
            answer_len=len(r.answer),
            n_cites_in_answer=n_cites,
            cite_validity_pct=round(validity, 1),
            confidence=r.confidence,
            deepened=m.get("deepened_citations") or [],
        )
    except Exception as e:
        return Run(label, q, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, "low", error=str(e))


async def amain(question_indices: list[int] | None):
    samples = json.loads((ROOT / "examples" / "sample_questions.json").read_text())
    if question_indices:
        samples = [samples[i - 1] for i in question_indices if 1 <= i <= len(samples)]
    p = Pipeline()
    print(f"Running {len(CONFIGS)} configs × {len(samples)} questions = "
          f"{len(CONFIGS)*len(samples)} runs")

    all_runs: List[Run] = []
    for s in samples:
        q = s["question"]
        print(f"\n══ {s['label']}: {q[:60]}…")
        for label, opts in CONFIGS:
            r = await run_one(p, q, label, opts)
            all_runs.append(r)
            err = f"  ✗ {r.error}" if r.error else ""
            print(f"  {label:<20} {r.total_s:>6.1f}s  "
                  f"ev={r.n_evidence:>2} src={r.n_sources}  cites={r.n_cites_in_answer:>2} "
                  f"valid={r.cite_validity_pct:>5.1f}%  conf={r.confidence}{err}")

    # Aggregate by config
    print("\n" + "═" * 78)
    print("AGGREGATE (mean across questions)")
    print(f"{'config':<22}{'lat_s':>8}{'ev':>5}{'cites':>7}{'valid%':>9}{'conf_high%':>13}")
    for label, _ in CONFIGS:
        runs = [r for r in all_runs if r.config == label and not r.error]
        if not runs:
            continue
        lat = statistics.mean(r.total_s for r in runs)
        ev = statistics.mean(r.n_evidence for r in runs)
        cit = statistics.mean(r.n_cites_in_answer for r in runs)
        val = statistics.mean(r.cite_validity_pct for r in runs)
        high = 100.0 * sum(1 for r in runs if r.confidence == "high") / len(runs)
        print(f"{label:<22}{lat:>8.1f}{ev:>5.1f}{cit:>7.1f}{val:>9.1f}{high:>13.1f}")
    print("═" * 78)

    out = ROOT / "scripts" / "ablation_results.json"
    out.write_text(json.dumps([asdict(r) for r in all_runs], indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, nargs="*", help="1-based indices to run (subset)")
    args = ap.parse_args()
    asyncio.run(amain(args.questions))
