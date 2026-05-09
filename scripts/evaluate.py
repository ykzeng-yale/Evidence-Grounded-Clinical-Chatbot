"""Evaluation harness.

Runs the pipeline against examples/sample_questions.json and reports:
  - latency (retrieval / synthesis / total)
  - evidence counts per source
  - citation coverage = (# bracketed cites in answer) / (# evidence items)
  - hallucination flag = inline cite IDs that don't appear in retrieved evidence
  - confidence distribution

This is a lightweight automated proxy for quality. A real eval would score
against gold-standard clinician answers; we surface the metrics that catch
the most common failure modes (no citations, made-up cites, runaway latency).
"""
from __future__ import annotations

import asyncio
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import Pipeline  # noqa: E402

CITE_RE = re.compile(r"\[(PMID|NCT|PMC|DOI|WEB|EPMC):([^\]\s]+)\]")


def analyze(question: str, response) -> dict:
    answer = response.answer or ""
    cites_in_text = CITE_RE.findall(answer)
    cite_ids_in_text = {f"{k}:{v}" for k, v in cites_in_text}
    evidence_ids = {e.id for e in response.evidence}
    # NCT ids appear in evidence as "NCT:NCT01234567" but in answer often just "[NCT01234567]"
    # Normalize to NCT:NCT... if needed
    norm_text_ids = set()
    for cid in cite_ids_in_text:
        if cid.startswith("NCT:") and not cid.startswith("NCT:NCT"):
            norm_text_ids.add(f"NCT:{cid.split(':',1)[1]}" if cid.split(':',1)[1].startswith("NCT")
                              else f"NCT:{cid.split(':',1)[1]}")
        else:
            norm_text_ids.add(cid)
    valid = norm_text_ids & evidence_ids
    hallucinated = norm_text_ids - evidence_ids
    return {
        "question": question,
        "n_cites_in_answer": len(cites_in_text),
        "unique_cited_ids": len(norm_text_ids),
        "valid_citations": len(valid),
        "hallucinated_citations": sorted(hallucinated),
        "n_evidence": len(response.evidence),
        "n_pubmed": (response.metadata or {}).get("n_pubmed", 0),
        "n_trials": (response.metadata or {}).get("n_trials", 0),
        "n_preprints": (response.metadata or {}).get("n_preprints", 0),
        "confidence": response.confidence,
        "retrieval_s": (response.metadata or {}).get("retrieval_seconds"),
        "synthesis_s": (response.metadata or {}).get("synthesis_seconds"),
        "total_s": (response.metadata or {}).get("total_seconds"),
        "answer_len": len(answer),
        "has_limitations": bool(response.limitations),
    }


async def amain(limit: int | None = None, no_cache: bool = True):
    samples = json.loads((ROOT / "examples" / "sample_questions.json").read_text())
    if limit:
        samples = samples[:limit]
    p = Pipeline()
    rows = []
    for s in samples:
        q = s["question"]
        print(f"\n▶ {s['label']}: {q}")
        try:
            r = await p.ask(q, use_cache=not no_cache)
            row = analyze(q, r)
            rows.append(row)
            print(
                f"  ✓ {row['total_s']}s · "
                f"{row['n_pubmed']}P+{row['n_trials']}T+{row['n_preprints']}pre · "
                f"cited {row['valid_citations']}/{row['unique_cited_ids']} valid · "
                f"conf={row['confidence']}"
            )
            if row["hallucinated_citations"]:
                print(f"  ⚠ HALLUCINATED: {row['hallucinated_citations']}")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            rows.append({"question": q, "error": str(e)})

    # Summary
    ok = [r for r in rows if "error" not in r]
    print("\n" + "=" * 72)
    print(f"RESULTS: {len(ok)}/{len(rows)} succeeded")
    if ok:
        totals = [r["total_s"] for r in ok if r["total_s"] is not None]
        if totals:
            print(f"  latency p50/p95: {statistics.median(totals):.1f}s / "
                  f"{sorted(totals)[max(0, int(0.95*len(totals))-1)]:.1f}s")
        valid_rate = sum(r["valid_citations"] for r in ok) / max(1, sum(r["unique_cited_ids"] for r in ok))
        print(f"  citation validity: {valid_rate*100:.1f}%")
        hallucinated_runs = sum(1 for r in ok if r["hallucinated_citations"])
        print(f"  runs with any hallucinated citations: {hallucinated_runs}/{len(ok)}")
        conf_dist = {}
        for r in ok:
            conf_dist[r["confidence"]] = conf_dist.get(r["confidence"], 0) + 1
        print(f"  confidence distribution: {conf_dist}")
    print("=" * 72)

    out = ROOT / "scripts" / "eval_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Run only the first N samples")
    ap.add_argument("--use-cache", action="store_true", help="Allow cache hits (faster, less informative)")
    args = ap.parse_args()
    asyncio.run(amain(limit=args.limit, no_cache=not args.use_cache))
