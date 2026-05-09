"""Command-line runner.

Usage:
  python cli.py "What is the evidence for SGLT2i in HFpEF?"
  python cli.py --no-cache "..."
  python cli.py --web "..."           # also use Tavily web search
  python cli.py --json "..."          # emit JSON instead of pretty text
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap

from src.pipeline import Pipeline


def render_pretty(r) -> str:
    bar = "─" * 72
    out = [
        bar,
        f"QUESTION: {r.question}",
        bar,
        "",
        "ANSWER:",
        textwrap.fill(r.answer, width=88, replace_whitespace=False, drop_whitespace=False)
            if "\n" not in r.answer else r.answer,
        "",
        f"CONFIDENCE: {r.confidence}",
    ]
    if r.limitations:
        out.append("")
        out.append("LIMITATIONS:")
        out.append(textwrap.fill(r.limitations, width=88))
    if r.citations:
        out.append("")
        out.append(f"CITATIONS ({len(r.citations)}):")
        for c in r.citations:
            tag = f"[{c.id}]"
            title = textwrap.shorten(c.title, width=80, placeholder="…")
            out.append(f"  {tag:<22}  {title}")
            out.append(f"  {' ':<22}  {c.url}")
            if c.summary:
                out.append(f"  {' ':<22}  → {textwrap.shorten(c.summary, width=80, placeholder='…')}")
    if r.metadata:
        m = r.metadata
        out.append("")
        out.append(
            f"⏱  retrieval: {m.get('retrieval_seconds')}s | synthesis: {m.get('synthesis_seconds')}s | total: {m.get('total_seconds')}s"
        )
        out.append(
            f"📚 PubMed:{m.get('n_pubmed', 0)} | Trials:{m.get('n_trials', 0)} | "
            f"Preprints:{m.get('n_preprints', 0)} | Web:{m.get('n_web', 0)} | from_cache={m.get('from_cache')}"
        )
    out.append("")
    out.append("⚠ " + r.disclaimer)
    out.append(bar)
    return "\n".join(out)


async def amain(args):
    p = Pipeline()
    r = await p.ask(
        question=args.question,
        use_web_search=args.web,
        use_cache=not args.no_cache,
    )
    if args.json:
        print(json.dumps(r.model_dump(), indent=2, ensure_ascii=False))
    else:
        print(render_pretty(r))


def main():
    ap = argparse.ArgumentParser(description="Evidence-Grounded Clinical Chatbot CLI")
    ap.add_argument("question", help="Clinical question to ask")
    ap.add_argument("--web", action="store_true", help="Also include Tavily web search")
    ap.add_argument("--no-cache", action="store_true", help="Bypass Supabase cache")
    ap.add_argument("--json", action="store_true", help="Emit JSON")
    args = ap.parse_args()
    if not args.question.strip():
        ap.error("question is required")
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
