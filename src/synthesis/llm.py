"""Claude synthesizer with two paths:
  - synthesize()        → one-shot, structured (tool_use), used by CLI/API/eval
  - synthesize_stream() → SSE streaming for the web UI

Both use Anthropic Claude (default: claude-sonnet-4-6 — best quality/cost balance).
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from anthropic import AsyncAnthropic

from src.models import Evidence
from src.synthesis.prompts import (
    SYSTEM_PROMPT,
    SYNTHESIS_TOOL,
    REFORMULATE_TOOL,
    build_user_prompt,
    build_reformulate_prompt,
)

log = logging.getLogger(__name__)


class ClaudeSynthesizer:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required")
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    # ---------- Query reformulation ----------
    async def reformulate(self, question: str) -> dict:
        """Convert natural-language question to retrieval-friendly queries.
        Returns {pubmed_query, trials_query, intent}. On failure, falls back
        to the raw question for both."""
        try:
            msg = await self.client.messages.create(
                model=self.model,
                max_tokens=400,
                system=(
                    "You convert clinical questions into precise retrieval queries. "
                    "Always call the reformulate_query tool."
                ),
                tools=[REFORMULATE_TOOL],
                tool_choice={"type": "tool", "name": "reformulate_query"},
                messages=[{"role": "user", "content": build_reformulate_prompt(question)}],
            )
            for block in msg.content:
                if getattr(block, "type", None) == "tool_use" and block.name == "reformulate_query":
                    return dict(block.input)
        except Exception as e:
            log.warning(f"Reformulate failed, using raw question: {e}")
        return {"pubmed_query": question, "trials_query": question, "intent": "other"}

    # ---------- One-shot structured synthesis ----------
    async def synthesize(self, question: str, evidence: list[Evidence]) -> dict:
        """Returns {answer, key_evidence, limitations, confidence}."""
        msg = await self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=[SYNTHESIS_TOOL],
            tool_choice={"type": "tool", "name": "submit_clinical_answer"},
            messages=[{"role": "user", "content": build_user_prompt(question, evidence)}],
        )
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "submit_clinical_answer":
                return dict(block.input)
        # Fallback if model returned plain text
        text = "".join(getattr(b, "text", "") for b in msg.content)
        return {
            "answer": text or "(synthesis returned no structured output)",
            "key_evidence": [],
            "limitations": "Model failed to return structured output.",
            "confidence": "low",
        }

    # ---------- Streaming synthesis (for SSE) ----------
    async def synthesize_stream(
        self, question: str, evidence: list[Evidence]
    ) -> AsyncIterator[dict]:
        """Stream synthesis. Yields events:
          {"type": "text_delta", "text": "..."}
          {"type": "structured", "data": {answer, key_evidence, limitations, confidence}}

        Implementation note: tool_use streaming yields partial JSON via
        input_json_delta events. We accumulate them and parse at message_stop.
        """
        accumulated_json_parts: list[str] = []
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=[SYNTHESIS_TOOL],
            tool_choice={"type": "tool", "name": "submit_clinical_answer"},
            messages=[{"role": "user", "content": build_user_prompt(question, evidence)}],
        ) as stream:
            async for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    dtype = getattr(delta, "type", None)
                    if dtype == "input_json_delta":
                        partial = getattr(delta, "partial_json", "") or ""
                        accumulated_json_parts.append(partial)
                        # Stream the *answer* field as it grows so the UI feels alive.
                        partial_text = _try_extract_answer_partial("".join(accumulated_json_parts))
                        if partial_text is not None:
                            yield {"type": "text_delta", "text_partial": partial_text}
                    elif dtype == "text_delta":
                        # In case the model emits text outside the tool (rare with tool_choice)
                        yield {"type": "text_delta", "text": getattr(delta, "text", "")}
        # Parse final JSON
        full = "".join(accumulated_json_parts).strip()
        try:
            data = json.loads(full) if full else {}
        except json.JSONDecodeError:
            data = {}
        if not data:
            data = {
                "answer": "(synthesis returned no structured output)",
                "key_evidence": [],
                "limitations": "Model failed to return structured output.",
                "confidence": "low",
            }
        # Ensure required fields exist
        data.setdefault("key_evidence", [])
        data.setdefault("limitations", "")
        data.setdefault("confidence", "moderate")
        yield {"type": "structured", "data": data}


def _try_extract_answer_partial(json_buf: str) -> str | None:
    """Best-effort: pull out a growing `"answer": "..."` field from a partial JSON
    blob so we can stream it to the UI before the full tool-use is closed.

    Returns the partial answer string, unescaping common sequences. Returns None
    if no `answer` field has started yet.
    """
    key = '"answer"'
    i = json_buf.find(key)
    if i < 0:
        return None
    # find the opening quote of the value
    j = json_buf.find('"', i + len(key))
    if j < 0:
        return None
    j += 1  # past the opening quote
    # walk chars, respecting escapes; stop at unescaped closing quote
    out = []
    k = j
    while k < len(json_buf):
        ch = json_buf[k]
        if ch == "\\" and k + 1 < len(json_buf):
            nxt = json_buf[k + 1]
            esc = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}.get(nxt)
            if esc is not None:
                out.append(esc)
                k += 2
                continue
            # leave \uXXXX etc as-is for partial; finalization will handle
            out.append(ch)
            k += 1
            continue
        if ch == '"':
            # closed
            break
        out.append(ch)
        k += 1
    return "".join(out)
