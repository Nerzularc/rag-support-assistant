"""Answer generation: Claude when an API key is set, extractive fallback otherwise."""
from __future__ import annotations

import logging
import re

from .retriever import Hit, tokenize

log = logging.getLogger(__name__)

HANDOFF_TOKEN = "[[HANDOFF]]"

SYSTEM_PROMPT = """You are the customer support assistant for {store}.

Rules:
- Answer ONLY from the numbered help-center excerpts provided. If they do not
  contain the answer, reply with exactly {handoff} and nothing else.
- Be concise: 2-5 sentences, friendly, no marketing language.
- Cite the excerpts you used with their number in square brackets, e.g. [1] or [1][3].
- Never invent prices, dates, policies, order details or links.
- Never ask for card numbers or passwords.
- If the customer writes in another language, answer in that language."""


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] {h.chunk.citation}\n{h.chunk.text}" for i, h in enumerate(hits, start=1)
    )


def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 3) -> str:
    """Pick the sentences that best overlap the question from the top chunks.

    Used in mock mode (no API key) and as a fallback if the API call fails.
    """
    q = set(tokenize(question))
    scored: list[tuple[float, int, int, str]] = []
    for ci, h in enumerate(hits[:2], start=1):
        sentences = re.split(r"(?<=[.!?])\s+", h.chunk.text)
        for si, s in enumerate(sentences):
            overlap = len(q & set(tokenize(s)))
            if overlap:
                # Prefer earlier chunks and earlier sentences on ties.
                scored.append((overlap - 0.1 * (ci - 1), -si, ci, s))
    if not scored:
        return HANDOFF_TOKEN
    # Only borrow from the second chunk if it matches the question better than
    # anything in the top chunk; otherwise answers drift off-topic.
    best_first = max((s[0] for s in scored if s[2] == 1), default=0)
    scored = [s for s in scored if s[2] == 1 or s[0] > best_first]
    scored.sort(reverse=True)
    picked = sorted(scored[:max_sentences], key=lambda x: (x[2], -x[1]))
    return " ".join(f"{s} [{ci}]" for _, _, ci, s in picked)


class AnswerGenerator:
    def __init__(self, api_key: str, model: str, store_name: str):
        self.model = model
        self.store_name = store_name
        self.client = None
        if api_key:
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=30)
            except ImportError:  # pragma: no cover
                log.warning("anthropic package not installed; using extractive mode")

    @property
    def mode(self) -> str:
        return "claude" if self.client else "extractive"

    def answer(self, question: str, hits: list[Hit], history: list[dict] | None = None) -> str:
        if not self.client:
            return extractive_answer(question, hits)
        messages = list(history or [])[-6:]
        messages.append(
            {
                "role": "user",
                "content": f"Help-center excerpts:\n\n{build_context(hits)}\n\nCustomer question: {question}",
            }
        )
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                temperature=0,
                system=SYSTEM_PROMPT.format(store=self.store_name, handoff=HANDOFF_TOKEN),
                messages=messages,
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
            return text or HANDOFF_TOKEN
        except Exception as exc:  # network, rate limit, auth...
            log.error("Claude call failed, falling back to extractive answer: %s", exc)
            return extractive_answer(question, hits)


def cited_indices(answer: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
