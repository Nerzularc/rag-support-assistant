"""Rules that decide when a human must take over, plus PII redaction for logs."""
from __future__ import annotations

import re
from dataclasses import dataclass

# Requests that should always go to a person, whatever the retrieval score.
ESCALATION_PATTERNS = {
    "asked_for_human": re.compile(r"\b(human|real person|agent|representative|speak to (someone|a person)|call me)\b", re.I),
    "legal_or_chargeback": re.compile(r"\b(lawyer|legal action|sue|chargeback|dispute (the )?charge|fraud|scam)\b", re.I),
    "safety": re.compile(r"\b(fire|smoke|smoking|burn(ed|t)?|electric shock|sparks?|injur(y|ed))\b", re.I),
    "angry": re.compile(r"\b(unacceptable|ridiculous|worst|furious|disgusting|never (buy|order) again)\b", re.I),
    "account_specific": re.compile(
        r"(#\d{4,}|\border (number|no\.?)\s*:?\s*\d+|\bwhere(?:'s| is) my (order|package|parcel|refund)"
        r"|\b(hasn'?t|haven'?t|didn'?t|never|not) (yet )?(arrived|been delivered|received|come))",
        re.I,
    ),
}

PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[card]"),
    (re.compile(r"\+?\d[\d\s().-]{7,}\d"), "[phone]"),
]


@dataclass
class Decision:
    handoff: bool
    reason: str | None
    confidence: float


def redact(text: str) -> str:
    for pattern, repl in PII_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def check_escalation(message: str) -> str | None:
    for reason, pattern in ESCALATION_PATTERNS.items():
        if pattern.search(message):
            return reason
    return None


def decide(message: str, top_score: float, coverage: float, min_score: float, min_coverage: float) -> Decision:
    """Combine escalation rules with retrieval confidence.

    confidence = 60% term coverage + 40% normalised BM25 score (capped at 1).
    """
    confidence = round(0.6 * coverage + 0.4 * min(top_score / (min_score * 3), 1.0), 3)
    reason = check_escalation(message)
    if reason:
        return Decision(True, reason, confidence)
    if top_score < min_score:
        return Decision(True, "no_relevant_article", confidence)
    if coverage < min_coverage:
        return Decision(True, "low_coverage", confidence)
    return Decision(False, None, confidence)
