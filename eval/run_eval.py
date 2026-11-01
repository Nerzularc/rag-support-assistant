"""Offline evaluation of retrieval quality and handoff decisions.

Usage:  python -m eval.run_eval            (from the repo root)

Metrics
- hit@1 / hit@3   : expected section is the first / within top 3 results
- MRR             : mean reciprocal rank of the expected section
- handoff accuracy: bot answered when it should and escalated when it should
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.guardrails import decide  # noqa: E402
from app.retriever import Retriever, load_chunks  # noqa: E402


def evaluate(verbose: bool = True) -> dict:
    retriever = Retriever(load_chunks(settings.kb_dir))
    rows = [json.loads(l) for l in (ROOT / "eval" / "questions.jsonl").read_text().splitlines() if l.strip()]
    hit1 = hit3 = rr = answerable = correct_handoff = 0
    failures = []
    for r in rows:
        hits = retriever.search(r["q"], k=settings.top_k)
        ids = [h.chunk.id for h in hits]
        cov = retriever.coverage(r["q"], hits[:2])
        d = decide(r["q"], hits[0].score if hits else 0.0, cov, settings.min_score, settings.min_coverage)
        if d.handoff == r["handoff"]:
            correct_handoff += 1
        else:
            failures.append(f"HANDOFF  expected={r['handoff']} got={d.handoff} ({d.reason}, cov={cov:.2f}) :: {r['q']}")
        if r["expect"]:
            answerable += 1
            if ids[:1] == [r["expect"]]:
                hit1 += 1
            if r["expect"] in ids[:3]:
                hit3 += 1
            if r["expect"] in ids:
                rr += 1 / (ids.index(r["expect"]) + 1)
            else:
                failures.append(f"RETRIEVAL expected={r['expect']} got={ids[:3]} :: {r['q']}")
    result = {
        "questions": len(rows),
        "hit@1": round(hit1 / answerable, 3),
        "hit@3": round(hit3 / answerable, 3),
        "mrr": round(rr / answerable, 3),
        "handoff_accuracy": round(correct_handoff / len(rows), 3),
    }
    if verbose:
        print(json.dumps(result, indent=2))
        for f in failures:
            print(" -", f)
    return result


if __name__ == "__main__":
    evaluate()
