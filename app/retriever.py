"""Knowledge-base loading, chunking and BM25 retrieval.

BM25 is used instead of embeddings on purpose: for a help center of a few
hundred sections it is fast, free, deterministic, explainable ("these words
matched") and needs no vector database. The `Retriever` interface makes it easy
to swap in embeddings (e.g. Voyage + pgvector) when the corpus grows.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOPWORDS = frozenset(
    """a an and are as at be been but by can could do does did for from have has had how i if in
    into is it its me my no not of on or our so than that the their them then there these they this
    to too us was we were what when where which who why will with would you your yours i'm im it's
    please hi hello thanks thank any some about after before just also get got want need should""".split()
)

SYNONYMS = {
    "refund": ["return", "money", "back"],
    "return": ["refund"],
    "delivery": ["shipping", "ship", "arrive"],
    "arrive": ["delivery", "shipping"],
    "swap": ["exchange"],
    "shipping": ["delivery", "ship"],
    "ship": ["shipping", "delivery"],
    "cancel": ["cancelling", "cancellation"],
    "broken": ["faulty", "damaged", "defect"],
    "faulty": ["defect", "broken"],
    "damaged": ["broken", "faulty"],
    "machine": ["espresso"],
    "pay": ["payment"],
    "price": ["cost", "pricing"],
    "cost": ["price", "pricing"],
    "sub": ["subscription"],
    "guarantee": ["warranty"],
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def stem(token: str) -> str:
    """Tiny suffix stripper; good enough for a support corpus.

    shipping -> ship, packed -> pack, exchanges -> exchang, exchange -> exchang,
    boxes -> box, arrives -> arriv.
    """
    t = token
    if len(t) > 4:
        if t.endswith("ing"):
            t = t[:-3]
        elif t.endswith("ed"):
            t = t[:-2]
        elif t.endswith("es") and t[-3] in "sxz" or t.endswith(("ches", "shes")):
            t = t[:-2]
        elif t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        if len(t) > 3 and t[-1] == t[-2] and t[-1] not in "aeiouls":
            t = t[:-1]  # shipp -> ship
    if len(t) > 4 and t.endswith("e"):
        t = t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


STEMMED_SYNONYMS = {stem(k): [stem(v) for v in vals] for k, vals in SYNONYMS.items()}


def expand(tokens: list[str]) -> list[str]:
    out = list(tokens)
    for t in tokens:
        out.extend(STEMMED_SYNONYMS.get(t, []))
    return out


@dataclass
class Chunk:
    id: str
    doc: str
    title: str
    section: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.title} › {self.section}"


def load_chunks(kb_dir: Path) -> list[Chunk]:
    """Split every markdown file into one chunk per `##` section."""
    chunks: list[Chunk] = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title_match = re.search(r"^#\s+(.+)$", raw, re.M)
        title = title_match.group(1).strip() if title_match else path.stem
        parts = re.split(r"^##\s+", raw, flags=re.M)[1:]
        for i, part in enumerate(parts):
            heading, _, body = part.partition("\n")
            body = " ".join(body.split())
            if body:
                chunks.append(
                    Chunk(
                        id=f"{path.stem}#{i}",
                        doc=path.stem,
                        title=title,
                        section=heading.strip(),
                        text=body,
                    )
                )
    if not chunks:
        raise ValueError(f"No markdown sections found in {kb_dir}")
    return chunks


@dataclass
class Hit:
    chunk: Chunk
    score: float
    matched_terms: list[str]


class Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        # Section headings are weighted x2 by repeating them.
        self.docs = [tokenize(f"{c.section} {c.section} {c.title} {c.text}") for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 4) -> list[Hit]:
        terms = expand(tokenize(query))
        if not terms:
            return []
        hits: list[Hit] = []
        for i, tf in enumerate(self.tf):
            score = 0.0
            matched = []
            for t in set(terms):
                if t not in tf:
                    continue
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                score += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
                matched.append(t)
            if score > 0:
                hits.append(Hit(self.chunks[i], score, sorted(matched)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    @staticmethod
    def coverage(query: str, hits: list[Hit]) -> float:
        """Share of the query's own key terms found anywhere in the retrieved chunks."""
        q_terms = set(tokenize(query))
        if not q_terms or not hits:
            return 0.0
        found = set()
        for h in hits:
            body = set(tokenize(f"{h.chunk.section} {h.chunk.text}"))
            for t in q_terms:
                if t in body or any(s in body for s in STEMMED_SYNONYMS.get(t, [])):
                    found.add(t)
        return len(found) / len(q_terms)
