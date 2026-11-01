"""FastAPI service: chat endpoint, feedback, analytics and the demo UI."""
from __future__ import annotations

import time
import uuid
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import ROOT, Settings, settings
from .guardrails import decide, redact
from .llm import HANDOFF_TOKEN, AnswerGenerator, cited_indices
from .retriever import Retriever, load_chunks
from .store import Store

HANDOFF_MESSAGE = (
    "I want to make sure you get the right answer, so I'm passing this to our support team. "
    "A person will reply by email within 1 business day (Mon–Fri, 9 AM–6 PM ET)."
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=64)
    history: list[dict] = Field(default_factory=list, max_length=12)


class Source(BaseModel):
    id: str
    title: str
    section: str
    score: float


class ChatResponse(BaseModel):
    interaction_id: str
    session_id: str
    answer: str
    handoff: bool
    handoff_reason: str | None
    confidence: float
    sources: list[Source]
    mode: str


class FeedbackRequest(BaseModel):
    interaction_id: str
    helpful: bool


class Assistant:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.chunks = load_chunks(cfg.kb_dir)
        self.retriever = Retriever(self.chunks)
        self.generator = AnswerGenerator(cfg.anthropic_api_key, cfg.model, cfg.store_name)
        self.store = Store(cfg.db_path)

    def chat(self, req: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        session_id = req.session_id or uuid.uuid4().hex[:12]
        question = req.message.strip()

        hits = self.retriever.search(question, k=self.cfg.top_k)
        top_score = hits[0].score if hits else 0.0
        coverage = self.retriever.coverage(question, hits[:2])
        decision = decide(question, top_score, coverage, self.cfg.min_score, self.cfg.min_coverage)

        answer, reason = HANDOFF_MESSAGE, decision.reason
        sources: list[Source] = []
        if not decision.handoff:
            history = [m for m in req.history if m.get("role") in ("user", "assistant") and m.get("content")]
            raw = self.generator.answer(question, hits, history)
            if HANDOFF_TOKEN in raw:
                reason = "model_unsure"
            else:
                answer = raw
                used = cited_indices(raw) or [1]
                sources = [
                    Source(id=hits[i - 1].chunk.id, title=hits[i - 1].chunk.title,
                           section=hits[i - 1].chunk.section, score=round(hits[i - 1].score, 2))
                    for i in used if 0 < i <= len(hits)
                ]
        handoff = reason is not None

        latency = int((time.perf_counter() - started) * 1000)
        iid = self.store.log(
            session_id=session_id, question=redact(question), answer=answer, handoff=handoff,
            handoff_reason=reason, confidence=decision.confidence,
            sources=[s.id for s in sources], mode=self.generator.mode, latency_ms=latency,
        )
        return ChatResponse(
            interaction_id=iid, session_id=session_id, answer=answer, handoff=handoff,
            handoff_reason=reason, confidence=decision.confidence, sources=sources,
            mode=self.generator.mode,
        )


@lru_cache
def get_assistant() -> Assistant:
    return Assistant(settings)


app = FastAPI(title="RAG Support Assistant", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/admin")
def admin() -> FileResponse:
    return FileResponse(ROOT / "static" / "admin.html")


@app.get("/health")
def health() -> dict:
    a = get_assistant()
    return {"status": "ok", "mode": a.generator.mode, "chunks": len(a.chunks)}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    return get_assistant().chat(req)


@app.post("/api/feedback")
def feedback(req: FeedbackRequest) -> dict:
    if not get_assistant().store.feedback(req.interaction_id, req.helpful):
        raise HTTPException(status_code=404, detail="interaction not found")
    return {"ok": True}


@app.get("/api/analytics")
def analytics() -> dict:
    return get_assistant().store.analytics()
