"""SQLite conversation log and analytics. No ORM, no extra dependency."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    handoff INTEGER NOT NULL,
    handoff_reason TEXT,
    confidence REAL NOT NULL,
    sources TEXT NOT NULL,
    mode TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    helpful INTEGER
);
CREATE INDEX IF NOT EXISTS idx_interactions_created ON interactions(created_at);
"""


class Store:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def log(self, *, session_id: str, question: str, answer: str, handoff: bool,
            handoff_reason: str | None, confidence: float, sources: list[str],
            mode: str, latency_ms: int) -> str:
        iid = uuid.uuid4().hex[:12]
        with self._lock:
            self.conn.execute(
                "INSERT INTO interactions VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL)",
                (iid, session_id, datetime.now(timezone.utc).isoformat(), question, answer,
                 int(handoff), handoff_reason, confidence, json.dumps(sources), mode, latency_ms),
            )
            self.conn.commit()
        return iid

    def feedback(self, interaction_id: str, helpful: bool) -> bool:
        with self._lock:
            cur = self.conn.execute(
                "UPDATE interactions SET helpful=? WHERE id=?", (int(helpful), interaction_id)
            )
            self.conn.commit()
        return cur.rowcount == 1

    def analytics(self) -> dict:
        c = self.conn
        total = c.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        if total == 0:
            return {"total_questions": 0, "automation_rate": None, "handoff_rate": None,
                    "helpful_rate": None, "avg_latency_ms": None, "handoff_reasons": {},
                    "top_sources": [], "unanswered_questions": [], "daily": []}
        handoffs = c.execute("SELECT COUNT(*) FROM interactions WHERE handoff=1").fetchone()[0]
        rated = c.execute("SELECT COUNT(*), SUM(helpful) FROM interactions WHERE helpful IS NOT NULL").fetchone()
        latency = c.execute("SELECT AVG(latency_ms) FROM interactions").fetchone()[0]
        reasons = {r[0]: r[1] for r in c.execute(
            "SELECT handoff_reason, COUNT(*) FROM interactions WHERE handoff=1 GROUP BY 1 ORDER BY 2 DESC")}
        source_counts: dict[str, int] = {}
        for (src,) in c.execute("SELECT sources FROM interactions WHERE handoff=0"):
            for s in json.loads(src):
                source_counts[s] = source_counts.get(s, 0) + 1
        unanswered = [dict(r) for r in c.execute(
            "SELECT question, handoff_reason, COUNT(*) AS times FROM interactions WHERE handoff=1 "
            "GROUP BY lower(question) ORDER BY times DESC, MAX(created_at) DESC LIMIT 10")]
        daily = [dict(r) for r in c.execute(
            "SELECT substr(created_at,1,10) AS day, COUNT(*) AS questions, SUM(handoff) AS handoffs "
            "FROM interactions GROUP BY day ORDER BY day DESC LIMIT 30")]
        return {
            "total_questions": total,
            "automation_rate": round(1 - handoffs / total, 3),
            "handoff_rate": round(handoffs / total, 3),
            "helpful_rate": round(rated[1] / rated[0], 3) if rated[0] else None,
            "avg_latency_ms": round(latency or 0),
            "handoff_reasons": reasons,
            "top_sources": sorted(source_counts.items(), key=lambda x: -x[1])[:8],
            "unanswered_questions": unanswered,
            "daily": daily,
        }
