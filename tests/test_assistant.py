import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Force extractive mode and an isolated database for every test.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    import importlib

    import app.config as config
    import app.main as main

    importlib.reload(config)
    importlib.reload(main)
    main.get_assistant.cache_clear()
    from fastapi.testclient import TestClient

    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "extractive"
    assert body["chunks"] > 30


def test_answers_policy_question_with_citation(client):
    r = client.post("/api/chat", json={"message": "How long do refunds take?"})
    body = r.json()
    assert r.status_code == 200
    assert body["handoff"] is False
    assert "5 business days" in body["answer"]
    assert body["sources"][0]["id"] == "returns#4"
    assert "[1]" in body["answer"]


@pytest.mark.parametrize(
    "message,reason",
    [
        ("I want to talk to a real person", "asked_for_human"),
        ("Where is my order #55120?", "account_specific"),
        ("The machine is sparking and smells of smoke", "safety"),
        ("Do you sell matcha whisks?", "no_relevant_article"),
    ],
)
def test_handoff_rules(client, message, reason):
    body = client.post("/api/chat", json={"message": message}).json()
    assert body["handoff"] is True
    assert body["handoff_reason"] == reason
    assert body["sources"] == []


def test_feedback_and_analytics(client):
    a = client.post("/api/chat", json={"message": "Do you accept PayPal?"}).json()
    client.post("/api/chat", json={"message": "Can I speak to a human?"})
    assert client.post("/api/feedback", json={"interaction_id": a["interaction_id"], "helpful": True}).json()["ok"]
    assert client.post("/api/feedback", json={"interaction_id": "nope", "helpful": True}).status_code == 404

    stats = client.get("/api/analytics").json()
    assert stats["total_questions"] == 2
    assert stats["handoff_rate"] == 0.5
    assert stats["helpful_rate"] == 1.0
    assert stats["handoff_reasons"] == {"asked_for_human": 1}


def test_pii_is_redacted_in_logs(client):
    import app.main as main

    client.post("/api/chat", json={"message": "My email is jane@example.com and card 4111 1111 1111 1111, speak to agent"})
    row = main.get_assistant().store.conn.execute("SELECT question FROM interactions").fetchone()
    assert "jane@example.com" not in row["question"]
    assert "4111" not in row["question"]


def test_validation_rejects_empty_message(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_retrieval_eval_meets_quality_bar():
    from eval.run_eval import evaluate

    result = evaluate(verbose=False)
    assert result["hit@3"] >= 0.9
    assert result["handoff_accuracy"] >= 0.9


class _FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeClient:
    """Stands in for anthropic.Anthropic so the Claude code path is tested offline."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Resp", (), {"content": [_FakeBlock(self.reply)]})()


def test_claude_mode_uses_citations_and_context(client):
    import app.main as main

    a = main.get_assistant()
    fake = _FakeClient("Refunds are issued within 5 business days after inspection [1].")
    a.generator.client = fake
    body = client.post("/api/chat", json={"message": "How long do refunds take?"}).json()
    assert body["mode"] == "claude"
    assert body["handoff"] is False
    assert body["sources"][0]["id"] == "returns#4"
    sent = fake.calls[0]["messages"][-1]["content"]
    assert "[1] Returns & Refunds › Refund timing" in sent


def test_claude_handoff_token_escalates(client):
    import app.main as main

    a = main.get_assistant()
    a.generator.client = _FakeClient("[[HANDOFF]]")
    body = client.post("/api/chat", json={"message": "Can I return opened coffee beans?"}).json()
    assert body["handoff"] is True
    assert body["handoff_reason"] == "model_unsure"
