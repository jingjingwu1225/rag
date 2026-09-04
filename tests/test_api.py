"""
API tests. No network: the agent pipeline is stubbed, so these exercise the
HTTP layer itself — auth, validation, rate limiting, response shape, and SSE
framing — which is where this service's own bugs live.
"""

import json

import pytest
from fastapi.testclient import TestClient

import api
import history_store


@pytest.fixture
def client(monkeypatch):
    """A client with the agent stubbed out and a known corpus size."""
    fake_state = {
        "retrieved_chunks": [
            {"id": "p::1", "source": "paper.pdf", "distance": 0.5,
             "bm25_score": 3.0, "rerank_score": 10, "text": "chunk one"},
            {"id": "p::2", "source": "paper.pdf", "distance": 0.6,
             "bm25_score": None, "rerank_score": 8, "text": "chunk two"},
        ],
        "search_query": "resolved query",
        "sub_queries": ["resolved query"],
        "retry_count": 1,
    }

    monkeypatch.setattr(api, "prepare_turn", lambda q, h: dict(fake_state))
    # Markdown with newlines on purpose — the SSE framing has to survive it.
    monkeypatch.setattr(
        api, "stream_answer",
        lambda q, s, on_complete=None: iter(["- first\n", "- second\n\n", "done."]),
    )
    monkeypatch.setattr(api, "_CORPUS_SIZE", 140)
    monkeypatch.setattr(api, "API_KEY", "testkey")
    monkeypatch.setattr(api, "RATE_LIMIT_PER_MIN", 1000)
    api._hits.clear()

    # TestClient runs the lifespan, which warms the real Chroma index and
    # validates config. Both are stubbed so these tests exercise the HTTP
    # layer only — and, importantly, so the suite needs no credentials and no
    # ambient .env. (Depending on a developer's local .env is what let an
    # import-time config failure reach CI unnoticed.)
    monkeypatch.setattr(api.rag_core, "warm_caches", lambda: 140)
    monkeypatch.setattr(api.rag_core, "validate_config", lambda: None)
    with TestClient(api.app) as c:
        yield c


class TestStartupChecks:
    """The lifespan's job is to refuse to start when something is wrong."""

    def test_startup_fails_on_missing_config(self, monkeypatch):
        def boom():
            raise RuntimeError("OPENAI_API_KEY is not set")

        monkeypatch.setattr(api.rag_core, "validate_config", boom)
        monkeypatch.setattr(api.rag_core, "warm_caches", lambda: 140)

        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            with TestClient(api.app):
                pass

    def test_startup_fails_on_empty_corpus(self, monkeypatch):
        """
        An empty collection does not raise at query time — it just turns every
        answer into "I don't know" with nothing in the logs. Refusing to start
        converts that silent quality collapse into an obvious deploy failure.
        """
        monkeypatch.setattr(api.rag_core, "validate_config", lambda: None)
        monkeypatch.setattr(api.rag_core, "warm_caches", lambda: 0)

        with pytest.raises(RuntimeError, match="empty"):
            with TestClient(api.app):
                pass


AUTH = {"x-api-key": "testkey"}


class TestHealth:
    def test_health_needs_no_auth(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ready_reports_corpus_size(self, client):
        body = client.get("/ready").json()
        assert body["status"] == "ready"
        assert body["corpus_chunks"] == 140


class TestAuthAndValidation:
    def test_missing_api_key_is_rejected(self, client):
        response = client.post("/ask", json={"question": "hi"})
        assert response.status_code == 401

    def test_wrong_api_key_is_rejected(self, client):
        response = client.post("/ask", json={"question": "hi"}, headers={"x-api-key": "nope"})
        assert response.status_code == 401

    def test_empty_question_is_rejected(self, client):
        assert client.post("/ask", json={"question": ""}, headers=AUTH).status_code == 422

    def test_overlong_question_is_rejected(self, client):
        """Caps prompt size — an unbounded question is unbounded spend."""
        response = client.post("/ask", json={"question": "x" * 5000}, headers=AUTH)
        assert response.status_code == 422

    def test_rate_limit_returns_429(self, client, monkeypatch):
        monkeypatch.setattr(api, "RATE_LIMIT_PER_MIN", 2)
        api._hits.clear()
        for _ in range(2):
            assert client.post("/ask", json={"question": "hi"}, headers=AUTH).status_code == 200
        assert client.post("/ask", json={"question": "hi"}, headers=AUTH).status_code == 429


class TestAsk:
    def test_returns_answer_and_grouped_sources(self, client):
        body = client.post("/ask", json={"question": "hi"}, headers=AUTH).json()

        assert body["answer"] == "- first\n- second\n\ndone."
        assert body["retry_count"] == 1
        assert body["search_query"] == "resolved query"
        # Two chunks from one file group into a single source entry.
        assert len(body["sources"]) == 1
        assert body["sources"][0] == {
            "source": "paper.pdf", "count": 2, "rerank_scores": [10, 8],
        }

    def test_generates_thread_id_when_absent(self, client):
        body = client.post("/ask", json={"question": "hi"}, headers=AUTH).json()
        assert body["thread_id"]

    def test_reuses_supplied_thread_id_and_persists_history(self, client):
        thread_id = "explicit-thread"
        history_store.reset(thread_id)
        try:
            body = client.post(
                "/ask", json={"question": "hi", "thread_id": thread_id}, headers=AUTH
            ).json()
            assert body["thread_id"] == thread_id
            # The turn must be persisted, or follow-ups lose their context.
            assert len(history_store.get_history(thread_id)) == 2
        finally:
            history_store.reset(thread_id)


class TestStreaming:
    def _events(self, raw: str) -> list[tuple[str, dict]]:
        events = []
        for block in raw.strip().split("\n\n"):
            name = payload = None
            for line in block.splitlines():
                if line.startswith("event: "):
                    name = line[7:]
                elif line.startswith("data: "):
                    payload = json.loads(line[6:])  # raises if framing broke
            if name:
                events.append((name, payload))
        return events

    def test_event_sequence(self, client):
        raw = client.post("/ask/stream", json={"question": "hi"}, headers=AUTH).text
        names = [n for n, _ in self._events(raw)]

        assert names[0] == "status"
        assert "sources" in names
        assert "token" in names
        assert names[-1] == "done"
        assert "error" not in names

    def test_tokens_with_newlines_survive_framing(self, client):
        """
        The bug this guards: SSE frames are blank-line delimited and answers
        are markdown full of newlines, so an unencoded token shatters the
        stream into garbage frames.
        """
        raw = client.post("/ask/stream", json={"question": "hi"}, headers=AUTH).text
        tokens = [d["t"] for n, d in self._events(raw) if n == "token"]

        assert "".join(tokens) == "- first\n- second\n\ndone."
        assert any("\n" in t for t in tokens), "test is meaningless without a newline token"

    def test_stream_requires_auth(self, client):
        assert client.post("/ask/stream", json={"question": "hi"}).status_code == 401
