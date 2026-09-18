from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Fresh app instance per test, auth OFF by default (blank API_KEY)."""
    monkeypatch.setattr("src.config.API_KEY", "")
    import importlib

    import src.api as api_module
    importlib.reload(api_module)
    return TestClient(api_module.app)


@pytest.fixture
def authed_client(monkeypatch):
    """Same app, but with a real API_KEY configured -- for auth tests."""
    monkeypatch.setattr("src.config.API_KEY", "secret-test-key-123")
    import importlib

    import src.api as api_module
    importlib.reload(api_module)
    return TestClient(api_module.app)


class TestHappyPaths:
    def test_health_needs_no_auth(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_metrics(self, client):
        with patch(
            "src.api.db.health_check",
            return_value={"postgresql": "PG 16.1", "pgvector": "0.7.0", "chunk_count": 6},
        ), patch("src.api.db.get_sources", return_value=[{"source": "rag_basics.md", "chunk_count": 6}]):
            r = client.get("/metrics")
        assert r.status_code == 200
        assert r.json()["total_chunks"] == 6

    def test_ask_happy_path(self, client):
        fake_result = {
            "answer": "RAG combines retrieval and generation.",
            "sources": [{"source": "rag_basics.md", "chunk_index": 0, "distance": 0.05}],
            "route": "cache",
        }
        with patch("src.api.ask", return_value=fake_result) as mock_ask:
            r = client.post("/ask", json={"question": "What is RAG?"})
        assert r.status_code == 200
        assert r.json()["route"] == "cache"
        mock_ask.assert_called_once_with("What is RAG?", provider=None, top_k=None, optimize=True)

    def test_ask_rejects_empty_question(self, client):
        r = client.post("/ask", json={"question": ""})
        assert r.status_code == 422

    def test_agent_happy_path(self, client):
        fake_result = {
            "answer": "| A | B |\n|---|---|",
            "task_type": "multi_step",
            "retrieved": [{"source": "doc.md", "chunk_index": 0, "content": "x", "distance": 0.1}],
            "check_passed": True,
        }
        with patch("src.api.run_agent", return_value=fake_result):
            r = client.post("/agent", json={"task": "Compare X and Y"})
        assert r.status_code == 200
        assert r.json()["task_type"] == "multi_step"
        assert r.json()["check_passed"] is True

    def test_ingest_rejects_unsupported_file_type(self, client):
        r = client.post("/ingest", files={"file": ("data.exe", b"junk", "application/octet-stream")})
        assert r.status_code == 415

    def test_ingest_preserves_original_filename(self, client):
        captured = {}

        def fake_ingest_file(path):
            captured["name"] = path.name
            return 4

        with patch("src.api.ingest_file", side_effect=fake_ingest_file), \
             patch("src.api.db.chunk_count", return_value=10):
            r = client.post("/ingest", files={"file": ("my_notes.md", b"# hello", "text/markdown")})

        assert r.status_code == 200
        assert r.json()["chunks_stored"] == 4
        assert r.json()["total_chunks_in_db"] == 10
        assert captured["name"] == "my_notes.md", "idempotent re-ingest depends on the real filename being kept"


class TestAuth:
    def test_health_still_open_with_api_key_set(self, authed_client):
        assert authed_client.get("/health").status_code == 200

    def test_no_key_header_is_rejected(self, authed_client):
        r = authed_client.post("/ask", json={"question": "What is RAG?"})
        assert r.status_code == 401

    def test_wrong_key_is_rejected(self, authed_client):
        r = authed_client.post(
            "/ask", json={"question": "What is RAG?"}, headers={"X-API-Key": "wrong-key"}
        )
        assert r.status_code == 401

    def test_correct_key_is_accepted(self, authed_client):
        with patch("src.api.ask", return_value={"answer": "ok", "sources": [], "route": "cache"}):
            r = authed_client.post(
                "/ask", json={"question": "What is RAG?"}, headers={"X-API-Key": "secret-test-key-123"}
            )
        assert r.status_code == 200

    def test_metrics_and_ingest_also_require_auth(self, authed_client):
        assert authed_client.get("/metrics").status_code == 401
        r = authed_client.post("/ingest", files={"file": ("x.md", b"hi", "text/markdown")})
        assert r.status_code == 401


class TestErrorHandling:
    def test_ask_generation_failure_returns_clean_502(self, client):
        with patch("src.api.ask", side_effect=RuntimeError("Gemini quota exhausted")):
            r = client.post("/ask", json={"question": "What is RAG?"})
        assert r.status_code == 502
        assert "Gemini quota exhausted" in r.json()["detail"]

    def test_agent_failure_returns_clean_502(self, client):
        with patch("src.api.run_agent", side_effect=RuntimeError("graph failed")):
            r = client.post("/agent", json={"task": "do something"})
        assert r.status_code == 502

    def test_db_down_returns_clean_503(self, client):
        with patch("src.api.db.health_check", side_effect=Exception("connection refused")):
            r = client.get("/metrics")
        assert r.status_code == 503

    def test_malformed_json_body_returns_422(self, client):
        r = client.post("/ask", data="not json", headers={"Content-Type": "application/json"})
        assert r.status_code == 422
