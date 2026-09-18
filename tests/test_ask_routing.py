from unittest.mock import patch

import src.ask
import src.db

from .conftest import make_vec, requires_redis

FAKE_CHUNKS = [
    {"source": "doc.md", "chunk_index": 0, "content": "RAG grounds answers in retrieved context.", "distance": 0.05}
]

QUESTION_VECS = {
    "What is RAG?": make_vec(1),
    "Compare chunking and embedding strategies in depth": make_vec(2),
    "What is chunking?": make_vec(3),
}


def _fake_embed(question):
    return QUESTION_VECS[question]


@requires_redis
class TestAskRouting:
    def test_short_question_routes_local(self):
        calls = []

        def fake_generate(prompt, system=None, provider=None):
            calls.append(provider or "default")
            return "Local model answer." if provider == "local" else "Cloud model answer."

        with patch("src.ask.generate", side_effect=fake_generate), \
             patch("src.ask.embed_query", side_effect=_fake_embed), \
             patch("src.cache.embed_query", side_effect=_fake_embed), \
             patch("src.db.search", return_value=FAKE_CHUNKS):
            result = src.ask.ask("What is RAG?")

        assert result["route"] == "local"
        assert calls == ["local"]

    def test_repeat_question_hits_cache_with_zero_llm_calls(self):
        calls = []

        def fake_generate(prompt, system=None, provider=None):
            calls.append(provider or "default")
            return "Local model answer." if provider == "local" else "Cloud model answer."

        with patch("src.ask.generate", side_effect=fake_generate), \
             patch("src.ask.embed_query", side_effect=_fake_embed), \
             patch("src.cache.embed_query", side_effect=_fake_embed), \
             patch("src.db.search", return_value=FAKE_CHUNKS):
            src.ask.ask("What is RAG?")
            calls.clear()
            result = src.ask.ask("What is RAG?")

        assert result["route"] == "cache"
        assert calls == [], "a cache hit must make zero LLM calls"

    def test_complex_question_routes_cloud(self):
        calls = []

        def fake_generate(prompt, system=None, provider=None):
            calls.append(provider or "default")
            return "Cloud model answer."

        with patch("src.ask.generate", side_effect=fake_generate), \
             patch("src.ask.embed_query", side_effect=_fake_embed), \
             patch("src.cache.embed_query", side_effect=_fake_embed), \
             patch("src.db.search", return_value=FAKE_CHUNKS):
            result = src.ask.ask("Compare chunking and embedding strategies in depth")

        assert result["route"] == "cloud"
        assert calls == ["default"]

    def test_explicit_provider_bypasses_cache_and_routing(self):
        calls = []

        def fake_generate(prompt, system=None, provider=None):
            calls.append(provider or "default")
            return "answer"

        with patch("src.ask.generate", side_effect=fake_generate), \
             patch("src.ask.embed_query", side_effect=_fake_embed), \
             patch("src.cache.embed_query", side_effect=_fake_embed), \
             patch("src.db.search", return_value=FAKE_CHUNKS):
            src.ask.ask("What is RAG?")  # populate cache
            calls.clear()
            result = src.ask.ask("What is RAG?", provider="groq")

        assert result["route"] == "cloud (explicit: groq)"
        assert calls == ["groq"], "explicit override must skip cache and local routing entirely"

    def test_optimize_false_forces_plain_cloud_path(self):
        calls = []

        def fake_generate(prompt, system=None, provider=None):
            calls.append(provider or "default")
            return "answer"

        with patch("src.ask.generate", side_effect=fake_generate), \
             patch("src.ask.embed_query", side_effect=_fake_embed), \
             patch("src.cache.embed_query", side_effect=_fake_embed), \
             patch("src.db.search", return_value=FAKE_CHUNKS):
            result = src.ask.ask("What is RAG?", optimize=False)

        assert result["route"] == "cloud (optimize=False)"
        assert calls == ["default"]

    def test_local_model_down_falls_back_to_cloud_without_crashing(self):
        calls = []

        def fake_generate(prompt, system=None, provider=None):
            calls.append(provider or "default")
            if provider == "local":
                raise ConnectionError("Ollama not running")
            return "Cloud fallback answer."

        with patch("src.ask.generate", side_effect=fake_generate), \
             patch("src.ask.embed_query", side_effect=_fake_embed), \
             patch("src.cache.embed_query", side_effect=_fake_embed), \
             patch("src.db.search", return_value=FAKE_CHUNKS):
            result = src.ask.ask("What is chunking?")

        assert result["route"] == "cloud (local fallback)"
        assert calls == ["local", "default"]
