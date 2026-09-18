from unittest.mock import patch

import numpy as np

import src.cache as cache

from .conftest import make_vec, requires_redis


@requires_redis
class TestSemanticCacheHitsAndMisses:
    def test_miss_on_empty_cache(self):
        with patch("src.cache.embed_query", return_value=make_vec(1)):
            assert cache.get("What is RAG?") is None

    def test_set_then_exact_match_hits(self):
        with patch("src.cache.embed_query", return_value=make_vec(1)):
            cache.set("What is RAG?", "RAG combines retrieval and generation.")
            assert cache.get("What is RAG?") == "RAG combines retrieval and generation."

    def test_near_paraphrase_hits(self):
        vec_a = np.array(make_vec(1))
        paraphrase = vec_a * 0.99 + np.array(make_vec(99)) * 0.01
        paraphrase = (paraphrase / np.linalg.norm(paraphrase)).tolist()

        with patch("src.cache.embed_query", return_value=make_vec(1)):
            cache.set("What is RAG?", "RAG combines retrieval and generation.")
        with patch("src.cache.embed_query", return_value=paraphrase):
            assert cache.get("Explain retrieval-augmented generation") == "RAG combines retrieval and generation."

    def test_unrelated_question_misses(self):
        with patch("src.cache.embed_query", return_value=make_vec(1)):
            cache.set("What is RAG?", "RAG combines retrieval and generation.")
        with patch("src.cache.embed_query", return_value=make_vec(2)):
            assert cache.get("What's the weather today?") is None

    def test_cap_enforcement(self):
        for i in range(10):
            with patch("src.cache.embed_query", return_value=make_vec(100 + i)):
                cache.set(f"question {i}", f"answer {i}")

        import redis

        import src.config as config
        r = redis.Redis.from_url(config.REDIS_URL)
        assert r.llen(cache.INDEX_KEY) == 10


class TestSemanticCacheFailsOpen:
    def test_get_returns_none_when_redis_unreachable(self, monkeypatch):
        monkeypatch.setattr("src.config.REDIS_URL", "redis://localhost:1/0")  # nothing listens here
        cache._client = None
        cache._connection_failed = False
        with patch("src.cache.embed_query", return_value=make_vec(1)):
            assert cache.get("anything") is None

    def test_set_does_not_raise_when_redis_unreachable(self, monkeypatch):
        monkeypatch.setattr("src.config.REDIS_URL", "redis://localhost:1/0")
        cache._client = None
        cache._connection_failed = False
        with patch("src.cache.embed_query", return_value=make_vec(1)):
            cache.set("anything", "some answer")  # must not raise
