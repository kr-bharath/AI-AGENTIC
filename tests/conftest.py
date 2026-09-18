"""
Shared fixtures. Sets fake-but-valid env vars before any src.* module
imports config at module load time, and provides a redis_available marker
so cache-dependent tests skip cleanly on a machine without Redis running,
rather than failing with a confusing connection error.
"""
import os

os.environ.setdefault("GEMINI_API_KEY", "test-fake-key")
os.environ.setdefault("GROQ_API_KEY", "test-fake-key")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

import pytest


def _redis_reachable() -> bool:
    try:
        import redis
        client = redis.Redis.from_url(os.environ["REDIS_URL"], socket_connect_timeout=1)
        client.ping()
        return True
    except Exception:
        return False


REDIS_AVAILABLE = _redis_reachable()

requires_redis = pytest.mark.skipif(
    not REDIS_AVAILABLE, reason="No Redis reachable at REDIS_URL -- start one to run this test"
)


@pytest.fixture(autouse=True)
def _flush_redis_between_tests():
    """Cache tests must not see leftover state from a previous test."""
    if REDIS_AVAILABLE:
        import redis
        redis.Redis.from_url(os.environ["REDIS_URL"]).flushall()
    yield


def make_vec(seed: int, dim: int = 384):
    """Deterministic, normalized fake embedding -- matches the real shape
    (sentence-transformers embeddings are L2-normalized) without needing the
    actual model downloaded."""
    import numpy as np
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()
