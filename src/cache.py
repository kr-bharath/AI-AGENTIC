"""
A semantic cache: instead of matching questions verbatim, it matches by
meaning. "What is RAG?" and "Explain retrieval-augmented generation" should
hit the same cache entry even though the text differs -- that's the whole
point versus a plain key-value cache.

Design notes (worth stating explicitly, since they're real tradeoffs):
- Similarity is brute-force cosine over a capped, recency-ordered list of
  cached entries (default 200), not a proper vector index. That's the right
  choice at this scale -- correct, simple, nothing extra to run. At a much
  larger cache size you'd swap this for Redis Stack's native vector search
  (RediSearch HNSW/FLAT index) instead of scaling the brute-force loop.
- Fails OPEN, not closed: if Redis is unreachable, every get() is treated as
  a miss and set() is skipped, with one warning printed -- not an exception.
  Same philosophy as tracing.py: an optional piece of infra being absent
  should never break the core pipeline.
- Embeddings from embeddings.py are already L2-normalized (see
  normalize_embeddings=True), so cosine similarity reduces to a plain dot
  product -- no extra normalization needed here.
"""
import json
import uuid
from typing import Optional

import numpy as np

from . import config
from .embeddings import embed_query

INDEX_KEY = "kf:cache:index"
ENTRY_PREFIX = "kf:cache:entry:"

_client = None
_connection_failed = False


def _get_client():
    """Lazily connect, and remember a failed connection so we don't retry
    (and re-print the warning) on every single call for the rest of the run."""
    global _client, _connection_failed
    if _connection_failed:
        return None
    if _client is None:
        try:
            import redis
            _client = redis.Redis.from_url(config.REDIS_URL, socket_connect_timeout=1)
            _client.ping()
        except Exception as e:
            print(f"  [semantic cache disabled -- couldn't reach Redis at {config.REDIS_URL}: {e}]")
            _connection_failed = True
            _client = None
    return _client


def get(question: str) -> Optional[str]:
    """Return a cached answer if a sufficiently similar question was asked
    before, else None. Never raises -- a cache miss and a cache error look
    the same to the caller."""
    client = _get_client()
    if client is None:
        return None

    try:
        query_vec = np.array(embed_query(question), dtype=np.float32)
        keys = client.lrange(INDEX_KEY, 0, config.SEMANTIC_CACHE_MAX_ENTRIES - 1)

        best_key, best_score, best_answer = None, -1.0, None
        for key in keys:
            entry = client.hgetall(key)
            if not entry:
                continue  # expired since it was indexed -- skip, don't crash
            cached_vec = np.array(json.loads(entry[b"embedding"]), dtype=np.float32)
            score = float(np.dot(query_vec, cached_vec))
            if score > best_score:
                best_key, best_score, best_answer = key, score, entry[b"answer"].decode("utf-8")

        if best_score >= config.SEMANTIC_CACHE_THRESHOLD:
            _touch(client, best_key)
            return best_answer
        return None
    except Exception as e:
        print(f"  [semantic cache read failed, treating as miss: {e}]")
        return None


def set(question: str, answer: str) -> None:
    """Store a question/answer pair. Never raises -- a failed cache write
    should never take down a successful answer."""
    client = _get_client()
    if client is None:
        return

    try:
        query_vec = embed_query(question)
        entry_key = f"{ENTRY_PREFIX}{uuid.uuid4().hex}"

        pipe = client.pipeline()
        pipe.hset(entry_key, mapping={
            "question": question,
            "answer": answer,
            "embedding": json.dumps(query_vec),
        })
        pipe.expire(entry_key, config.SEMANTIC_CACHE_TTL_SECONDS)
        pipe.lpush(INDEX_KEY, entry_key)
        pipe.ltrim(INDEX_KEY, 0, config.SEMANTIC_CACHE_MAX_ENTRIES - 1)
        pipe.execute()
    except Exception as e:
        print(f"  [semantic cache write failed, continuing without caching: {e}]")


def _touch(client, key: bytes) -> None:
    """Move a hit entry to the front of the index -- cheap LRU-ish behavior
    so frequently-asked questions are checked first and survive LTRIM longer."""
    try:
        pipe = client.pipeline()
        pipe.lrem(INDEX_KEY, 0, key)
        pipe.lpush(INDEX_KEY, key)
        pipe.execute()
    except Exception:
        pass  # touching is an optimization, never worth failing over


def clear() -> int:
    """Wipe the cache. Returns number of entries removed. Used by benchmark.py
    to get a clean cold-cache baseline."""
    client = _get_client()
    if client is None:
        return 0
    keys = client.lrange(INDEX_KEY, 0, -1)
    if keys:
        client.delete(*keys)
    client.delete(INDEX_KEY)
    return len(keys)
