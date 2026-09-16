"""
Embeddings run 100% locally via sentence-transformers — no API key, no per-call
cost, no rate limit. This is deliberate: it keeps the most frequent operation
in the whole pipeline (embedding every chunk and every query) completely free.
"""
from functools import lru_cache
from typing import List

import numpy as np

from . import config


@lru_cache(maxsize=1)
def _get_model():
    # Imported lazily so `python -m src.ask --help` etc. doesn't pay the
    # (one-time, few-second) model load cost unless embedding is actually needed.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of strings. Returns one vector per input string."""
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors).tolist()


def embed_query(text: str) -> List[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
