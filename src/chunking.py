"""
Simple word-based sliding-window chunker.
Good enough for Phase 1 — swap for a token-aware or semantic chunker later
if you want to demonstrate that refinement in interviews.
"""
from typing import List

from . import config


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Split `text` into overlapping chunks, measured in words.

    Overlap keeps context from being severed exactly at a chunk boundary —
    e.g. a sentence that straddles two chunks still appears whole in one of them.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start += step

    return chunks
