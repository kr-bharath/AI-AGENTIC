"""
All Postgres + pgvector access goes through this module. Nothing else in the
codebase should import psycopg2 directly — keeps the storage layer swappable
(e.g. to Supabase-hosted Postgres, or a different vector store later) without
touching ingest.py / retrieval callers.
"""
from contextlib import contextmanager
from typing import List, Tuple

import psycopg2
from pgvector.psycopg2 import register_vector

from . import config


@contextmanager
def get_connection():
    conn = psycopg2.connect(config.DATABASE_URL)
    register_vector(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_chunks(rows: List[Tuple[str, int, str, List[float]]]) -> int:
    """
    rows: list of (source, chunk_index, content, embedding)
    Returns the number of rows inserted.
    """
    if not rows:
        return 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (source, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s::vector)
                """,
                rows,
            )
    return len(rows)


def delete_source(source: str) -> int:
    """Remove all chunks for a given source file — lets ingest.py be re-run safely."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source = %s", (source,))
            return cur.rowcount


def search(query_embedding: List[float], top_k: int = None):
    """
    Cosine-similarity nearest-neighbor search via pgvector's <=> operator
    (cosine distance — smaller is more similar).
    Returns list of dicts: source, chunk_index, content, distance.
    """
    top_k = top_k or config.TOP_K

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source, chunk_index, content, embedding <=> %s::vector AS distance
                FROM chunks
                ORDER BY distance ASC
                LIMIT %s
                """,
                (query_embedding, top_k),
            )
            rows = cur.fetchall()

    return [
        {"source": r[0], "chunk_index": r[1], "content": r[2], "distance": float(r[3])}
        for r in rows
    ]


def chunk_count() -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            return cur.fetchone()[0]
