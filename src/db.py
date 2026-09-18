"""
PostgreSQL + pgvector storage layer for KnowledgeForge AI.

All database operations go through this module.

Responsibilities:
    - Connect to PostgreSQL
    - Store document chunks
    - Store embeddings
    - Delete existing document chunks
    - Perform vector similarity search
    - Count stored chunks
    - Provide database diagnostics
"""

from contextlib import contextmanager

import psycopg2
from pgvector.psycopg2 import register_vector

from . import config
from .tracing import observe

# ============================================================
# DATABASE CONNECTION
# ============================================================

@contextmanager
def get_connection():
    """
    Create and manage a PostgreSQL connection.

    The transaction is committed on success and rolled back
    automatically if an exception occurs.
    """

    conn = psycopg2.connect(config.DATABASE_URL)

    # Register pgvector with psycopg2.
    register_vector(conn)

    try:
        yield conn
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ============================================================
# VECTOR CONVERSION
# ============================================================

def _to_pgvector(values: list[float]) -> str:
    """
    Convert a Python list of floats into pgvector literal format.

    Example:

        [0.1, 0.2, 0.3]

    becomes:

        '[0.1,0.2,0.3]'

    We use an explicit string representation so PostgreSQL
    receives an unambiguous vector value.
    """

    if values is None:
        raise ValueError("Embedding cannot be None")

    if len(values) == 0:
        raise ValueError("Embedding cannot be empty")

    return "[" + ",".join(
        str(float(value))
        for value in values
    ) + "]"


# ============================================================
# INSERT CHUNKS
# ============================================================

def insert_chunks(
    rows: list[tuple[str, int, str, list[float]]]
) -> int:
    """
    Insert document chunks and embeddings.

    Each row must contain:

        (
            source,
            chunk_index,
            content,
            embedding
        )

    Returns:
        Number of inserted rows.
    """

    if not rows:
        return 0

    with get_connection() as conn, conn.cursor() as cur:

        prepared_rows = []

        for source, chunk_index, content, embedding in rows:

            vector = _to_pgvector(embedding)

            prepared_rows.append(
                (
                    source,
                    chunk_index,
                    content,
                    vector,
                )
            )

        cur.executemany(
            """
                INSERT INTO chunks (
                    source,
                    chunk_index,
                    content,
                    embedding
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s::vector
                )
                """,
            prepared_rows,
        )

    return len(rows)


# ============================================================
# DELETE SOURCE
# ============================================================

def delete_source(source: str) -> int:
    """
    Delete all chunks belonging to a document source.

    This allows ingestion to be repeated safely without
    creating duplicate chunks.
    """

    with get_connection() as conn, conn.cursor() as cur:

        cur.execute(
            """
                DELETE FROM chunks
                WHERE source = %s
                """,
            (source,),
        )

        return cur.rowcount


# ============================================================
# VECTOR SEARCH
# ============================================================

@observe(name="vector_search", as_type="retriever")
def search(
    query_embedding: list[float],
    top_k: int | None = None,
):
    """
    Perform cosine-similarity vector search using pgvector.

    pgvector operator:

        <=>

    calculates cosine distance.

    Smaller distance = more similar.

    Returns:

        [
            {
                "source": "rag_basics.md",
                "chunk_index": 0,
                "content": "...",
                "distance": 0.123
            }
        ]
    """

    if query_embedding is None:
        return []

    if len(query_embedding) == 0:
        return []

    if top_k is None:
        top_k = config.TOP_K

    top_k = int(top_k)

    if top_k <= 0:
        return []

    # Convert Python list into an explicit pgvector literal.
    query_vector = _to_pgvector(query_embedding)

    with get_connection() as conn, conn.cursor() as cur:

        cur.execute(
            """
                SELECT
                    source,
                    chunk_index,
                    content,
                    embedding <=> %s::vector AS distance
                FROM chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s
                """,
            (
                query_vector,
                query_vector,
                top_k,
            ),
        )

        rows = cur.fetchall()

    results = []

    for row in rows:

        results.append(
            {
                "source": row[0],
                "chunk_index": row[1],
                "content": row[2],
                "distance": float(row[3]),
            }
        )

    return results


# ============================================================
# COUNT CHUNKS
# ============================================================

def chunk_count() -> int:
    """
    Return total number of chunks in the database.
    """

    with get_connection() as conn, conn.cursor() as cur:

        cur.execute(
            """
                SELECT COUNT(*)
                FROM chunks
                """
        )

        result = cur.fetchone()

    return int(result[0])


# ============================================================
# GET SOURCES
# ============================================================

def get_sources():
    """
    Return all document sources and their chunk counts.
    """

    with get_connection() as conn, conn.cursor() as cur:

        cur.execute(
            """
                SELECT
                    source,
                    COUNT(*) AS chunk_count
                FROM chunks
                GROUP BY source
                ORDER BY source
                """
        )

        rows = cur.fetchall()

    return [
        {
            "source": row[0],
            "chunk_count": int(row[1]),
        }
        for row in rows
    ]


# ============================================================
# DATABASE HEALTH CHECK
# ============================================================

def health_check() -> dict:
    """
    Return basic PostgreSQL and pgvector information.
    """

    with get_connection() as conn, conn.cursor() as cur:

        # PostgreSQL version
        cur.execute(
            "SELECT version()"
        )

        postgres_version = cur.fetchone()[0]

        # pgvector version
        cur.execute(
            """
                SELECT extversion
                FROM pg_extension
                WHERE extname = 'vector'
                """
        )

        vector_result = cur.fetchone()

        # Number of chunks
        cur.execute(
            """
                SELECT COUNT(*)
                FROM chunks
                """
        )

        chunk_total = cur.fetchone()[0]

    return {
        "postgresql": postgres_version,
        "pgvector": (
            vector_result[0]
            if vector_result
            else None
        ),
        "chunk_count": int(chunk_total),
    }


# ============================================================
# DEBUG VECTOR SEARCH
# ============================================================

def debug_search(
    query_embedding: list[float],
    top_k: int | None = None,
):
    """
    Debug helper for vector retrieval.

    Returns the raw search results and useful diagnostics.
    """

    if query_embedding is None:
        raise ValueError("query_embedding is None")

    if not query_embedding:
        raise ValueError("query_embedding is empty")

    query_vector = _to_pgvector(query_embedding)

    if top_k is None:
        top_k = config.TOP_K

    with get_connection() as conn, conn.cursor() as cur:

        # Total documents
        cur.execute(
            """
                SELECT COUNT(*)
                FROM chunks
                """
        )

        total_chunks = cur.fetchone()[0]

        # Stored vector dimensions
        cur.execute(
            """
                SELECT
                    source,
                    chunk_index,
                    vector_dims(embedding)
                FROM chunks
                WHERE embedding IS NOT NULL
                """
        )

        dimensions = cur.fetchall()

        # Actual vector search
        cur.execute(
            """
                SELECT
                    source,
                    chunk_index,
                    embedding <=> %s::vector AS distance
                FROM chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s
                """,
            (
                query_vector,
                query_vector,
                int(top_k),
            ),
        )

        search_results = cur.fetchall()

    return {
        "total_chunks": total_chunks,
        "stored_dimensions": dimensions,
        "query_dimensions": len(query_embedding),
        "search_results": search_results,
    }
