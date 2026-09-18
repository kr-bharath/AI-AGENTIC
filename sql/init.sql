-- Runs automatically on first container startup (mounted into /docker-entrypoint-initdb.d/)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           SERIAL PRIMARY KEY,
    source       TEXT NOT NULL,        -- filename the chunk came from
    chunk_index  INT NOT NULL,         -- position of this chunk within the source doc
    content      TEXT NOT NULL,        -- the chunk text itself
    embedding    VECTOR(384),          -- all-MiniLM-L6-v2 produces 384-dim embeddings
    created_at   TIMESTAMP DEFAULT now()
);

-- Approximate nearest-neighbor index for fast cosine-similarity search.
-- ivfflat needs data in the table before it's useful; fine to create up front for a small corpus.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (source);
