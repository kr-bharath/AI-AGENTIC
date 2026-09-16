"""
Usage:
    python -m src.ingest data/sample_docs
    python -m src.ingest path/to/a/single/file.pdf

Loads .txt, .md, and .pdf files, chunks them, embeds every chunk locally,
and stores them in Postgres/pgvector. Re-running on the same file replaces
its old chunks (delete-then-insert) so ingestion is idempotent.
"""
import argparse
import sys
from pathlib import Path

from . import config, db
from .chunking import chunk_text
from .embeddings import embed_texts

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return path.read_text(encoding="utf-8", errors="ignore")


def discover_files(target: Path):
    if target.is_file():
        return [target] if target.suffix.lower() in SUPPORTED_EXTENSIONS else []
    return sorted(
        p for p in target.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def ingest_file(path: Path) -> int:
    text = read_file(path)
    if not text.strip():
        print(f"  ! {path.name}: no extractable text, skipping")
        return 0

    chunks = chunk_text(text)
    if not chunks:
        print(f"  ! {path.name}: produced 0 chunks, skipping")
        return 0

    embeddings = embed_texts(chunks)

    source_name = path.name
    db.delete_source(source_name)  # idempotent re-ingest
    rows = [
        (source_name, i, chunk, emb) for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    inserted = db.insert_chunks(rows)
    print(f"  ok {path.name}: {inserted} chunks")
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into KnowledgeForge AI")
    parser.add_argument(
        "path",
        nargs="?",
        default="data/sample_docs",
        help="File or directory to ingest (default: data/sample_docs)",
    )
    args = parser.parse_args()

    config.validate()

    target = Path(args.path)
    if not target.exists():
        print(f"Error: path does not exist: {target}", file=sys.stderr)
        sys.exit(1)

    files = discover_files(target)
    if not files:
        print(f"No supported files (.txt, .md, .pdf) found under {target}")
        sys.exit(0)

    print(f"Ingesting {len(files)} file(s)...")
    total = 0
    for f in files:
        total += ingest_file(f)

    print(f"\nDone. {total} chunks stored. Total chunks in DB: {db.chunk_count()}")


if __name__ == "__main__":
    main()
