"""
Usage:
    python -m src.ask "What is retrieval-augmented generation?"
    python -m src.ask "Summarize the ingested docs" --provider groq
"""
import argparse

from . import config, db
from .embeddings import embed_query
from .llm import generate
from .prompts import SYSTEM_QA, build_qa_prompt


def ask(question: str, provider: str = None, top_k: int = None) -> dict:
    query_embedding = embed_query(question)
    retrieved = db.search(query_embedding, top_k=top_k)

    if not retrieved:
        return {
            "answer": "No documents have been ingested yet — run `python -m src.ingest` first.",
            "sources": [],
        }

    prompt = build_qa_prompt(question, retrieved)
    answer = generate(prompt, system=SYSTEM_QA, provider=provider)

    return {
        "answer": answer,
        "sources": [
            {"source": r["source"], "chunk_index": r["chunk_index"], "distance": round(r["distance"], 4)}
            for r in retrieved
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Ask KnowledgeForge AI a question")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument("--provider", choices=["gemini", "groq"], default=None,
                         help="Override LLM_PROVIDER from .env for this call")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    args = parser.parse_args()

    config.validate()

    result = ask(args.question, provider=args.provider, top_k=args.top_k)

    print("\n--- Answer ---")
    print(result["answer"])

    if result["sources"]:
        print("\n--- Sources (closer to 0 = more relevant) ---")
        for s in result["sources"]:
            print(f"  {s['source']} #{s['chunk_index']}  (distance: {s['distance']})")


if __name__ == "__main__":
    main()
