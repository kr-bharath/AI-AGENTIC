"""
Usage:
    python -m src.ask "What is retrieval-augmented generation?"
    python -m src.ask "Summarize the ingested docs" --provider groq
    python -m src.ask "What is RAG?" --no-optimize   # force the Phase 1-3 path exactly
"""
import argparse

from . import cache, config, db
from .cost_router import should_use_local
from .embeddings import embed_query
from .llm import generate
from .prompts import SYSTEM_QA, build_qa_prompt


def ask(question: str, provider: str | None = None, top_k: int | None = None, optimize: bool = True) -> dict:
    """
    provider: explicit override ("gemini"/"groq"/"local"). When set, this
    bypasses caching and local-routing entirely and behaves exactly like
    Phase 1-3 -- an explicit choice always wins over automatic optimization.
    optimize: set False to force the plain cloud path even with provider=None
    (used by eval.py and benchmark.py to get an uncached, unrouted baseline).
    """
    query_embedding = embed_query(question)
    retrieved = db.search(query_embedding, top_k=top_k)

    if not retrieved:
        return {
            "answer": "No documents have been ingested yet — run `python -m src.ingest` first.",
            "sources": [],
            "route": "n/a",
        }

    prompt = build_qa_prompt(question, retrieved)

    if provider is not None or not optimize:
        # Explicit override, or optimization deliberately disabled -- exact
        # Phase 1-3 behavior, no cache, no local routing.
        answer = generate(prompt, system=SYSTEM_QA, provider=provider)
        route = f"cloud (explicit: {provider})" if provider else "cloud (optimize=False)"
    else:
        cached_answer = cache.get(question)
        if cached_answer is not None:
            answer, route = cached_answer, "cache"
        elif should_use_local(question):
            try:
                answer = generate(prompt, system=SYSTEM_QA, provider="local")
                route = "local"
            except Exception as e:
                print(f"  [local model unavailable ({e}) -- falling back to cloud]")
                answer = generate(prompt, system=SYSTEM_QA)
                route = "cloud (local fallback)"
        else:
            answer = generate(prompt, system=SYSTEM_QA)
            route = "cloud"

        cache.set(question, answer)

    return {
        "answer": answer,
        "sources": [
            {"source": r["source"], "chunk_index": r["chunk_index"], "distance": round(r["distance"], 4)}
            for r in retrieved
        ],
        "route": route,
    }


def main():
    parser = argparse.ArgumentParser(description="Ask KnowledgeForge AI a question")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument("--provider", choices=["gemini", "groq", "local"], default=None,
                         help="Override LLM_PROVIDER from .env for this call (bypasses cache/routing)")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    parser.add_argument("--no-optimize", action="store_true",
                         help="Disable semantic cache and local routing for this call")
    args = parser.parse_args()

    config.validate()

    result = ask(args.question, provider=args.provider, top_k=args.top_k, optimize=not args.no_optimize)

    print(f"\n--- Route: {result['route']} ---")
    print("\n--- Answer ---")
    print(result["answer"])

    if result["sources"]:
        print("\n--- Sources (closer to 0 = more relevant) ---")
        for s in result["sources"]:
            print(f"  {s['source']} #{s['chunk_index']}  (distance: {s['distance']})")


if __name__ == "__main__":
    main()
