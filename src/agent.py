"""
Usage:
    python -m src.agent "Why do teams use RAG instead of fine-tuning?"
    python -m src.agent "Compare chunking and embedding, and give me a table"

The agent reuses everything from Phase 1 (db.search, llm.generate,
embeddings.embed_query, prompts.build_qa_prompt) rather than duplicating it --
this graph only adds routing, multi-source grouping, and a self-check loop
on top.

Graph shape:

    START -> route -> retrieve --[no docs]--> END
                          |
                     [has docs]
                          v
                     synthesize -> self_check --[fail, retry_count < 1]--> synthesize
                                        |
                                   [pass, or retry_count >= 1]
                                        v
                                       END
"""
import argparse
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from . import config, db
from .embeddings import embed_query
from .llm import generate
from .prompts import (
    SYSTEM_COMPARISON,
    SYSTEM_QA,
    SYSTEM_ROUTER,
    SYSTEM_SELF_CHECK,
    ROUTER_TEMPLATE,
    build_comparison_prompt,
    build_qa_prompt,
    build_self_check_prompt,
)

# Cheap first pass before spending an LLM call on classification -- same
# cost-conscious instinct that becomes the full router in Phase 4.
MULTI_STEP_KEYWORDS = [
    "compare", "comparison", " vs ", "versus", "contrast", "difference between",
    "table", "report on", "summarize all", "pros and cons",
]

NO_DOCS_MESSAGE = "No documents have been ingested yet — run `python -m src.ingest` first."

MAX_RETRIES = 1  # bounded self-check retry -- never loop forever


class AgentState(TypedDict, total=False):
    task: str
    task_type: str
    retrieved: List[dict]
    context_by_source: Dict[str, List[dict]]
    answer: str
    check_passed: bool
    check_feedback: str
    retry_count: int


def classify_task(task: str, provider: Optional[str] = None) -> str:
    lowered = task.lower()
    if any(kw in lowered for kw in MULTI_STEP_KEYWORDS):
        return "multi_step"
    verdict = generate(ROUTER_TEMPLATE.format(task=task), system=SYSTEM_ROUTER, provider=provider)
    return "multi_step" if "multi" in verdict.lower() else "simple"


def route_node(state: AgentState) -> dict:
    return {"task_type": classify_task(state["task"])}


def retrieve_node(state: AgentState) -> dict:
    top_k = config.TOP_K * 3 if state["task_type"] == "multi_step" else config.TOP_K
    query_embedding = embed_query(state["task"])
    retrieved = db.search(query_embedding, top_k=top_k)

    if not retrieved:
        return {
            "retrieved": [],
            "context_by_source": {},
            "answer": NO_DOCS_MESSAGE,
        }

    context_by_source: Dict[str, List[dict]] = {}
    for r in retrieved:
        context_by_source.setdefault(r["source"], []).append(r)

    return {"retrieved": retrieved, "context_by_source": context_by_source}


def synthesize_node(state: AgentState) -> dict:
    if state["task_type"] == "multi_step":
        prompt = build_comparison_prompt(state["task"], state["context_by_source"])
        system = SYSTEM_COMPARISON
    else:
        prompt = build_qa_prompt(state["task"], state["retrieved"])
        system = SYSTEM_QA

    feedback = state.get("check_feedback")
    if feedback:
        prompt += f"\n\nYour previous attempt had this problem -- fix it: {feedback}"

    answer = generate(prompt, system=system)
    return {"answer": answer}


def self_check_node(state: AgentState) -> dict:
    # Simple lookups are cheap and low-risk -- skip the extra LLM call to save cost.
    # This mirrors Phase 4's cost-optimization philosophy: only spend an API
    # call where it actually changes the outcome.
    if state["task_type"] != "multi_step":
        return {"check_passed": True}

    verdict = generate(
        build_self_check_prompt(state["task"], state["answer"], state["retrieved"]),
        system=SYSTEM_SELF_CHECK,
    )
    passed = verdict.strip().upper().startswith("PASS")
    return {
        "check_passed": passed,
        "check_feedback": "" if passed else verdict,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def has_docs(state: AgentState) -> str:
    return "yes" if state["retrieved"] else "no"


def should_retry(state: AgentState) -> str:
    if state["check_passed"]:
        return "end"
    if state.get("retry_count", 0) > MAX_RETRIES:
        return "end"
    return "retry"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("self_check", self_check_node)

    graph.add_edge(START, "route")
    graph.add_edge("route", "retrieve")
    graph.add_conditional_edges("retrieve", has_docs, {"yes": "synthesize", "no": END})
    graph.add_edge("synthesize", "self_check")
    graph.add_conditional_edges("self_check", should_retry, {"retry": "synthesize", "end": END})

    return graph.compile()


_compiled_graph = None


def get_graph():
    """Compiled once, reused across calls -- compiling is not free."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(task: str) -> dict:
    config.validate()
    graph = get_graph()
    return graph.invoke({"task": task, "retry_count": 0})


def main():
    parser = argparse.ArgumentParser(description="Run a task through the KnowledgeForge AI agent")
    parser.add_argument("task", help="The question or multi-step task")
    args = parser.parse_args()

    result = run_agent(args.task)

    print(f"\n--- Task type: {result.get('task_type', 'n/a')} ---")
    print("\n--- Answer ---")
    print(result.get("answer", "(no answer produced)"))

    if result.get("retrieved"):
        print("\n--- Sources ---")
        seen = set()
        for r in result["retrieved"]:
            key = (r["source"], r["chunk_index"])
            if key not in seen:
                seen.add(key)
                print(f"  {r['source']} #{r['chunk_index']}")

    if result.get("task_type") == "multi_step":
        status = "passed" if result.get("check_passed") else "failed after retry"
        print(f"\n--- Self-check: {status} ---")


if __name__ == "__main__":
    main()
