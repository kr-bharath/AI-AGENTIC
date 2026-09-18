"""
Usage:
    python -m src.eval
    python -m src.eval --report path/to/eval_report.md

Runs every question in data/golden_qa.json through the Phase 1 ask() pipeline,
scores each answer with an LLM-as-judge (faithfulness, answer_relevancy,
context_precision), and writes a report. Exits with code 1 if the average
faithfulness score drops below config.EVAL_FAITHFULNESS_THRESHOLD -- this is
the exact check Phase 7 wires into CI as a merge gate.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

from . import config, db
from .ask import ask
from .embeddings import embed_query
from .llm import generate
from .prompts import SYSTEM_JUDGE, build_judge_prompt

GOLDEN_SET_PATH = Path("data/golden_qa.json")
DEFAULT_REPORT_PATH = Path("eval_report.md")


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_json(text: str) -> dict:
    """
    Judges are instructed to return raw JSON, but models sometimes wrap it in
    ```json fences anyway despite instructions -- strip those defensively
    rather than letting one stray fence crash the whole eval run.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text)


def judge_answer(question: str, retrieved: list, answer: str) -> dict:
    prompt = build_judge_prompt(question, retrieved, answer)
    raw = generate(prompt, system=SYSTEM_JUDGE)
    try:
        scores = _extract_json(raw)
    except (json.JSONDecodeError, ValueError):
        # A malformed judge response shouldn't crash the whole eval run --
        # record it as a scored failure instead so it's visible in the report.
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "notes": f"JUDGE PARSE ERROR -- raw response: {raw[:200]}",
        }
    return scores


def run_eval(golden_set_path: Path = GOLDEN_SET_PATH) -> dict:
    config.validate()
    golden_set = load_golden_set(golden_set_path)

    results = []
    for item in golden_set:
        question = item["question"]
        # optimize=False -- eval must score the actual configured provider's
        # output, not a cached answer or a cheaper local-model substitution.
        # Phase 4 added that routing to ask(); scoring it here would make
        # eval results depend on cache state and non-reproducible.
        rag_result = ask(question, optimize=False)

        # ask() only returns source/chunk_index/distance (not chunk text) --
        # the judge needs the actual retrieved content, so re-run retrieval
        # here rather than changing ask()'s return contract. This is a cheap,
        # free, local operation (no extra LLM call), so it costs nothing to
        # duplicate.
        retrieved_for_judge = db.search(embed_query(question))

        scores = judge_answer(question, retrieved_for_judge, rag_result["answer"])

        results.append({
            "id": item["id"],
            "question": question,
            "answer": rag_result["answer"],
            **scores,
        })

    return {
        "results": results,
        "avg_faithfulness": mean(r["faithfulness"] for r in results),
        "avg_answer_relevancy": mean(r["answer_relevancy"] for r in results),
        "avg_context_precision": mean(r["context_precision"] for r in results),
    }


def write_report(summary: dict, path: Path) -> None:
    lines = [
        "# KnowledgeForge AI -- Evaluation Report",
        "",
        f"- Questions evaluated: {len(summary['results'])}",
        f"- Average faithfulness: {summary['avg_faithfulness']:.2f}",
        f"- Average answer relevancy: {summary['avg_answer_relevancy']:.2f}",
        f"- Average context precision: {summary['avg_context_precision']:.2f}",
        f"- Regression threshold (faithfulness): {config.EVAL_FAITHFULNESS_THRESHOLD}",
        "",
        "| ID | Question | Faithfulness | Relevancy | Precision | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["results"]:
        lines.append(
            f"| {r['id']} | {r['question'][:50]} | {r['faithfulness']:.2f} "
            f"| {r['answer_relevancy']:.2f} | {r['context_precision']:.2f} | {r.get('notes', '')} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run the KnowledgeForge AI eval harness")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH,
                         help="Where to write the markdown report")
    parser.add_argument("--golden-set", type=Path, default=GOLDEN_SET_PATH)
    args = parser.parse_args()

    summary = run_eval(args.golden_set)
    write_report(summary, args.report)

    print(f"Evaluated {len(summary['results'])} questions.")
    print(f"  Avg faithfulness:      {summary['avg_faithfulness']:.2f}")
    print(f"  Avg answer relevancy:  {summary['avg_answer_relevancy']:.2f}")
    print(f"  Avg context precision: {summary['avg_context_precision']:.2f}")
    print(f"Report written to {args.report}")

    if summary["avg_faithfulness"] < config.EVAL_FAITHFULNESS_THRESHOLD:
        print(
            f"\nFAIL: avg faithfulness {summary['avg_faithfulness']:.2f} is below "
            f"threshold {config.EVAL_FAITHFULNESS_THRESHOLD}"
        )
        sys.exit(1)

    print("\nPASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
