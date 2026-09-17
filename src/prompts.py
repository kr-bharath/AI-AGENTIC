"""
Deliberately separate templates rather than one generic prompt -- this is the
"prompt engineering" skill made visible and demoable, not just implicit in
one f-string buried in ask.py.
"""

SYSTEM_QA = (
    "You are a precise research assistant. Answer ONLY using the provided "
    "context. If the context does not contain the answer, say so plainly -- "
    "never invent information. Keep answers concise and cite which source "
    "each fact came from using the [source] markers already in the context."
)

QA_TEMPLATE = """Context:
{context}

Question: {question}

Answer the question using only the context above. Cite sources by their [source] tag."""


SYSTEM_SUMMARY = (
    "You are a technical summarizer. Produce summaries that a busy engineer "
    "could skim in 20 seconds and still get the key points."
)

SUMMARY_TEMPLATE = """Summarize the following context in 3-5 bullet points.
Focus on concrete facts and figures, not generic statements.

Context:
{context}"""


SYSTEM_EXTRACTION = (
    "You extract structured data. Respond with valid JSON only -- no prose, "
    "no markdown code fences, no explanation."
)

EXTRACTION_TEMPLATE = """From the context below, extract the following fields as JSON:
{fields}

Context:
{context}

Return ONLY a JSON object."""


def build_qa_prompt(question: str, retrieved_chunks: list) -> str:
    """
    retrieved_chunks: list of dicts with 'source', 'chunk_index', 'content'
    (the shape returned by db.search)
    """
    context_blocks = [
        f"[source: {c['source']} #{c['chunk_index']}]\n{c['content']}"
        for c in retrieved_chunks
    ]
    context = "\n\n".join(context_blocks)
    return QA_TEMPLATE.format(context=context, question=question)


# ---------------------------------------------------------------------------
# Phase 2 additions -- agent layer: routing, multi-source comparison, self-check
# ---------------------------------------------------------------------------

SYSTEM_ROUTER = (
    "You classify user requests for a document Q&A system. Respond with "
    "exactly one word: SIMPLE if the request is a direct factual question "
    "answerable from one pass of retrieval, or MULTI if it requires "
    "comparing multiple sources, producing a structured report or table, "
    "or several reasoning steps."
)

ROUTER_TEMPLATE = "Request: {task}\n\nRespond with exactly one word: SIMPLE or MULTI."


SYSTEM_COMPARISON = (
    "You are a research analyst. Using ONLY the grouped context provided "
    "(grouped by source document), complete the task. Use a markdown table "
    "where it helps. If a source doesn't cover a point, say so explicitly "
    "instead of guessing. Cite sources by their [source] tag."
)

COMPARISON_TEMPLATE = """Task: {task}

Context, grouped by source:
{context}

Complete the task using only the context above."""


def build_comparison_prompt(task: str, context_by_source: dict) -> str:
    """
    context_by_source: {source_name: [chunk_dict, ...]}
    (the shape agent.py builds from db.search results)
    """
    blocks = []
    for source, chunks in context_by_source.items():
        joined = "\n".join(f"  {c['content']}" for c in chunks)
        blocks.append(f"[source: {source}]\n{joined}")
    context = "\n\n".join(blocks)
    return COMPARISON_TEMPLATE.format(context=context, task=task)


SYSTEM_SELF_CHECK = (
    "You are a strict reviewer. Check whether the ANSWER is fully grounded "
    "in the CONTEXT and actually completes the TASK. Respond with 'PASS' if "
    "it's good, or 'FAIL: <one sentence reason>' if something is missing, "
    "ungrounded, or the task wasn't actually completed (e.g. no table when "
    "one was requested)."
)

SELF_CHECK_TEMPLATE = """Task: {task}

Context:
{context}

Answer to review:
{answer}

Respond with PASS or FAIL: <reason>."""


def build_self_check_prompt(task: str, answer: str, retrieved_chunks: list) -> str:
    context = "\n\n".join(
        f"[{c['source']} #{c['chunk_index']}] {c['content']}" for c in retrieved_chunks
    )
    return SELF_CHECK_TEMPLATE.format(task=task, context=context, answer=answer)


# ---------------------------------------------------------------------------
# Phase 3 additions -- LLM-as-judge evaluation (faithfulness, relevancy, precision)
# ---------------------------------------------------------------------------
# Deliberately NOT using the RAGAS library here: as of this build, ragas==0.4.3
# pulls in langchain_community/langchain_openai versions that hard-conflict with
# langgraph==1.2.11 (which needs langchain-core>=1.4) -- installing both breaks
# the environment. This harness reimplements the same three metrics as direct
# LLM-as-judge calls against the provider you already have configured, with no
# extra dependency and no conflict risk.

SYSTEM_JUDGE = (
    "You are a strict, impartial evaluator of a RAG (retrieval-augmented "
    "generation) system. You will be given a QUESTION, the CONTEXT that was "
    "retrieved for it, and the ANSWER the system produced. Score three things "
    "independently, each from 0.0 to 1.0:\n"
    "- faithfulness: does every claim in the ANSWER actually appear in or "
    "follow from the CONTEXT? A correct refusal (saying the context doesn't "
    "cover it, when that's true) is fully faithful (1.0) -- faithfulness is "
    "about not fabricating unsupported claims, not about whether a question "
    "got answered.\n"
    "- answer_relevancy: does the ANSWER actually address what the QUESTION "
    "asked (regardless of whether it drew on the context correctly)?\n"
    "- context_precision: was the retrieved CONTEXT actually relevant to "
    "answering the QUESTION, or mostly irrelevant material?\n"
    "Respond with ONLY a JSON object, no markdown fences, no prose: "
    '{"faithfulness": <float>, "answer_relevancy": <float>, '
    '"context_precision": <float>, "notes": "<one short sentence>"}'
)

JUDGE_TEMPLATE = """QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Return the JSON scoring object now."""


def build_judge_prompt(question: str, retrieved_chunks: list, answer: str) -> str:
    if retrieved_chunks:
        context = "\n\n".join(
            f"[{c['source']} #{c['chunk_index']}] {c['content']}" for c in retrieved_chunks
        )
    else:
        context = "(no context was retrieved)"
    return JUDGE_TEMPLATE.format(question=question, context=context, answer=answer)
