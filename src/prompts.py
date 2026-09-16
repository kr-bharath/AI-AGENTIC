"""
Deliberately separate templates rather than one generic prompt — this is the
"prompt engineering" skill made visible and demoable, not just implicit in
one f-string buried in ask.py.
"""

SYSTEM_QA = (
    "You are a precise research assistant. Answer ONLY using the provided "
    "context. If the context does not contain the answer, say so plainly — "
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
    "You extract structured data. Respond with valid JSON only — no prose, "
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
