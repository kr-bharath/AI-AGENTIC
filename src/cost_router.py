"""
Cheap, free pre-filter deciding whether a question is simple enough to route
to a local Ollama model instead of a cloud API call. Deliberately NOT an LLM
call itself -- spending an API call to decide whether to avoid an API call
would undercut the entire point. A short word-count + keyword heuristic
captures most of the benefit at zero cost; a small classifier model (or the
same pattern already used in agent.py's classify_task) would be the natural
upgrade if this needed to be more accurate at larger scale.
"""

SIMPLE_MAX_WORDS = 12

COMPLEXITY_KEYWORDS = [
    "compare", "why", "explain in detail", "analyze", "pros and cons",
    "difference between", "step by step", "summarize all", "in depth",
]


def should_use_local(question: str) -> bool:
    lowered = question.lower()
    if any(kw in lowered for kw in COMPLEXITY_KEYWORDS):
        return False
    return len(question.split()) <= SIMPLE_MAX_WORDS
