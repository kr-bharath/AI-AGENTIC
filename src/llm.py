"""
One function, two free providers. This thin abstraction is what makes Phase 4's
cost/latency router trivial to add later -- it'll just call generate(..., provider=X)
with a third option ("local") pointed at Ollama, no changes needed elsewhere.
"""
import time

from . import config
from .tracing import observe


def _generate_gemini(prompt: str, system: str = None) -> str:
    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        config.GEMINI_MODEL,
        system_instruction=system,
    )
    # retry=None explicitly overrides the SDK's own baked-in retry-with-sleep
    # (confirmed via the gapic client signature: the default is a distinct
    # _MethodDefault sentinel, so passing None here really does disable it,
    # rather than falling back to the default). Without this, a 429 gets
    # retried twice -- once silently inside the SDK, then again by our own
    # _with_retry -- which stacks into a long, silent-looking wait.
    response = model.generate_content(prompt, request_options={"retry": None})
    return response.text


def _generate_groq(prompt: str, system: str = None) -> str:
    from groq import Groq

    # max_retries=0: same reasoning as Gemini's retry=None above -- the groq
    # client retries 429s internally by default (confirmed: default is 2),
    # which happens BEFORE our own _with_retry ever sees the exception. That
    # produced exactly the "looks frozen" experience -- disable it here so
    # _with_retry has sole, visible control over backoff.
    client = Groq(api_key=config.GROQ_API_KEY, max_retries=0)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content


def _is_rate_limit_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "429" in msg or "resourceexhausted" in msg or "rate limit" in msg or "quota" in msg


def _is_daily_quota_error(err: Exception) -> bool:
    """
    A per-minute rate limit is worth a short backoff-and-retry. A per-day
    quota is not -- no amount of waiting inside one process fixes that, so
    retrying just wastes time and still fails. Distinguish the two so we
    fail fast with a useful message instead.
    """
    msg = str(err).lower().replace(" ", "")
    return "perday" in msg or "requestsperday" in msg


def _with_retry(fn, max_retries: int = 3, base_delay: float = 10.0):
    """
    Free-tier APIs rate-limit aggressively (Gemini's free tier can be as low
    as 20 requests/DAY on some models). Rather than crash the whole agent
    run on a transient 429, back off and retry a bounded number of times --
    this is the same instinct Phase 4's cost/latency router builds out
    fully, just applied at the smallest scale that keeps development from
    being painful right now.
    """
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if _is_daily_quota_error(e):
                raise RuntimeError(
                    "Daily free-tier request quota is exhausted for this provider/model. "
                    "Waiting won't help -- it resets on the provider's schedule (often ~24h). "
                    "Switch LLM_PROVIDER in .env to another configured provider to keep working today."
                ) from e
            if not _is_rate_limit_error(e) or attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  [rate limited -- retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})]")
            time.sleep(delay)
    raise last_err  # pragma: no cover -- loop always returns or raises above


@observe(name="llm_generate", as_type="generation")
def generate(prompt: str, system: str = None, provider: str = None) -> str:
    """
    provider: "gemini" | "groq" | None (falls back to config.LLM_PROVIDER)

    Traced via Langfuse's @observe -- every call from ask.py and agent.py
    passes through here, so both get tracing for free with no changes to
    either file. If Langfuse isn't configured (no keys in .env), this
    decorator silently no-ops -- confirmed safe, not just assumed.
    """
    provider = (provider or config.LLM_PROVIDER).lower()

    if provider == "gemini":
        return _with_retry(lambda: _generate_gemini(prompt, system))
    elif provider == "groq":
        return _with_retry(lambda: _generate_groq(prompt, system))
    else:
        raise ValueError(f"Unknown provider '{provider}' -- use 'gemini' or 'groq'")
