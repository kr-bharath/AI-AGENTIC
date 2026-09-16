"""
One function, two free providers. This thin abstraction is what makes Phase 4's
cost/latency router trivial to add later — it'll just call generate(..., provider=X)
with a third option ("local") pointed at Ollama, no changes needed elsewhere.
"""
from . import config


def _generate_gemini(prompt: str, system: str = None) -> str:
    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        config.GEMINI_MODEL,
        system_instruction=system,
    )
    response = model.generate_content(prompt)
    return response.text


def _generate_groq(prompt: str, system: str = None) -> str:
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content


def generate(prompt: str, system: str = None, provider: str = None) -> str:
    """
    provider: "gemini" | "groq" | None (falls back to config.LLM_PROVIDER)
    """
    provider = (provider or config.LLM_PROVIDER).lower()

    if provider == "gemini":
        return _generate_gemini(prompt, system)
    elif provider == "groq":
        return _generate_groq(prompt, system)
    else:
        raise ValueError(f"Unknown provider '{provider}' — use 'gemini' or 'groq'")
