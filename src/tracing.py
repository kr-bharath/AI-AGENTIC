"""
llm.py and db.py both import `observe` from here instead of directly from
`langfuse`. The reason: langfuse's own @observe, when the client has no
valid keys, does NOT silently skip the network call -- it still tries to
export every span to the configured host and fails per-call (401/403),
which is noisy and adds a wasted round-trip to every single LLM/DB call.

This wrapper checks config up front and substitutes a true no-op decorator
when Langfuse isn't configured, so "leave it blank" actually means zero
network activity, not "attempt and fail silently."
"""
from . import config

_TRACING_ENABLED = bool(config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY)

if _TRACING_ENABLED:
    from langfuse import observe  # real tracing -- keys are present
else:
    def observe(*dargs, **dkwargs):
        """Drop-in no-op replacement for langfuse.observe when untracked."""
        def decorator(func):
            return func
        # Support both @observe and @observe(name=..., as_type=...) forms
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        return decorator
