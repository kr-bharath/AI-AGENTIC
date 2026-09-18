"""
Central place every other module pulls settings from.
Keeping this separate means Phase 4 (cost routing), Phase 5 (FastAPI), etc.
can all import the same config without re-reading .env everywhere.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GEMINI_MODEL = "gemini-flash-latest"  # alias -- auto-tracks Google's current stable Flash model
GROQ_MODEL = "openai/gpt-oss-120b"

# --- Database ---
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://kf_user:kf_password@localhost:5432/knowledgeforge"
)

# --- Embeddings ---
# Local, free, no API key. 384-dim output — must match VECTOR(384) in sql/init.sql.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Chunking / retrieval ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "4"))

# --- Observability (Phase 3) ---
# Optional: leave blank and tracing silently no-ops (confirmed safe -- @observe
# just logs a warning and the app runs normally without a Langfuse account).
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# --- Evaluation (Phase 3) ---
# Regression gate: eval.py exits non-zero if the average faithfulness score
# across the golden set drops below this. Becomes the CI/CD quality gate in Phase 7.
EVAL_FAITHFULNESS_THRESHOLD = float(os.getenv("EVAL_FAITHFULNESS_THRESHOLD", "0.7"))

# --- Cost & latency optimization (Phase 4) ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.93"))
SEMANTIC_CACHE_TTL_SECONDS = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "86400"))
SEMANTIC_CACHE_MAX_ENTRIES = int(os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "200"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Rough per-call cost estimates for the benchmark report, in USD. These are
# deliberately approximate (free-tier APIs don't bill you directly) -- they
# exist to make the "$ saved by routing/caching" number concrete rather than
# to be an exact bill. Based on published per-token pricing for a typical
# short RAG prompt+answer (~800 input / ~200 output tokens) if you were on
# a paid tier; local Ollama and cache hits are genuinely $0.
EST_COST_PER_CLOUD_CALL_USD = float(os.getenv("EST_COST_PER_CLOUD_CALL_USD", "0.0006"))

# --- API (Phase 5) ---
# Blank disables auth entirely (fine for local dev) -- same "blank = off"
# pattern as Langfuse. Set a real value before this is reachable off your machine.
API_KEY = os.getenv("API_KEY", "")


def validate():
    """Fail fast with a clear message instead of a cryptic error mid-pipeline."""
    problems = []
    if LLM_PROVIDER not in ("gemini", "groq"):
        problems.append(f"LLM_PROVIDER must be 'gemini' or 'groq', got '{LLM_PROVIDER}'")
    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        problems.append("GEMINI_API_KEY is empty but LLM_PROVIDER=gemini — set it in .env")
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        problems.append("GROQ_API_KEY is empty but LLM_PROVIDER=groq — set it in .env")
    if problems:
        raise RuntimeError("Config problem(s):\n  - " + "\n  - ".join(problems))
