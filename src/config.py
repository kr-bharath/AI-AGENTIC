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

GEMINI_MODEL = "gemini-2.0-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- Database ---
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://kf_user:kf_password@localhost:5432/knowledgeforge"
)

# --- Embeddings ---
# Local, free, no API key. 384-dim output — must match VECTOR(384) in sql/init.sql.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Chunking / retrieval ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K", 4))


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
