"""
Request/response schemas, kept separate from api.py so they're easy to
scan on their own -- this is what a consumer of the API actually reads.
"""

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    source: str
    chunk_index: int
    distance: float


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is RAG?"])
    provider: str | None = Field(
        None, description="Explicit override: 'gemini', 'groq', or 'local'. Bypasses cache/routing."
    )
    top_k: int | None = Field(None, ge=1, le=20)
    optimize: bool = Field(
        True, description="Set false to force the plain cloud path (no cache/local routing)."
    )


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    route: str


class AgentRequest(BaseModel):
    task: str = Field(..., min_length=1, examples=["Compare chunking and embedding in a table"])


class AgentResponse(BaseModel):
    answer: str
    task_type: str
    sources: list[dict]
    check_passed: bool | None = None


class IngestResponse(BaseModel):
    filename: str
    chunks_stored: int
    total_chunks_in_db: int


class SourceInfo(BaseModel):
    source: str
    chunk_count: int


class MetricsResponse(BaseModel):
    postgresql_version: str
    pgvector_version: str | None
    total_chunks: int
    sources: list[SourceInfo]
