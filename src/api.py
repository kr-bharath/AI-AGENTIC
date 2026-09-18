"""
Usage:
    uvicorn src.api:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive Swagger docs.

This is a thin HTTP layer over everything already built -- it doesn't
reimplement ingestion, retrieval, generation, or the agent graph, it just
calls the same functions ask.py, agent.py, and ingest.py already use. That's
deliberate: the pipeline logic gets tested exactly once, here it's just
exposed.
"""
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from . import config, db
from .agent import run_agent
from .ask import ask
from .ingest import SUPPORTED_EXTENSIONS, ingest_file
from .schemas import (
    AgentRequest,
    AgentResponse,
    AskRequest,
    AskResponse,
    IngestResponse,
    MetricsResponse,
    SourceInfo,
)

app = FastAPI(
    title="KnowledgeForge AI",
    description="RAG + agentic knowledge assistant -- API layer over Phases 1-4.",
    version="0.5.0",
)


async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """
    Blank config.API_KEY means auth is off (local dev default). Once set,
    every protected endpoint requires a matching X-API-Key header.
    """
    if not config.API_KEY:
        return
    if x_api_key != config.API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")


@app.get("/health", tags=["ops"])
def health():
    """Unauthenticated -- for uptime checks / Docker healthcheck (Phase 6)."""
    return {"status": "ok"}


@app.get("/metrics", response_model=MetricsResponse, dependencies=[Depends(verify_api_key)], tags=["ops"])
def metrics():
    try:
        health_info = db.health_check()
        sources = db.get_sources()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")

    return MetricsResponse(
        postgresql_version=health_info["postgresql"],
        pgvector_version=health_info["pgvector"],
        total_chunks=health_info["chunk_count"],
        sources=[SourceInfo(**s) for s in sources],
    )


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key)], tags=["rag"])
def ask_endpoint(body: AskRequest):
    try:
        result = ask(
            body.question,
            provider=body.provider,
            top_k=body.top_k,
            optimize=body.optimize,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Generation failed: {e}")

    return AskResponse(**result)


@app.post("/agent", response_model=AgentResponse, dependencies=[Depends(verify_api_key)], tags=["rag"])
def agent_endpoint(body: AgentRequest):
    try:
        result = run_agent(body.task)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent run failed: {e}")

    return AgentResponse(
        answer=result.get("answer", ""),
        task_type=result.get("task_type", "n/a"),
        sources=result.get("retrieved", []),
        check_passed=result.get("check_passed"),
    )


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_key)], tags=["rag"])
async def ingest_endpoint(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}' -- allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    # Write to a temp dir but keep the ORIGINAL filename -- ingest_file() uses
    # path.name as the source identifier for idempotent re-ingestion (it
    # deletes old chunks for that filename before inserting new ones), so a
    # random temp name would break that and orphan chunks under a UUID.
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / file.filename
        contents = await file.read()
        dest.write_bytes(contents)

        try:
            chunks_stored = ingest_file(dest)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Ingestion failed: {e}")

    return IngestResponse(
        filename=file.filename,
        chunks_stored=chunks_stored,
        total_chunks_in_db=db.chunk_count(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    # Last-resort safety net -- a bug in the pipeline should return a clean
    # 500 with a message, not an unhandled traceback leaking to the client.
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc}"})
