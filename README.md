# KnowledgeForge AI — Phase 1: Core RAG Pipeline

A grounded question-answering CLI: ingest documents, ask questions, get answers
cited to their source chunks. Built entirely on free tools — local embeddings,
free-tier LLM APIs, self-hosted Postgres+pgvector.

## What's in Phase 1

- `docker-compose.yml` — Postgres with the pgvector extension, auto-initialized
- `src/ingest.py` — loads `.txt` / `.md` / `.pdf` files, chunks, embeds, stores
- `src/ask.py` — embeds your question, retrieves the closest chunks, asks the LLM
- `src/embeddings.py` — local, free embedding model (no API key needed)
- `src/llm.py` — swappable Gemini / Groq interface (both have free tiers)
- `src/prompts.py` — explicit prompt templates (QA, summary, structured extraction)
- `src/db.py` — all Postgres/pgvector access lives here
- `data/sample_docs/` — one sample doc so you can test the pipeline immediately

## Setup

### 1. Get free API keys (you only need one to start)

- Gemini: https://aistudio.google.com/apikey
- Groq: https://console.groq.com/keys

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and paste in your key(s). Leave `LLM_PROVIDER=gemini` (default) or
change it to `groq`.

### 3. Start Postgres

```bash
docker compose up -d
```

This pulls the `pgvector/pgvector:pg16` image and runs `sql/init.sql`
automatically on first boot, creating the `chunks` table with a vector column.

### 4. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run of the embedding model will download ~80MB of weights from Hugging
Face — one-time, then it's cached locally.

### 5. Ingest the sample doc

```bash
python -m src.ingest
```

This ingests everything under `data/sample_docs/` by default. To ingest your
own documents instead:

```bash
python -m src.ingest path/to/your/docs
```

### 6. Ask a question

```bash
python -m src.ask "Why do teams use RAG instead of fine-tuning?"
```

You should get a grounded answer plus a list of which chunks it was drawn
from, ordered by relevance.

## Sanity checks if something breaks

- `docker compose ps` — is the `knowledgeforge-postgres` container healthy?
- `docker compose logs postgres` — did `init.sql` actually run? Look for
  "CREATE EXTENSION" / "CREATE TABLE" in the log.
- Wrong API key → `ask.py` / `ingest.py` will raise a clear error from
  `config.validate()` before doing any work.
- Empty answer / "No documents ingested" → run `python -m src.ingest` first.

## What's next (Phase 2)

The agentic layer (LangGraph) gets added on top of this — it will call the
same `db.search()` and `llm.generate()` functions you already have here, so
nothing in Phase 1 gets thrown away.

## Phase 2: Agent Layer

Adds `src/agent.py` — a LangGraph agent that sits on top of everything in
Phase 1 without changing it.

**What it does differently from `ask.py`:**

- **Routes** each request: a quick keyword check, falling back to an LLM
  classification call only when the request is ambiguous, decides whether
  it's a simple lookup or a multi-step task.
- **Simple requests** go through the same retrieval + QA prompt as Phase 1.
- **Multi-step requests** (e.g. "compare X and Y in a table") retrieve more
  broadly, group the retrieved chunks by source document, and use a
  comparison prompt built for structured output.
- **Self-checks** multi-step answers with an LLM-as-judge pass: if the
  answer isn't grounded or didn't actually complete the task (e.g. no table
  when one was asked for), it retries once with the failure reason fed back
  into the prompt, then stops — it will never loop forever.
- Simple lookups skip the self-check call entirely to avoid spending an
  extra API call where it wouldn't change the outcome — the same
  cost-conscious instinct Phase 4 builds out fully.

### Install the new dependency

```bash
pip install -r requirements.txt
```

(This adds `langgraph` to what you already installed in Phase 1.)

### Try it

```bash
python -m src.agent "Why do teams use RAG instead of fine-tuning?"
```

Simple question → routes to `simple`, answers directly, no self-check.

```bash
python -m src.agent "Compare chunking and embedding, and give me a table"
```

Multi-step → routes to `multi_step`, retrieves broadly, produces a table,
self-checks it, and reports whether the check passed.

### Sanity check the routing yourself

```bash
python -c "from src.agent import classify_task; print(classify_task('What is RAG?'))"
python -c "from src.agent import classify_task; print(classify_task('Summarize all the docs and compare them'))"
```
Should print `simple` and `multi_step` respectively.

## What's next (Phase 3)

Evaluation & observability — a golden Q&A set scored with RAGAS, plus
Langfuse tracing on every call this agent already makes.
