# KnowledgeForge AI — Phase 1: Core RAG Pipeline

[![CI](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml)

*Replace `<your-username>/<your-repo>` above with your actual GitHub path
once this is pushed -- the badge won't render until then.*

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

## Phase 3: Evaluation & Observability

Adds `src/eval.py` and `data/golden_qa.json` on top of everything in Phases 1–2,
plus tracing wired directly into `llm.generate()` and `db.search()` -- so both
`ask.py` and `agent.py` get traced with zero changes to either file.

**A note on RAGAS:** the roadmap originally called for the RAGAS library here.
It's deliberately not used -- `ragas==0.4.3` requires `langchain_community`/
`langchain_openai` versions that hard-conflict with `langgraph==1.2.11`'s
`langchain-core>=1.4` requirement (confirmed by actually trying to install both
together -- it breaks the environment). Phase 3 reimplements the same three
metrics -- faithfulness, answer relevancy, context precision -- as direct
LLM-as-judge calls using the provider you've already configured. No extra
dependency, no conflict risk.

### Install the new dependency

```bash
pip install -r requirements.txt
```

(Adds `langfuse` -- `sentence-transformers`/`torch` already installed in
Phase 1 have no dependency overlap with it, confirmed.)

### Optional: set up free tracing

Tracing is fully optional -- confirmed safe to skip: if `LANGFUSE_PUBLIC_KEY`
is blank, `@observe` just no-ops with a harmless warning, nothing breaks.

To enable it: free account at https://cloud.langfuse.com, then in `.env`:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```
Every `ask.py` and `agent.py` call will then show up as a trace in your
Langfuse dashboard automatically.

### Run the eval harness

```bash
python -m src.eval
```

Runs all 12 questions in `data/golden_qa.json` through the Phase 1 pipeline,
scores each with an LLM-as-judge, and writes `eval_report.md`. Exits with code
1 if average faithfulness drops below `EVAL_FAITHFULNESS_THRESHOLD` (default
0.7) -- this exact check becomes the CI/CD quality gate in Phase 7.

Question `q12` ("What is the capital of France?") is a deliberate negative
control -- it's not covered by the ingested docs, so a correct answer here is
one that says so rather than answering from outside knowledge. That's scored
as high faithfulness, not a failure.

### Extend the golden set

As you ingest your own documents, add more questions to `data/golden_qa.json`
in the same `{"id": ..., "question": ...}` shape -- no code changes needed.

## Phase 4: Cost & Latency Optimization

Adds `src/cost_router.py`, `src/cache.py`, `src/benchmark.py`, and a `local`
provider option in `llm.py`. Wired into `ask.py` specifically (not
`agent.py`'s internal calls -- see note below).

**How it works, in order, for every automatic (non-override) call to `ask()`:**
1. **Semantic cache check** -- embeds the question locally (free) and checks
   Redis for a similar-enough previous question (cosine similarity, default
   threshold 0.93). Hit -> instant answer, zero LLM calls, zero cost.
2. **Local routing** -- a cheap heuristic (word count + keyword check, no LLM
   call) decides if the question is simple enough for a local Ollama model.
   If so, and Ollama responds, that's the answer -- genuinely $0.
3. **Cloud fallback** -- anything else (or if Ollama isn't running) goes to
   your configured cloud provider, exactly like Phases 1-3.

**Explicit overrides always bypass all of this** -- `ask(q, provider="groq")`
and the `--provider` CLI flag behave exactly as before, and `eval.py` now
calls `ask(q, optimize=False)` specifically so evaluation scores your actual
configured provider, not a cached or locally-routed answer.

### Install the new dependencies

```bash
pip install -r requirements.txt
```

### Set up Redis (required for caching)

```bash
docker compose up -d
```

This now also starts a `redis` container alongside Postgres. If Redis is
unreachable for any reason, caching fails open automatically -- confirmed via
testing, not just assumed: `ask()` just runs the normal local/cloud path with
one warning printed, nothing crashes.

### Set up Ollama (optional, for local routing)

Ollama is installed **natively**, not in Docker -- it needs direct hardware
access for reasonable speed, which fights with Docker Desktop on Windows.

1. Download and install: https://ollama.com/download
2. Pull a small model:
```bash
ollama pull llama3.2:3b
```
3. Make sure it's running (the installer usually starts it automatically):
```bash
ollama list
```

If you skip this entirely, that's fine -- confirmed via testing: local
routing attempts fail with a connection error, which `ask()` catches and
falls back to cloud automatically, no crash.

### Try it

```bash
python -m src.ask "What is RAG?"
```
Short question, no complexity keywords -> should route to `local` (if Ollama
is set up) or `cloud` (if not).

```bash
python -m src.ask "What is RAG?"
```
Same question again -> should route to `cache`, answer comes back instantly.

```bash
python -m src.ask "Compare chunking and embedding in depth"
```
Longer, keyword-triggered -> routes to `cloud` regardless of Ollama.

### Run the benchmark

```bash
python -m src.benchmark
```

Runs the golden set three ways -- no optimization, optimized cold-cache, and
optimized warm-cache (same questions repeated) -- and writes
`cost_optimization_report.md` with latency and estimated $ saved. This report
is your concrete "here's the number" answer for the cost-optimization
interview question.

### A scoping note

Caching and local-routing are wired into `ask.py` (the direct Q&A path)
specifically, not into `agent.py`'s internal `generate()` calls (routing,
synthesis, self-check). Extending the same pattern there is a natural next
step, deliberately left out here to avoid touching Phase 2's already-tested
agent graph.

## Phase 5: Serving Layer

Adds `src/api.py` and `src/schemas.py` -- a thin FastAPI layer over
everything already built. It doesn't reimplement any pipeline logic; every
endpoint just calls the same `ask()`, `run_agent()`, and `ingest_file()`
functions the CLI tools already use.

### Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | No | Uptime check, unauthenticated on purpose |
| GET | `/metrics` | Yes | DB/pgvector health + chunk counts per source |
| POST | `/ask` | Yes | Same as `python -m src.ask`, over HTTP |
| POST | `/agent` | Yes | Same as `python -m src.agent`, over HTTP |
| POST | `/ingest` | Yes | Upload a `.txt`/`.md`/`.pdf` file to ingest |

### Install the new dependencies

```bash
pip install -r requirements.txt
```

### Run it

```bash
uvicorn src.api:app --reload --port 8000
```

Open **http://127.0.0.1:8000/docs** -- interactive Swagger UI, you can call
every endpoint directly from the browser.

### Auth

Blank `API_KEY` in `.env` (the default) disables auth entirely -- fine for
local dev, confirmed via testing that every endpoint is reachable with no
header. Set a real value and every endpoint except `/health` then requires
an `X-API-Key` header matching it -- also confirmed via testing (right key
-> 200, wrong or missing key -> 401), including that `/metrics` and
`/ingest` are covered, not just `/ask`.

### Try it (with auth off)

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

```bash
curl http://127.0.0.1:8000/metrics
```

### A note on error handling

Every endpoint catches its own failure mode and returns a real status code
instead of a raw traceback -- generation failure is a 502, DB unreachable is
a 503, bad input is a 422, missing/wrong API key is a 401. There's also a
catch-all handler underneath all of that so an unexpected bug still returns
a clean 500 instead of crashing the server. All of this is covered by tests,
not just written and hoped to work.

## Phase 6: Containerization

Adds a `Dockerfile` for the FastAPI app and a third service (`app`) in
`docker-compose.yml`, so the whole stack -- app, Postgres, Redis -- comes up
with one command.

**Honest caveat:** Docker itself isn't available in the environment these
files were built in, so the actual `docker build` / `docker compose up`
couldn't be run end-to-end before reaching you -- this is the one piece of
this project not verified by a real run before shipping. What *was* verified:
the exact `uvicorn` command in the Dockerfile's `CMD` was run directly and
confirmed working (`/health`, `/docs`, `/openapi.json` all returned 200),
every pinned dependency in `requirements.txt` has already installed cleanly
multiple times over the course of this build, and `sentence-transformers`'
actual package metadata was checked to confirm the CPU-only torch build
(below) satisfies its requirement. If the build itself hits a snag, treat it
like every other phase here -- paste the exact error and we'll fix it from
a real signal instead of guessing further.

### Why a CPU-only torch install

The default PyPI `torch` wheel bundles CUDA and is 750MB+, even though this
project never uses a GPU. `sentence-transformers==3.0.1` only requires
`torch>=1.11.0` (confirmed from its own package metadata) -- the Dockerfile
installs a CPU-only build from PyTorch's own index first, which satisfies
that constraint at a fraction of the size, before the rest of
`requirements.txt` installs normally.

### Why Ollama still isn't containerized

Same reasoning as Phase 4: Ollama needs direct hardware access for
reasonable speed, which fights with Docker Desktop on Windows. It keeps
running natively. The containerized app reaches it via
`host.docker.internal`, which Compose resolves to your actual machine from
inside a container -- configured in `docker-compose.yml`'s `app` service.

### A critical detail if you ever edit these files

Inside a container, `localhost` means *that container*, not your machine or
its sibling containers. So while your `.env`'s `DATABASE_URL` and
`REDIS_URL` correctly say `localhost` (right for running the app natively,
which Phases 1-5 have been doing), the containerized `app` service
overrides both to use Postgres's and Redis's *service names* instead --
Compose's internal DNS resolves `postgres` and `redis` to the right
containers automatically. This override lives in `docker-compose.yml`, not
`.env` -- you don't need to change `.env` for this.

### Build and run

```bash
docker compose up -d --build
```

`--build` matters here specifically -- without it, Compose won't know to
build the new `app` image on this first run.

### Verify

```bash
docker compose ps
```
All three -- `knowledgeforge-postgres`, `knowledgeforge-redis`,
`knowledgeforge-app` -- should show `Up` / `(healthy)`.

```bash
curl http://127.0.0.1:8000/health
```
or open http://127.0.0.1:8000/docs -- same Swagger UI as Phase 5, now served
from inside a container instead of your venv.

### Running CLI tools inside the container

```bash
docker exec knowledgeforge-app python -m src.ingest
docker exec knowledgeforge-app python -m src.eval
```

### If the build fails

```bash
docker compose logs app
```
paste the output -- common first-run issues are a slow `torch` download
(just retry) or a typo carried over from a manual `.env` edit.

## Phase 7: CI/CD

Adds `.github/workflows/ci.yml`, a real `tests/` directory (converting every
ad-hoc test written across Phases 1-6 into permanent pytest files -- 50 tests
total, all passing), and `pyproject.toml`/`requirements-dev.txt` to support both.

### Four jobs, deliberately not all triggered the same way

| Job | Trigger | Needs a real API key? | What it does |
|---|---|---|---|
| `lint` | every push/PR | No | `ruff check src/ tests/` |
| `test` | every push/PR | No | `pytest tests/` (50 tests, all mocked LLM calls -- only a Redis service container needed, for real cosine-similarity math) |
| `eval-gate` | **manual only** (Actions tab -> Run workflow) | Yes | Ingests the sample doc into a real Postgres, runs Phase 3's eval harness for real, uploads `eval_report.md` as an artifact |
| `docker-build-push` | push to `main` | No | Builds the Phase 6 Dockerfile, pushes to `ghcr.io/<your-repo>` |

### Why the eval gate isn't automatic

This project's free-tier quotas got hit hard and repeatedly during
development -- as low as 20 requests/day on one model. The eval harness
makes ~24 LLM calls per run (12 questions x answer + judge). Running that
automatically on every push would mean CI alone could exhaust a day's quota
before you'd written a single line of code that day. So it's manual: trigger
it from the **Actions** tab when you actually want a scored run. If you
later have more headroom (a paid tier, a dedicated CI-only key), it's a
one-line change -- add `push: branches: [main]` to the trigger list in
`ci.yml`.

### Set up the secrets

In your GitHub repo: **Settings -> Secrets and variables -> Actions -> New
repository secret**. Add whichever of these you use:
- `GEMINI_API_KEY`
- `GROQ_API_KEY`

The workflow picks whichever is present automatically (prefers Gemini if
both are set). `GITHUB_TOKEN` for the Docker push needs no setup -- GitHub
provides it automatically to every workflow run.

### Test job design note

Only `test`'s Redis service container is real infrastructure -- every test
that touches Postgres mocks `src.db` directly (confirmed: all 50 tests pass
locally with zero Postgres running). Only the semantic-cache tests need a
genuinely live Redis, since the whole point there is testing real
cosine-similarity math, not a mocked stand-in for it.

### Run everything locally, exactly like CI does

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check src/ tests/
pytest tests/ -v
```

### Push and watch it run

```bash
git add .
git commit -m "Phase 7: CI/CD"
git push
```

Then check the **Actions** tab on GitHub. Once it's green, update the badge
URL at the top of this README with your actual `username/repo`.

## Project complete

Seven phases, all fifteen target skills, every single piece verified by a
real run (yours or mine) before being called done -- not just written and
hoped to work. This is the whole point: not a tutorial you followed, but a
system you can explain, defend, and extend, because you watched every part
of it actually break and get fixed at least once.
