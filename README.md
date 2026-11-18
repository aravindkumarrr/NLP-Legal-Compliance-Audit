# NLP-Based Legal Compliance Auditing

## Stack
- Backend: Python + FastAPI
- Relational DB: PostgreSQL
- Vector DB: Qdrant
- Graph DB: Neo4j Community Edition (Deontic Logic Knowledge Graph)
- Embeddings: sentence-transformers (all-MiniLM-L6-v2, CPU-friendly)
- Deontic classifier baseline: Ollama + a small local LLM (prompted, no training needed to start)
- Frontend: React + Vite (`frontend/`), black theme

## Getting started

### 1. Start the databases
```bash
docker compose up -d
```
This brings up Postgres (`localhost:5432`), Qdrant (`localhost:6333`), and Neo4j
(`localhost:7687`, browser UI at `localhost:7474`).

### 2. Set up the backend
```bash
cd backend
python3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. (Optional, if using Ollama) Install Ollama
Download from https://ollama.com, then pull a small model:
```bash
ollama pull llama3.2:3b
```

### 4. Configure environment
```bash
cp .env.example .env
```
Defaults run everything locally against Ollama. To use Groq instead (see
"Switching to Groq" below), edit `.env` after copying it.

### 5. Run the API
```bash
uvicorn app.main:app --reload
```
Visit `http://localhost:8000/docs` for the interactive API docs, and
`http://localhost:8000/health` to confirm all three databases are reachable.
On first run, the app creates the Postgres tables automatically (simple
`create_all`, not Alembic migrations — fine at this stage).

### 6. Seed the regulatory reference corpus (for compliance checks)
For quick smoke-testing with zero setup:
```bash
python -m scripts.ingest_eurlex_sample
```
Loads `sample_data/eurlex_sample.jsonl` — 8 hand-written illustrative
regulatory-style passages — into Qdrant. Fine for exercising the
pipeline end to end, **not a substitute for the real corpus** when
actually evaluating compliance output.

For the real dataset, once you've downloaded EURLEX57K (train.jsonl /
dev.jsonl / test.jsonl — see
[nlp.cs.aueb.gr/software_and_datasets/EURLEX57K](http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/)):
```bash
python -m scripts.ingest_eurlex57k --data-dir /path/to/EURLEX57K
```
e.g. with a layout like:
```
NLP Project/
├── Dataset/data/datasets/EURLEX57K/{train,dev,test}.jsonl
└── Code Folder/v1/scaffold/backend/        <- run the command from here
```
run (from `backend/`):
```bash
python -m scripts.ingest_eurlex57k --data-dir "../../../Dataset/data/datasets/EURLEX57K"
```
This replaces the 8-passage sample (deletes + recreates the collection
by default) with a random, reproducible sample of real documents —
500 from `train` by default — each run through the same
`segment_clauses()` used on uploaded contracts so the reference corpus
and audited documents sit at a comparable passage granularity (header /
recitals / each article, further split wherever an article has internal
lettered/numbered sub-points). See the script's docstring for why a
full 45k-document pass isn't the default, and `--max-documents` /
`--splits` / `--seed` to change the sample size, which splits to draw
from, or get a different (or a wider, uncapped) sample.

### 7. Try the pipeline end to end
```bash
curl -X POST http://localhost:8000/documents/run-pipeline \
  -F "file=@/path/to/a/contract.pdf"
```
Or call each stage separately for visible per-stage progress:
`/documents/upload` → `/documents/{id}/index` → `/documents/{id}/classify`
→ `/documents/{id}/graph` → `/documents/{id}/audit` → `/documents/{id}/report`.

## Frontend
```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```
The API must be running on `localhost:8000`. Vite proxies `/api/*` to it
(change the target with `VITE_BACKEND_URL`). Features: service health,
PDF/DOCX upload, per-stage progress (or one-call run), clause preview,
and a report view with clauses, contradictions, compliance results and
JSON download. Recent audits are remembered in the browser only, since
the backend has no list-documents endpoint.

## LLM backend: Ollama vs Groq
Classifying every clause with a local 3B model over Ollama is slow --
one request at a time, on CPU. Groq hosts much larger models
(`openai/gpt-oss-120b` by default) at high inference speed over an API.
Both LLM-touching stages (deontic classification, stage 3; compliance
judgment, stage 7b) go through the same `pipeline/llm_client.py`
abstraction and never talk to Ollama or Groq directly.

**Which backend is active is a hardcoded constant**, not an env var --
see `ACTIVE_LLM_PROVIDER` at the top of `app/core/config.py`:
```python
ACTIVE_LLM_PROVIDER: str = "groq"  # "ollama" | "groq"
```
Flip that one line and restart the API to switch backends everywhere.
It currently defaults to `"groq"` in this repo. `LLM_PROVIDER` in `.env`
still works as a one-off override if you want to test the other backend
without touching code, but the constant is the source of truth.

To actually use Groq you still need your own key:

1. Get one from https://console.groq.com/keys
2. Put it in `backend/.env` (already gitignored -- **never** put a real
   key in `config.py` or anywhere else that could end up committed or
   zipped into a submission):
   ```
   GROQ_API_KEY=your_key_here
   ```
3. `pip install -r requirements.txt` (pulls in the `groq` package)
4. Restart the API.

`LLM_MAX_WORKERS` (default `5`) controls how many clauses are
classified/assessed concurrently -- that's what actually fixes
`/classify` latency on Groq, since it's a hosted API built for
concurrent traffic (check your account's rate limit before raising it
further). Drop it back to `1` if you switch `ACTIVE_LLM_PROVIDER` back
to `"ollama"` -- a local single-instance server just queues concurrent
requests anyway, so concurrency does nothing for it but add overhead.
`openai/gpt-oss-20b` is the faster/cheaper Groq model if 120b is
overkill for clause-level classification.

Ollama and Groq deprecate/rename models over time (Meta's
`llama-3.1-8b-instant` and `llama-3.3-70b-versatile` were both retired
from Groq's catalog in mid-2026, for instance) -- if `GROQ_MODEL`
starts returning errors, check https://console.groq.com/docs/models for
the current list.

## Project layout
```
backend/
  app/
    api/        # FastAPI route handlers (one per pipeline stage, see documents.py)
    core/       # config, DB clients
    models/     # SQLAlchemy models (db_models.py) / Pydantic schemas (schemas.py)
    pipeline/   # extraction, segmentation, embedding, deontic, graph_builder,
                #   graphrag, reranker, contradiction, compliance, report
    services/   # pipeline_service.py — orchestrates all stages, called by the API
    main.py     # FastAPI app entrypoint
  scripts/      # ingest_eurlex_sample.py — small illustrative seed (smoke-testing only)
                #   ingest_eurlex57k.py   — seeds the reference corpus from the real EURLEX57K dataset
  sample_data/  # eurlex_sample.jsonl — illustrative regulatory passages, not the real dataset
  tests/
docker-compose.yml
```

## Build order
1. ✅ Preprocessing (text extraction + clause segmentation)
2. ✅ Embeddings + Dense Passage Retrieval (Qdrant)
3. ✅ Deontic logic classifier (Ollama-prompted baseline)
4. ✅ Deontic Logic Knowledge Graph construction (Neo4j)
5. ✅ GraphRAG retrieval (combine Qdrant + Neo4j)
6. ✅ Cross-encoder re-ranking
7. ✅ Contradiction detection & compliance assessment
8. ✅ Reporting module
9. ✅ Frontend (`frontend/`)

## Known limitations / next things to harden
- **Penalty linkage is regex-based**, matching explicit cross-references
  ("Section 3.2", "Clause 5") in penalty clause text. Penalties described
  without an explicit reference won't get linked in the graph — worth
  calling out honestly in a demo rather than treating the graph as complete.
- **The deontic classifier and compliance judgments are prompted, not
  trained.** LEDGAR fine-tuning (classifier) and ContractNLI fine-tuning
  (contradiction detection) are the documented upgrade path once the
  prompted baselines' error patterns are understood on real documents —
  don't skip evaluating the baseline first. Note that switching to Groq
  changes *which* model is prompted, not this limitation — a bigger
  hosted model may follow the prompt's JSON format more reliably, but
  it's still a prompted baseline, not a fine-tuned classifier.
- **The EURLEX57K reference corpus, by default, is a random sample** (500
  documents from `train`, see `scripts/ingest_eurlex57k.py`), not the full
  57k-document set — a full pass is possible (`--max-documents 0
  --splits all`) but slow on CPU and mostly indexes legislative material
  unrelated to the kind of commercial clauses this tool audits. The
  compliance relevance gate (`compliance.py`'s `_RELEVANCE_MIN_SCORE`)
  is what keeps that irrelevant majority from corrupting individual
  judgments either way. `scripts/ingest_eurlex_sample.py`'s 8
  hand-written passages remain available for zero-setup smoke-testing.
- **No auth, no retries on the Ollama calls, no batching** — fine for a
  project-stage build, worth flagging as future work rather than pretending
  it's production-ready.
