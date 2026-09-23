# 🎧 Handbook Assistant — Higher Institute Customer Support Agent

> An AI agent that answers student questions from the CS and IS department handbooks, escalates the ones it cannot answer confidently to a human advisor, and learns from resolved escalations — through a human-gated promotion step, not automatically.

**Status:** 🟢 Core loop, long-term memory and the promotion gate working end to end. The advisor UI is not built yet — see [Roadmap](#roadmap).

---

## What is this?

A support agent for the Higher Institute for Computer Science and Information Systems' CS and IS handbooks. Unlike a static FAQ bot, it runs a **Corrective RAG loop with confidence-gated escalation**: when the agent cannot answer from the handbook with confidence, it stops, writes a ticket, and routes the question to a human academic advisor instead of guessing.

The escalation is a **lifecycle, not an event**. The graph run ends the moment it escalates — nothing stays parked in memory waiting for a human who may take days. When an advisor resolves the ticket, a second write reconnects to the same conversation through its `thread_id`, and the student reads the answer whenever they next return.

Resolved answers can then be promoted back into the knowledge base as `instructor_resolved` chunks, so the same question is answered automatically next time. Promotion is a **separate, deliberate act** — never a side effect of resolving.

This is the second project in a 4-part AI engineering portfolio, building on patterns from [Mizan](https://github.com/OmarAmir2001/mizan) (CRAG, long-term memory) and adding orthogonal confidence gating, an async ticket lifecycle, and a human-gated learning loop.

---

## Core design principles

- **One source of truth, one derived index.** Postgres holds the authoritative state — tickets, chunks, conversation checkpoints. The pgvector collection is a **rebuildable projection** of it. Data flows one way: source → vectors. Drop the collection and re-push; nothing authoritative is lost.
- **Escalation is decided by orthogonal judges, never a self-reported confidence score.** Three single-purpose checks, each judging a different thing against different evidence. Escalate if **any** fails. No "rate your confidence 0–1" — models are badly calibrated at that, and averaging several prompts of the same model adds no independent information.
- **Escalation ends the run; resolution is a separate write.** An open ticket costs a database row, not a live process. That is what survives a restart and scales past a handful of open tickets.
- **The judges fail closed.** An unparseable or unreachable judge escalates to a human. Failing open would wave answers through precisely when the verification machinery is broken.
- **Promotion is human-gated and additive-only.** The handbook always outranks promoted ticket answers, enforced at retrieval — pgvector has no concept of "authoritative", and ticket answers are phrased in student language, so they out-retrieve formal policy text unless you rank deliberately.
- **Thin nodes, real logic in controllers.** A graph node reads state, calls one controller method, writes the result back. Every controller is testable without a graph, and the advisor's HTTP endpoint reuses the exact method the graph uses.

---

## Architecture

```
POST /api/v1/chat
   │
   ▼
retrieve_node            embed query → pgvector search → department filter →
   │                     handbook-precedence ordering
   ▼
grade_node               Gate 1: context relevance (pre-generation, cheapest first)
   │
   ├── fails ──────────► escalate_node ──► ticket (pending) ──► RUN ENDS
   │                                       "I've escalated this to an advisor."
   └── passes
          │
          ▼
   generate_node         answer built only from the retrieved excerpts
          │
          ├── no answer ─► escalate_node ──► RUN ENDS
          └── drafted
                 │
                 ▼
          judge_node     Gate 2a: faithfulness   ┐ run concurrently — independent
                         Gate 2b: answer relevance ┘ checks, so no added latency
                 │
                 ├── either fails ─► escalate_node ──► RUN ENDS
                 └── both pass ────► END  (answer returned to the student)
```

Gate 1 scores **coverage** of the retrieved excerpts, before paying for a generation call. Gate 2a decomposes the drafted answer into individual claims and scores `supported / total`. Gate 2b deliberately **does not see the excerpts** — an answer can be perfectly grounded and still answer the wrong question, and showing it the excerpts would reintroduce exactly the blind spot faithfulness already has.

**Resolution — a separate request, hours or days later:**

```
Advisor lists pending tickets       GET  /api/v1/escalation/tickets?ticket_status=pending
   │                                    (question, which gate tripped and why,
   ▼                                     and the excerpts the bot actually saw)
claims one                          POST /api/v1/escalation/tickets/{id}/claim   → under_review
   │
   ▼
submits an answer                   POST /api/v1/escalation/tickets/{id}/resolve → resolved
   │
   ├─► written into the persisted thread state, keyed by the ticket's thread_id
   │   → the student reads it from GET /api/v1/chat/{thread_id} on their next visit
   │
   └─► if promote_to_kb: embedded into pgvector as `instructor_resolved`,
       tagged with ticket_id, via delete-then-insert so a re-sync cannot duplicate it
```

Both derived writes are idempotent and repairable from the ticket alone, so neither can fail the advisor's request after their answer has committed.

---

## What works today

Verified end to end against real Postgres and live model calls:

- ✅ **Three orthogonal judge gates** — context relevance (pre-generation), faithfulness and answer relevance (post-generation, concurrent). Every verdict emits one `gate_evaluated` log line with score and threshold.
- ✅ **Confidence-gated escalation** with the judge's own reason string as the advisor-facing summary.
- ✅ **Fail-closed judges** — an unparseable verdict escalates rather than passing.
- ✅ **Ticket lifecycle** — `pending → under_review → resolved / rejected`, plus `reopened`, `closed`, `duplicate`, with a full `ticket_status_history` trail and optimistic concurrency (the expected status is in the `UPDATE ... WHERE`, so two advisors resolving at once cannot silently overwrite each other).
- ✅ **Passive delivery** — the advisor's answer is written into the LangGraph checkpointer under the original `thread_id`; the student pulls it on return.
- ✅ **Human-gated promotion** — `promote_to_kb` on resolve; idempotent delete-then-insert keyed on `ticket_id`; un-promoting or reopening removes the vector row.
- ✅ **Department-aware retrieval** — a CS student is never answered from the IS handbook.
- ✅ **Handbook precedence at retrieval** — relevance decides which chunks are used, precedence decides the order they are presented in.
- ✅ **The learning loop closes** — a question that escalated is answered automatically once its resolution is promoted.
- ✅ **Structured logging** — structlog, JSON, with the graph's `thread_id` bound as `correlation_id`. One grep follows a question from the chat request through each gate verdict into the ticket and on to the advisor's resolution. Uvicorn and SQLAlchemy log through the same renderer, so every line is the same shape.
- ✅ **Ingestion pipeline** — upload, chunk, embed, push to pgvector, with every chunk tagged `source` and `department` so the filter and the ranking have something real to work with.
- ✅ **Alembic migrations** and a Postgres checkpointer, both running automatically in the container.
- ✅ **Long-term student memory** — one patched profile per student (identity, not history). Extraction is frequency-gated (most turns cost no model call), runs on a smaller model, and happens after the response is sent. A patch can never delete a known fact.
- ✅ **The stored profile outranks the request body** for `department`, so a student cannot read the other handbook by claiming to be in it.
- ✅ **Conversation transcript** — turns accumulate across runs through a reducer, so an advisor's answer appends instead of overwriting. Bounded in storage (a trim reducer) and in the prompt (a turn cap plus a character budget, oldest evicted first).
- ✅ **Human-gated promotion judges** — a generalizability check sets the advisor's checkbox default, and a contradiction check *holds* promotion and flags the handbook for review. Deliberately asymmetric: only the contradiction check can block, because a wrong generalizability call would silently discard good knowledge while a wrong hold is merely visible. Both fail safe, in opposite directions.
- ✅ **Section-aware chunking** — markdown splits on its own headings, so a chunk's `section` is a real citable path (`... > أحكام وشروط الدراسة > مادة (١٠)`) rather than a character offset.
- ✅ **Idempotent handbook re-ingest** — `sync_sections` delete-then-inserts per `(source, section)`, so pushing twice replaces rather than duplicates, with no full collection rebuild.
- ✅ **One-command Docker setup** via `make up`, a Locust load profile, and 89 passing tests that need no database.

---

## Tech stack

| Component               | Technology                                                |
| ----------------------- | --------------------------------------------------------- |
| Agent framework         | LangGraph (Postgres checkpointer)                         |
| LLM — generation        | Groq · `openai/gpt-oss-120b`                              |
| LLM — judges            | Groq · `openai/gpt-oss-20b` (smaller: up to 3 calls/question) |
| Embeddings              | Cohere · `embed-multilingual-light-v3.0` (384-dim)        |
| Vector store            | pgvector                                                  |
| Tickets, chunks, assets | Postgres · SQLAlchemy 2 async + asyncpg                   |
| Migrations              | Alembic                                                   |
| Validation              | Pydantic v2 + pydantic-settings                           |
| API layer               | FastAPI                                                   |
| Logging                 | structlog (JSON, correlation ids)                         |
| Packaging               | uv · Python 3.13                                          |
| Local orchestration     | Docker Compose                                            |

**One database, three drivers.** The app talks to Postgres over **asyncpg**, LangGraph's `AsyncPostgresSaver` over **psycopg 3**, and Alembic over **psycopg2**. All three URLs are built from the same `Settings` object, so they cannot drift.

> There is no MongoDB. Earlier design notes describe Mongo as the source of truth for tickets; that was consolidated onto Postgres, which means resolving a ticket — loading thread state and writing the promoted answer to vectors — touches one database.

A Qdrant provider also exists behind the same `VectorDBInterface`, selectable with `VECTOR_DB_BACKEND=QDRANT`, but only the pgvector path is exercised.

---

## Knowledge base

Built from the institute's official handbooks in `data/handbooks/`:

- `CS_2023.md` — Computer Science
- `IS_2023.md` — Information Systems

Chunks are embedded into pgvector with metadata that retrieval depends on:

| Metadata key | Purpose                                                              |
| ------------ | -------------------------------------------------------------------- |
| `source`     | `CS_2023` / `IS_2023` / `instructor_resolved` — drives precedence      |
| `department` | `CS` / `IS` / unset (shared) — drives the retrieval filter            |
| `section`    | Handbook section, so an answer can be traced back                     |
| `ticket_id`  | On promoted answers only: the stable key the sync deletes and re-inserts by |

`source` is normalised to the handbook name rather than the file path, because stored filenames carry a random prefix (`3EsFAHwA7Z8L_CS_2023.md`) and precedence matches on `source`.

---

## Running it

### Docker (one command)

```bash
git clone <repo-url> && cd Customer_Support

cp .env.example .env
# Fill in GROQ_API_KEY and COHERE_API_KEY, plus POSTGRES_* credentials

make up          # or: make rebuild to build first
```

That starts Postgres with pgvector, waits for it to accept connections, applies migrations, and serves the API on <http://127.0.0.1:8000> (docs at `/docs`). `POSTGRES_HOST` is overridden to the `pgvector` service name inside the network, so the same `.env` works on the host and in the container. `make help` lists the rest.

**Why `make` and not `docker compose` directly.** The compose file lives in `docker/` while the stack is built and configured from the repo root, and that needs two flags:

```bash
docker compose -f docker/docker-compose.yml --env-file .env up -d
```

`--env-file` is load-bearing. Compose takes its *project directory* from the compose file's location, so without it `${POSTGRES_USERNAME}` is looked up in `docker/.env` and silently expands to an empty string. `--project-directory ..` is **not** the fix — it also re-roots every relative path in the file, turning `env_file: ../.env` into a path outside the repo. `--env-file` changes only where variables come from.

`.dockerignore` stays at the repo root on purpose: Docker resolves it against the build *context*, which is the root, not against the Dockerfile's directory.

### Locally with uv

```bash
docker compose -f docker/docker-compose.yml --env-file .env up -d pgvector   # just the DB
cp .env.example .env               # POSTGRES_HOST=localhost
cp alembic.ini.example alembic.ini

uv sync
uv run alembic upgrade head
uv run uvicorn customer_support.main:app --reload
```

Set `LOG_JSON=false` for coloured, human-readable logs while developing; keep JSON everywhere else.

### Loading the knowledge base

```bash
# 1. Upload a handbook (.md, .pdf, .txt, or a pre-chunked .json)
curl -F "file=@data/handbooks/CS_2023.md" \
     http://127.0.0.1:8000/api/v1/admin/ingest/1

# 2. Chunk it into Postgres
curl -X POST http://127.0.0.1:8000/api/v1/admin/process/1 \
     -H 'Content-Type: application/json' \
     -d '{"chunk_size":1000,"overlap":50,"do_reset":1}'

# 3. Embed and index into pgvector
curl -X POST http://127.0.0.1:8000/api/v1/admin/knowledge_base/push/1 \
     -H 'Content-Type: application/json' -d '{"do_reset":1}'
```

`KB_COLLECTION_NAME` must match the project you pushed (`collection_1` for project `1`).

### Tests

```bash
uv run pytest tests -q --no-cov
```

No database, no API keys, no graph — the gates, the ticket state machine and the retrieval ranking are all pure logic by design.

---

## API

| Method | Path                                             | Purpose                                            |
| ------ | ------------------------------------------------ | -------------------------------------------------- |
| `GET`  | `/`                                              | Health — app name and version                      |
| `POST` | `/api/v1/chat`                                   | Ask a question. Returns an answer, or an escalation with `ticket_id` |
| `GET`  | `/api/v1/chat/{thread_id}`                       | Read a conversation, including an advisor's answer  |
| `GET`  | `/api/v1/escalation/tickets`                     | Advisor queue. Filter by `ticket_status`, `department`; paginated |
| `GET`  | `/api/v1/escalation/tickets/{id}`                | Full ticket: gate scores and the excerpts the bot saw |
| `POST` | `/api/v1/escalation/tickets/{id}/claim`          | `pending → under_review`                            |
| `POST` | `/api/v1/escalation/tickets/{id}/resolve`        | Deliver an answer, optionally promote it to the KB  |
| `POST` | `/api/v1/admin/ingest/{project_id}`              | Upload a handbook file                              |
| `POST` | `/api/v1/admin/process/{project_id}`             | Chunk uploaded files into Postgres                  |
| `POST` | `/api/v1/admin/knowledge_base/push/{project_id}` | Embed and index chunks into pgvector                |
| `POST` | `/api/v1/admin/knowledge_base/search/{project_id}` | Debug retrieval exactly as the agent sees it      |
| `GET`  | `/api/v1/admin/index_info/info/{project_id}`     | Collection stats                                    |

`/api/v1/profile/{student_id}` returns what long-term memory knows about a student, and `DELETE` on it wipes that profile (privacy and reset requests). Conversation history is read per thread from `GET /api/v1/chat/{thread_id}`; there is no per-student history endpoint yet — see the roadmap.

`rejected`, `closed` and `reopened` transitions exist in the state machine and on `EscalationController` but are not yet exposed over HTTP.

---

## Project structure

```
src/customer_support/
  main.py                    # lifespan: build clients, wire layers, compile the graph
  helpers/
    config.py                # Settings — every threshold and model id
    logging_config.py        # structlog + correlation ids
  graph/                     # thin nodes + edges + wiring, no logic
    nodes.py  edges.py  builder.py  dependencies.py
  controllers/               # all real logic lives here
    RetrievalController.py   # pgvector query, department filter, precedence
    GradingController.py     # the three judge gates
    GenerationController.py  # answer drafting
    EscalationController.py  # ticket lifecycle + vector sync (the only writer)
    judge_prompts.py         # prompts + provenance labelling, versioned as code
    ProcessController.py     # chunking + metadata tagging
    KBController.py          # embed and index
    DataController.py  ProjectController.py  BaseController.py
  models/                    # data shapes, plus the SQL that persists them
    TicketModel.py           # the only place ticket SQL is issued
    ChunkModel.py  AssetModel.py  ProjectModel.py
    db_schemas/.../ticket.py # tickets + ticket_status_history tables
    enums/                   # GateEnum, TicketStatusEnum (the state machine)
    graph/graph_state.py     # the typed state flowing through the graph
    llm_schemas/             # GateResult, JudgeVerdict — parsed LLM output
  routers/                   # HTTP only: no queries, no prompts
    chat.py  escalation.py  admin.py  health.py  schemas/
  stores/                    # external systems behind interfaces
    checkpointer.py          # LangGraph Postgres saver
    llm/                     # LLMInterface + OpenAI/Groq and Cohere providers
    vectordb/                # VectorDBInterface + pgvector and Qdrant providers
migrations/                  # Alembic (ignores LangGraph's own checkpoint tables)
data/handbooks/              # CS_2023.md, IS_2023.md — the ingestion input
handbook/                    # the original scanned PDFs, archival source of truth
docker/                      # Dockerfile, compose file, entrypoint
  Dockerfile                 # built with the REPO ROOT as context
  docker-compose.yml         # paths point up; see "Why make" above
  docker-entrypoint.sh       # waits for Postgres, migrates, then starts uvicorn
tests/                       # gates, transitions, ranking, locales, memory, promotion
  load/locustfile.py         # read-only and full-pipeline load profiles
Makefile                     # wraps the compose flags so they cannot be forgotten
.dockerignore                # stays at the ROOT: resolved against the build context
```

**Dependency direction:** `routers/` and `graph/` call `controllers/`; `controllers/` call `models/` and `stores/`. Never the reverse. `EscalationController` receives the graph as an injected `thread_writer` rather than importing it, so the arrow stays one-way.

---

## Configuration worth knowing

| Setting                            | Default | Why it matters                                                        |
| ---------------------------------- | ------- | --------------------------------------------------------------------- |
| `GATE_CONTEXT_RELEVANCE_THRESHOLD` | `0.5`   | Starting values, not tuned. These three are the biggest lever on the escalation rate; the `gate_evaluated` log lines are the raw material for tuning them |
| `GATE_FAITHFULNESS_THRESHOLD`      | `0.8`   |                                                                        |
| `GATE_ANSWER_RELEVANCE_THRESHOLD`  | `0.7`   |                                                                        |
| `RETRIEVAL_TOP_K`                  | `5`     | Chunks handed to the generator                                         |
| `RETRIEVAL_OVERFETCH_FACTOR`       | `3`     | Fetch `top_k × this`, because filtering discards rows                   |
| `JUDGE_MAX_OUTPUT_TOKENS`          | `1200`  | Reasoning models spend tokens before emitting JSON; too low truncates the verdict and the whole judgement is rejected |
| `INPUT_DEFAULT_MAX_CHARACTERS`     | `8000` in `.env.example` (unset = no truncation) | `generate_text` truncates its prompt to this — set it below the context size and the excerpts get cut out of the answer prompt |
| `LOG_JSON`                         | `true`  | `false` for coloured local logs                                        |

---

## Skills demonstrated

- Corrective RAG with orthogonal, single-purpose judge gates — hand-written, no eval framework in the request path
- Confidence-gated escalation with an async, durable human-in-the-loop resolution flow
- A ticket lifecycle reconnected across separate requests via `thread_id`, with a real state machine and an audit trail
- Source-of-truth discipline: an idempotent, rebuildable derived index with delete-then-insert sync keyed on a stable id
- Retrieval ranking that enforces a policy (handbook precedence) the vector store has no concept of
- Clean controller architecture — thin nodes, logic reused by both the graph and the API, testable without either
- Production logging: one renderer for app and library logs, correlation ids through `contextvars`
- FastAPI + Pydantic v2, SQLAlchemy 2 async, Alembic migrations, Docker Compose, uv

---

## Roadmap

Deferred deliberately — the core loop works without them:

- [ ] **Per-student conversation index** — a `conversations` table (`thread_id` PK, `subject_id`, `created_at`, `last_message_at`, `turn_count`) upserted on each turn. Needed for "continue where I left off", for an advisor to see a student's other threads, and for the stale-ticket scanner to find abandoned ones. The checkpointer cannot answer this: it is keyed by `thread_id` and stores state as opaque blobs, and `tickets` only links a thread to a student when the conversation escalated. Lands with the advisor dashboard.
- [ ] **Handbook review queue** — held tickets are recorded with `promotion_held` and a reason, but nothing surfaces them yet. The partial index exists; the endpoint and dashboard view do not.
- [ ] **Stale-ticket scanner** — one scheduled job over `ticket_status_history` ("time since last transition"), never a timer per ticket. Remind for never-picked-up tickets, auto-close resolved-but-unconfirmed ones.
- [ ] **Duplicate detection** — match an incoming question against already-resolved tickets and auto-resolve by pointing at the existing answer.
- [ ] **Offline evaluation** — Hit Rate / MRR on a labelled set, to calibrate the gate thresholds that are currently guessed.
- [ ] **Advisor endpoints for `reject` / `reopen` / `close`**, and the UI — a **Tailwind** front end built to existing designs (student chat + advisor dashboard). Not Gradio; earlier notes that say Gradio are superseded.
- [ ] **Streaming responses** and a deployed demo.
- [ ] **Domain-agnostic configuration** (deliberately deferred — see below).

### On making this domain-agnostic

The target is **multi-domain** — one client per deployment, configured — not multi-tenant. Multi-tenant is a different product: `tenant_id` on every table and query, KB partitioning, thread-id scoping, isolation tests.

It is deferred because the coupling is shallow, not because it is hard. The engine layers — gates, ticket lifecycle, source-of-truth sync, locale parser, provider factories — are already domain-neutral. The academic vocabulary is confined to:

- `RetrievalController.HANDBOOK_SOURCES`
- `ProcessController.HANDBOOK_DEPARTMENTS`
- `student_profile.DEPARTMENTS`
- the two `^(CS|IS)$` patterns, in `ChatRequest` and the escalation list query
- the domain nouns in `locales/*/rag.py`, `judges.py` and `GateFailureReason`
- `student_id` as a field name (52 occurrences, and part of the public request body)

Generalising means moving those to `Settings`, putting the domain nouns behind placeholders, and adding an `attributes` JSONB bag to the profile for tenant-specific facts like `gpa`. The rule for what stays fixed: **a state field is core if the engine reads it, and an attribute if only prompts and filters read it.** A partition key must exist for retrieval to filter on; that it is called "department" and holds CS/IS is configuration.

Renaming `student_id` is the one item that gets more expensive with time, and the trigger is the UI — that is when a field name first gets hardcoded by a client. Renaming the `instructor_resolved` metadata tag is *not* expensive: pgvector is a rebuildable projection, so it is a re-push.

---

## License

MIT

---

*Part of an AI Engineering portfolio. Other projects: [Mizan](https://github.com/OmarAmir2001/mizan), Research & Report Generator, AI Code Reviewer.*
