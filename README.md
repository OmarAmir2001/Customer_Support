# 🎧 Handbook Assistant — Higher Institute Customer Support Agent

> An AI agent that answers student questions from the CS and IS department handbooks, escalates uncertain or sensitive cases to a human advisor, and gets smarter over time from resolved escalations — safely.

**Status:** 🟡 In Progress

---

## What is this?

This agent answers questions about the Higher Institute for Computer Science and Information Systems' CS and IS department handbooks. Unlike a static FAQ bot, it uses a **Corrective RAG (CRAG) loop with confidence-based escalation** — when the agent isn't confident enough in its answer, it stops and routes the question to a human academic advisor instead of guessing.

It also remembers each student across sessions (name, student ID, department, GPA, preferred language) and **learns from resolved escalations** — once an advisor answers a question the agent couldn't, that Q&A pair can be promoted back into the knowledge base (through a human-gated quality check) so the same question is auto-resolved next time.

This is the second project in a 4-part AI engineering portfolio, building directly on patterns established in [Mizan](https://github.com/OmarAmir2001/mizan) (CRAG, long-term memory, Trustcall) while adding new skills: orthogonal confidence-gated escalation, an async ticket lifecycle, a safe human-gated learning loop, and a clean controller-based service architecture.

---

## Core Design Principles

This project is built on a set of deliberate architecture decisions:

- **Two databases, one source of truth.** MongoDB holds documents and state (student profiles, escalation tickets). pgvector holds embeddings. Mongo (and the handbook files on disk) are the **source of truth**; pgvector is a **rebuildable derived index**. Data flows one direction only: source → pgvector.
- **Escalation is decided by orthogonal judges, not a self-reported confidence score.** The agent runs separate single-purpose checks (context relevance → faithfulness → answer relevance) and escalates if **any** of them fails. No "rate your confidence 0–1."
- **Escalation ends the run; resolution is a fresh run.** The graph does not stay paused waiting for a human who may take hours or days. It writes a ticket and ends. When the advisor resolves, a separate short run reconnects to the same conversation via `thread_id`.
- **The learning loop is human-gated.** Resolving a ticket delivers the answer to the student. Promoting that answer into the knowledge base is a *separate*, human-confirmed act — the machine only pre-fills the decision. The handbook always outranks resolved-ticket answers.
- **Clean separation.** Graph nodes are thin orchestrators; all real logic (LLM calls, DB queries, ticket state) lives in controllers that both the graph and the API reuse.

---

## Architecture

```
Student Question
    │
    ▼
load_memory              ← reads the student profile (single patched JSON doc)
    │
    ▼
retrieve_node            ← embeds query, searches pgvector (filtered by department: CS/IS)
    │
    ▼
context_relevance_gate   ← LLM judge: do retrieved chunks actually address the question?
    │
    ├── fails ──────────► escalate_node ──► writes ticket (status: pending) ──► RUN ENDS
    │                                        "I've escalated this to an advisor."
    │
    └── passes
            │
            ▼
     generate_response
            │
            ▼
  faithfulness_+_relevance_gate  ← LLM judge: is the answer grounded AND on-topic?
            │
            ├── fails ──────────► escalate_node ──► writes ticket ──► RUN ENDS
            │
            └── passes
                    │
                    ▼
              save_memory        ← Trustcall patches the student profile (async, small model)
                    │
                    ▼
                   END
```

**The escalation resolution path (a separate run, hours/days later):**

```
Advisor opens dashboard ──► sees pending tickets (from Mongo)
    │
    ▼
picks one (→ under_review), types an answer
    │
    ├─► answer delivered to the student (waits in persisted thread state; student sees it on return)
    │
    ▼
machine pre-checks: is this answer general? does it contradict the handbook?
    │
    ▼
advisor confirms the "Add to knowledge base" checkbox
    │
    ├── checked  ──► answer embedded into pgvector as `instructor_resolved`
    │                (unless it contradicts the handbook → held for handbook review)
    │
    ▼
ticket → resolved
```

---

## Key Features

- [ ] **Corrective RAG (CRAG)** with orthogonal, single-purpose LLM-judge gates (context relevance, faithfulness, answer relevance)
- [ ] **Department-aware retrieval** — filters pgvector results to CS or IS handbook based on the student profile
- [ ] **Confidence-based escalation** — escalates if any judge gate fails, rather than guessing
- [ ] **Async ticket lifecycle** — escalation ends the graph run; resolution is a separate run reconnected by `thread_id`
- [ ] **Structured escalation summaries** — advisor sees student context, the question, what was retrieved, and which gate tripped and why
- [ ] **Ticket state machine** — `pending → under_review → resolved / rejected`, plus `reopened` and `duplicate`, with full `status_history`
- [ ] **Long-term student memory** — single patched profile (name, student ID, department, GPA, preferred language) via LangGraph Store + Trustcall
- [ ] **Safe self-learning knowledge base** — resolved escalations are promoted to pgvector through a human-gated quality check, never automatically
- [ ] **Handbook precedence** — the handbook always outranks resolved-ticket answers; contradictions flag the handbook for review
- [ ] **FastAPI service layer** — `/chat`, `/history/{student_id}`, `/resolve`, `/health` endpoints with routers and Pydantic validation
- [ ] **Gradio UI** — chat interface with department selector and escalation status indicator
- [ ] **Streaming responses**

---

## Tech Stack

| Component            | Technology                        |
| -------------------- | --------------------------------- |
| Agent Framework      | LangGraph                         |
| LLM (generation)     | Groq — llama-3.3-70b-versatile    |
| LLM (memory extract) | Smaller/faster model (right-sized)|
| Embeddings           | intfloat/multilingual-e5-large    |
| Vector Store         | pgvector (Postgres)               |
| Document/State Store | MongoDB (Motor, async)            |
| Checkpointer         | Postgres (LangGraph)              |
| Memory (short-term)  | LangGraph checkpointer (thread-scoped) |
| Memory (long-term)   | LangGraph Store + Trustcall       |
| Validation           | Pydantic v2                       |
| API Layer            | FastAPI                           |
| UI                   | Gradio                            |
| Package Management    | uv                               |
| Deployment           | HuggingFace Spaces                |

> **Note on databases:** Mongo holds tickets and profiles for now; a later migration to consolidate on Postgres is possible but not planned yet. The `thread_id` link between the checkpointer and the escalation ticket works across databases regardless.

---

## Knowledge Base

The agent's knowledge base is built from the Higher Institute's official department handbooks:

- `CS_2023.md` — Computer Science department handbook
- `IS_2023.md` — Information Systems department handbook

Each file is split by section headers (`##`) into chunks, embedded with `multilingual-e5-large`, and stored in pgvector with metadata (`source`, `section`) so retrieved answers can be traced back to the exact handbook section. Resolved escalations that pass the human-gated quality check are added as additional chunks tagged `source: instructor_resolved` and linked back to their Mongo ticket via `ticket_id`.

---

## Project Structure

```
app.py                     # Gradio UI — entry point
api/
  main.py                  # FastAPI app
  routers/
    chat.py                # /chat, /history, /health
    escalation.py          # /resolve + advisor endpoints
controllers/               # all real logic lives here
  retrieval.py             # RetrievalController — pgvector query + department filter
  grading.py               # GradingController — the CRAG judge gates
  escalation.py            # EscalationController — ticket lifecycle + vector sync
  memory.py                # MemoryController — Trustcall extractor (small model, gated)
model/                     # data shapes ONLY (no logic)
  state.py                 # graph State schema
  schemas.py               # Pydantic models: GateResult, Ticket, StudentProfile
graph/                     # thin nodes + edges + wiring
  nodes.py                 # thin nodes (read state → call controller → write state)
  edges.py                 # conditional edges (read verdict → point)
  builder.py               # graph assembly + checkpointer + compile
data/                      # database access ONLY
  vector_store.py          # pgvector access
  mongo.py                 # Mongo ticket + profile access
  checkpointer.py          # Postgres checkpointer setup
ingest/
  ingest.py                # handbook chunking + embedding
docs/
  CS_2023.md
  IS_2023.md
  DESIGN_NOTES.md          # full design rationale for all of the above
pyproject.toml             # project metadata + dependencies (uv)
uv.lock
.env                       # API keys (not committed)
```

**Dependency direction:** `api/` and `graph/` call `controllers/`; `controllers/` call `data/`. Never the reverse.

---

## How It Works

### 1. Ingestion (`ingest/ingest.py`)

`CS_2023.md` and `IS_2023.md` are split by section headers, embedded with `multilingual-e5-large`, and upserted into pgvector with `source` and `section` metadata for filtered, traceable retrieval. Ingestion is idempotent — re-running after a handbook edit replaces the affected chunks (delete-then-insert keyed on `source` + `section`) rather than duplicating them.

### 2. CRAG + Escalation Loop (`graph/`)

**retrieve_node** — embeds the query, filters pgvector results by the student's department if known.

**context_relevance_gate** — an LLM judge checks whether the retrieved chunks actually address the question. If not, escalate immediately (skip generation).

**generate_response** — combines the retrieved chunks with the student profile to produce an answer.

**faithfulness_+_relevance_gate** — an LLM judge checks that the answer is grounded in the retrieved chunks AND actually addresses the question. If either fails, escalate.

**escalate_node** — writes a structured ticket to Mongo (student context, question, what was retrieved, which gate tripped and why) with `status: pending`, saves the `thread_id`, and **ends the run**.

### 3. Escalation Resolution (`controllers/escalation.py` + `api/routers/escalation.py`)

An advisor reviews pending tickets in a dashboard and answers. Resolution is a **separate short run** keyed to the ticket's `thread_id`: it loads the conversation state from the checkpointer, appends the advisor's message, optionally promotes the answer to the knowledge base (human-gated), and marks the ticket `resolved`. The student sees the answer when they next return.

### 4. Memory (`controllers/memory.py`)

**load_memory** — loads the student's single-document profile at the start of each session.

**save_memory** — Trustcall patches the profile with any new, clearly-stated identity facts, using a small model, gated so it doesn't run on every turn, and off the critical path so it never slows the student's answer.

### 5. The Learning Loop (safe by design)

When an advisor resolves a question, delivering it to the student and promoting it to the knowledge base are **two separate acts**. A generalizability check pre-fills an "Add to knowledge base" checkbox; the advisor confirms. Promoted answers are additive only — the handbook always outranks them at retrieval, and any answer that contradicts the handbook is held and flagged for handbook review rather than added as a competing chunk.

---

## Running Locally

```bash
git clone <repo-url>
cd Customer_Support

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Add GROQ_API_KEY, LANGSMITH_API_KEY, POSTGRES_URL, MONGO_URL

# Run the FastAPI backend
uv run uvicorn api.main:app --reload

# Run the Gradio UI (separate terminal)
uv run app.py
```

---

## Skills Demonstrated

- Corrective RAG with orthogonal, single-purpose judge gates (hand-written, not framework-dependent)
- Confidence-based escalation and an async human-in-the-loop resolution flow
- Durable ticket lifecycle reconnected across separate runs via `thread_id`
- Safe, human-gated self-improving knowledge base (resolved tickets → vector store)
- Source-of-truth discipline across two databases with idempotent, rebuildable sync
- Long-term memory with LangGraph Store + Trustcall (single patched profile, right-sized model)
- pgvector search with metadata filtering and handbook precedence
- Clean controller-based architecture — thin nodes, reusable logic, testable in isolation
- FastAPI service layer with routers and Pydantic v2 validation
- Modern Python tooling — uv

---

## Roadmap

- [ ] Phase 1 — Ingest handbooks into pgvector; build retrieve → generate pipeline
- [ ] Phase 2 — Add the two judge gates and the escalation router
- [ ] Phase 3 — Build the ticket lifecycle, advisor resolution flow, and `thread_id` reconnection
- [ ] Phase 4 — Add long-term student memory (single patched profile, gated extraction)
- [ ] Phase 5 — Add the human-gated learning loop (promotion checkbox, handbook precedence, contradiction flagging)
- [ ] Phase 6 — Wrap in FastAPI with routers; build Gradio UI; deploy; write final documentation
- [ ] Later — offline evaluation (Hit Rate / MRR / threshold tuning); stale-ticket scanner; duplicate detection; chunk expiry / re-review

---

## License

MIT

---

*Part of an AI Engineering portfolio. Other projects: [Mizan](https://github.com/OmarAmir2001/mizan), Research & Report Generator, AI Code Reviewer.*
