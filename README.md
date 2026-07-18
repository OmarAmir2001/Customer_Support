# 🎧 Handbook Assistant — Higher Institute Customer Support Agent

> An AI agent that answers student questions from the CS and IS department handbooks, escalates uncertain or sensitive cases to a human advisor, and gets smarter over time from resolved escalations.

**Status:** 🟡 Early scaffolding — file ingestion & chunking pipeline is functional; the agent (RAG, grading, escalation, memory) is not yet implemented. See [Current Implementation Status](#current-implementation-status) below.

---

## What is this?

This agent will answer questions about the Higher Institute for Computer Science and Information Systems' CS and IS department handbooks. Unlike a static FAQ bot, the plan is for it to use a **Corrective RAG (CRAG) loop with confidence-based escalation** — when the agent isn't confident enough in its answer, it pauses and routes the question to a human academic advisor instead of guessing.

It's also meant to remember each student across sessions (name, department, past questions) and **learn from every escalation it resolves** — once an advisor answers a question the agent couldn't, that Q&A pair gets added back into the knowledge base so the same question is auto-resolved next time.

This is the second project in a 4-part AI engineering portfolio, building directly on patterns established in [Mizan](https://github.com/OmarAmir2001/mizan) (CRAG, long-term memory, Trustcall) while adding new skills: confidence-scored escalation, human-in-the-loop interrupts, vector search, and a FastAPI service layer.

---

## Current Implementation Status

What's actually built today, vs. what's still on the roadmap:

| Piece | Status |
|---|---|
| FastAPI service with routers (`admin`, `chat`, `escalation`, `history`, `profile`, health check) | ✅ Implemented |
| MongoDB persistence (`motor`) for projects and document chunks | ✅ Implemented |
| File upload / ingestion endpoint (`POST /api/v1/admin/ingest/{project_id}`) — validates type/size, saves to disk per project | ✅ Implemented |
| Chunking pipeline (`POST /api/v1/admin/proccess/{project_id}`) — loads `.txt`/`.md`/`.pdf`, splits with LangChain's `RecursiveCharacterTextSplitter`, stores chunks in MongoDB | ✅ Implemented |
| `/chat`, `/chat/stream`, `/escalation/*`, `/history/*`, `/profile/*`, `/admin/knowledge_base/stats` | 🟡 Placeholder — routes exist and return mock data, no LLM or logic behind them yet |
| Vector search (Qdrant), embeddings | ⬜ Not implemented — chunks are currently stored in MongoDB, not embedded |
| CRAG graph (LangGraph), confidence-based routing, HITL escalation | ⬜ Not implemented |
| Long-term student memory (LangGraph Store + Trustcall) | ⬜ Not implemented |
| Self-learning loop (advisor answers → re-embedded) | ⬜ Not implemented |
| Gradio UI | ⬜ Not implemented |

The sections below (Planned Architecture, Tech Stack, etc.) describe the target design this project is being built toward.

---

## Planned Architecture

```
Student Question
    │
    ▼
load_memory            ← reads student profile + question history from LangGraph Store
    │
    ▼
retrieve_node          ← embeds query, searches Qdrant (filtered by department: CS/IS)
    │
    ▼
grader                 ← LLM scores chunk relevance + overall confidence
    │
    ├── low confidence ──► escalate_node ──► [HITL interrupt] ──► academic advisor review
    │                                                                    │
    │                                                          resolved answer ──► Qdrant
    │                                                          (learning loop)
    │
    └── high confidence ──► generate_response ← uses student profile + instructions
                                  │
                                  ▼
                            save_memory   ← Trustcall updates student profile
                                  │
                                  ▼
                                END
```

---

## Key Features (Planned)

- [ ] **Corrective RAG (CRAG)** with confidence scoring per retrieved chunk
- [ ] **Department-aware retrieval** — filters Qdrant results to CS or IS handbook based on student profile
- [ ] **Human-in-the-loop escalation** — graph pauses via `interrupt_before` when confidence is low, waits for advisor review
- [ ] **Structured escalation summaries** — advisor sees student context, the question, what was found, and why it escalated
- [ ] **Long-term student memory** — name, student ID, department, GPA, past questions, preferred language (LangGraph Store + Trustcall)
- [ ] **Self-learning knowledge base** — resolved escalations are embedded and added back to Qdrant
- [ ] **FastAPI service layer** — `/chat`, `/history/{student_id}`, `/health` endpoints with routers and Pydantic validation
- [ ] **Gradio UI** — chat interface with department selector and escalation status indicator
- [ ] **Streaming responses**

---

## Tech Stack

| Component | Technology | Status |
|---|---|---|
| API Layer | FastAPI | ✅ in use |
| Document store | MongoDB (`motor` async driver) | ✅ in use |
| Document loading & chunking | LangChain (`langchain-community` loaders + `RecursiveCharacterTextSplitter`) | ✅ in use |
| Package management | uv | ✅ in use |
| Agent Framework | LangGraph | ⬜ planned |
| LLM | Groq — llama-3.3-70b-versatile | ⬜ planned (key is configured, not yet called) |
| Embeddings | intfloat/multilingual-e5-large | ⬜ planned |
| Vector Store | Qdrant | ⬜ planned |
| Memory (short-term) | LangGraph MemorySaver | ⬜ planned |
| Memory (long-term) | LangGraph Store + Trustcall | ⬜ planned |
| UI | Gradio | ⬜ planned |
| Deployment | HuggingFace Spaces | ⬜ planned |

---

## Knowledge Base

The agent's knowledge base is built from the Higher Institute's official department handbooks:

- `CS_2023.md` — Computer Science department handbook
- `IS_2023.md` — Information Systems department handbook

Today, handbook files are uploaded per-project via `POST /api/v1/admin/ingest/{project_id}`, saved under `src/assets/files/{project_id}/`, then chunked via `POST /api/v1/admin/proccess/{project_id}` (recursive character splitting, configurable `chunk_size`/`overlap`) and stored as `chunks` documents in MongoDB with order and source metadata.

Eventually each chunk will be embedded and stored in Qdrant with metadata (`source`, `section`) so retrieved answers can be traced back to the exact handbook section — that step isn't wired up yet.

---

## Project Structure

```
Customer_Support/
├── docker/
│   └── docker-compose.yml    # MongoDB service
├── src/
│   ├── main.py                # FastAPI app entry point, registers routers, MongoDB lifespan
│   ├── .env                   # Environment variables (see Configuration below)
│   ├── pyproject.toml         # Project metadata + dependencies (uv)
│   ├── uv.lock
│   ├── assets/files/          # Uploaded project documents (created at runtime, per project_id)
│   ├── controllers/           # BaseController, DataController, ProjectController, ProccessController
│   ├── helpers/
│   │   └── config.py           # Pydantic Settings loaded from .env
│   ├── models/
│   │   ├── BaseDataModel.py
│   │   ├── ProjectModel.py     # Mongo access for the "projects" collection
│   │   ├── ChunkModel.py       # Mongo access for the "chunks" collection
│   │   ├── db_schemas/         # Pydantic schemas: Project, DataChunk
│   │   └── enums/              # DatabaseEnum, ProcessingEnum, ResponseEnum
│   └── routers/
│       ├── health.py           # GET /
│       ├── admin.py            # /api/v1/admin — ingest, process, knowledge_base/stats
│       ├── chat.py             # /api/v1/chat — placeholder
│       ├── escalation.py       # /api/v1/escalation — placeholder
│       ├── history.py          # /api/v1/history — placeholder
│       ├── profile.py          # /api/v1/profile — placeholder
│       └── schemas/            # Request/response models (ProcessRequest, etc.)
```

---

## API Endpoints

| Method | Path | Status |
|---|---|---|
| GET | `/` | ✅ health check — returns app name/version |
| POST | `/api/v1/admin/ingest/{project_id}` | ✅ upload a handbook file (`.txt`/`.md`/`.pdf`) into a project |
| POST | `/api/v1/admin/proccess/{project_id}` | ✅ chunk an ingested file and store chunks in MongoDB |
| GET | `/api/v1/admin/knowledge_base/stats` | 🟡 placeholder — returns hardcoded stats |
| POST | `/api/v1/chat/chat` | 🟡 placeholder — echoes input, no agent behind it |
| POST | `/api/v1/chat/chat/stream` | 🟡 placeholder — fake streamed response |
| GET | `/api/v1/escalation/escalations` | 🟡 placeholder — hardcoded list |
| GET | `/api/v1/escalation/escalation/{escalation_id}` | 🟡 placeholder |
| POST | `/api/v1/escalation/escalation/{escalation_id}/resolve` | 🟡 placeholder |
| GET / DELETE | `/api/v1/history/history/{user_id}` | 🟡 placeholder |
| GET / DELETE | `/api/v1/profile/{user_id}/profile` | 🟡 placeholder |

Interactive docs are available at `/docs` (Swagger UI) once the server is running.

---

## How It Will Work (Planned)

### 1. Ingestion
Handbook files are split by section, embedded with `multilingual-e5-large`, and upserted into a Qdrant collection with `department` and `section` metadata for filtered retrieval. *(Currently: files are chunked and stored in MongoDB, without embedding.)*

### 2. CRAG + Escalation Loop

**retrieve_node** — embeds the query, filters Qdrant results by the student's department if known.

**grader** — scores each chunk's relevance and computes an overall confidence score (not just yes/no).

**route_after_grading** — if confidence ≥ threshold, generate an answer. If confidence is low, escalate.

**escalate_node** — builds a structured summary (student context, question, what was found, why it's escalating) for the academic advisor.

**HITL interrupt** — the graph pauses before escalating so an advisor can review and either approve the escalation or answer directly.

**generate_response** — combines retrieved chunks with student profile and behavioral instructions to produce the final answer.

### 3. Memory

**load_memory** — loads the student's profile (name, department, past questions) at the start of each session.

**save_memory** — Trustcall extracts and patches updated student facts after each interaction, without overwriting existing data.

### 4. The Learning Loop

When an advisor resolves an escalated question, that question-answer pair is embedded and added to Qdrant with `source: instructor_resolved` metadata — so the same question is auto-resolved by the agent next time it's asked.

---

## Running Locally

```bash
git clone https://github.com/OmarAmir2001/Customer_Support.git
cd Customer_Support

# Start MongoDB
cd docker
docker compose up -d
cd ..

# Install dependencies
cd src
uv sync

# Configure environment variables — edit src/.env and set at least:
#   APP_NAME, APP_VERSION, GROQ_API_KEY, MONGODB_URL, MONGODB_DATABASE,
#   File_Allowed_Types, File_Max_Size, File_Default_CHUNK_SIZE

# Run the FastAPI backend (from src/)
uv run uvicorn main:app --reload
```

The API is then available at `http://127.0.0.1:8000`, with docs at `http://127.0.0.1:8000/docs`.

---

## Skills Demonstrated

- [x] FastAPI service layer with routers and Pydantic validation
- [x] Async MongoDB access (`motor`) with schema models
- [x] File upload validation and document chunking (LangChain)
- [x] Modern Python tooling — uv
- [ ] Corrective RAG with confidence-based routing
- [ ] Human-in-the-loop (HITL) interrupts and resume flow
- [ ] Long-term memory with LangGraph Store + Trustcall
- [ ] Self-improving knowledge base (resolved tickets → vector store)
- [ ] Qdrant vector search with metadata filtering
- [ ] Gradio UI with department-aware context

---

## Roadmap

- [x] Phase 1a — FastAPI service scaffolding, MongoDB persistence, router layout
- [x] Phase 1b — Handbook file ingestion + chunking pipeline (MongoDB-backed)
- [ ] Phase 1c — Embed chunks and move storage to Qdrant, build basic retrieve/grade/generate pipeline
- [ ] Phase 2 — Add confidence scoring, escalation router, HITL interrupt
- [ ] Phase 3 — Add long-term student memory
- [ ] Phase 4 — Wire `/chat`, `/escalation`, `/history`, `/profile` routes to real agent logic
- [ ] Phase 5 — Build Gradio UI, deploy, write final documentation

---

## License

MIT

---

*Part of an AI Engineering portfolio. Other projects: [Mizan](https://github.com/OmarAmir2001/mizan), Research & Report Generator, AI Code Reviewer.*