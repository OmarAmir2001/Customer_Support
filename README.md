# 🎧 Handbook Assistant — Higher Institute Customer Support Agent

> An AI agent that answers student questions from the CS and IS department handbooks, escalates uncertain or sensitive cases to a human advisor, and gets smarter over time from resolved escalations.

**Status:** 🟡 In Progress

---

## What is this?

This agent answers questions about the Higher Institute for Computer Science and Information Systems' CS and IS department handbooks. Unlike a static FAQ bot, it uses a **Corrective RAG (CRAG) loop with confidence-based escalation** — when the agent isn't confident enough in its answer, it pauses and routes the question to a human academic advisor instead of guessing.

It also remembers each student across sessions (name, department, past questions) and **learns from every escalation it resolves** — once an advisor answers a question the agent couldn't, that Q&A pair is added back into the knowledge base so the same question gets auto-resolved next time.

This is the second project in a 4-part AI engineering portfolio, building directly on patterns established in [Mizan](https://github.com/OmarAmir2001/mizan) (CRAG, long-term memory, Trustcall) while adding new skills: confidence-scored escalation, human-in-the-loop interrupts, Qdrant vector search, and a FastAPI service layer.

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

| Component | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | Groq — llama-3.3-70b-versatile |
| Embeddings | intfloat/multilingual-e5-large |
| Vector Store | Qdrant |
| Memory (short-term) | LangGraph MemorySaver |
| Memory (long-term) | LangGraph Store + Trustcall |
| API Layer | FastAPI |
| UI | Gradio |
| Package Management | uv |
| Deployment | HuggingFace Spaces |

---

## Knowledge Base

The agent's knowledge base is built from the Higher Institute's official department handbooks:

- `CS_2023.md` — Computer Science department handbook
- `IS_2023.md` — Information Systems department handbook

Each file is split by section headers (`##`) into chunks, embedded, and stored in Qdrant with metadata (`source`, `section`) so retrieved answers can be traced back to the exact handbook section.

---

## Project Structure (Planned)

```
customer-support-agent/
├── app.py                   # Gradio UI — entry point
├── api/
│   ├── main.py               # FastAPI app
│   └── routers/
│       └── chat.py           # /chat, /history, /health endpoints
├── model/
│   ├── graph.py               # LangGraph graph — nodes, edges, state, compilation
│   ├── memory.py               # Long-term memory — StudentProfile, Trustcall extractors
│   └── ingest.py                # Handbook ingestion — chunking, embedding, Qdrant storage
├── docs/
│   ├── CS_2023.md
│   └── IS_2023.md
├── langgraph.json             # LangGraph deployment config
├── pyproject.toml             # Project metadata + dependencies (uv)
├── uv.lock
└── .env                        # API keys (not committed)
```

---

## How It Works

### 1. Ingestion (`ingest.py`)
`CS_2023.md` and `IS_2023.md` are split by section headers, embedded with `multilingual-e5-large`, and upserted into a Qdrant collection with `department` and `section` metadata for filtered retrieval.

### 2. CRAG + Escalation Loop (`graph.py`)

**retrieve_node** — embeds the query, filters Qdrant results by the student's department if known.

**grader** — scores each chunk's relevance and computes an overall confidence score (not just yes/no).

**route_after_grading** — if confidence ≥ threshold, generate an answer. If confidence is low, escalate.

**escalate_node** — builds a structured summary (student context, question, what was found, why it's escalating) for the academic advisor.

**HITL interrupt** — the graph pauses before escalating so an advisor can review and either approve the escalation or answer directly.

**generate_response** — combines retrieved chunks with student profile and behavioral instructions to produce the final answer.

### 3. Memory (`memory.py`)

**load_memory** — loads the student's profile (name, department, past questions) at the start of each session.

**save_memory** — Trustcall extracts and patches updated student facts after each interaction, without overwriting existing data.

### 4. The Learning Loop

When an advisor resolves an escalated question, that question-answer pair is embedded and added to Qdrant with `source: instructor_resolved` metadata — so the same question is auto-resolved by the agent next time it's asked.

---

## Running Locally

```bash
git clone <repo-url>
cd customer-support-agent

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Add GROQ_API_KEY, LANGSMITH_API_KEY, QDRANT_URL (if using Qdrant Cloud)

# Run the FastAPI backend
uv run uvicorn api.main:app --reload

# Run the Gradio UI (separate terminal)
uv run app.py
```

---

## Skills Demonstrated

- Corrective RAG with confidence-based routing
- Human-in-the-loop (HITL) interrupts and resume flow
- Long-term memory with LangGraph Store + Trustcall
- Self-improving knowledge base (resolved tickets → vector store)
- Qdrant vector search with metadata filtering
- FastAPI service layer with routers and Pydantic validation
- Gradio UI with department-aware context
- Modern Python tooling — uv

---

## Roadmap

- [ ] Phase 1 — Ingest handbooks into Qdrant, build basic retrieve/grade/generate pipeline
- [ ] Phase 2 — Add confidence scoring, escalation router, HITL interrupt
- [ ] Phase 3 — Add long-term student memory
- [ ] Phase 4 — Wrap in FastAPI with routers
- [ ] Phase 5 — Build Gradio UI, deploy, write final documentation

---

## License

MIT

---

*Part of an AI Engineering portfolio. Other projects: [Mizan](https://github.com/OmarAmir2001/mizan), Research & Report Generator, AI Code Reviewer.*