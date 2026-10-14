# Gradio operator console

A showcase and operator UI for the handbook assistant. **Not the product front end** —
the student-facing interface is separate design work; this exists so the system can be
driven and demonstrated without curl.

```bash
# the API must be running first
make up

# then, in another terminal
make ui
# or: uv run --extra ui python app/gradio/main.py
```

Opens on <http://127.0.0.1:7860>.

## What each tab is for

| Tab | Shows |
|---|---|
| **Ask** | The student's view, with **verified** and **streaming** side by side — the project's central trade-off in one radio button. Renders `sources[]` as citation chips and the escalation banner with its ticket id |
| **Conversation** | A thread by id, including an advisor's reply. Demonstrates that delivery is **passive** — nothing is pushed when a ticket resolves |
| **Advisor queue** | The escalation lifecycle: filters, which gate tripped, the excerpts the bot actually saw, gate scores, and every lifecycle move. Resolving and promoting are separate buttons, because they are separate decisions |
| **System** | `documents_indexed`, the long-term profile, and links to Grafana, Prometheus, Langfuse and MLflow |

## Design decisions

**It is a separate process that talks HTTP.** It imports nothing from
`customer_support` — no database handle, no provider key, no settings object. Every
decision it renders (which gate tripped, whether promotion is held, whether a
transition is legal) was made by the API. Consequences worth having: it can point at a
remote deployment by changing one environment variable, it can be deployed on its own,
the API image never carries Gradio, and breaking this cannot break the assistant.

**There is no `__init__.py` here, deliberately.** The directory is called `gradio`. If
it were an importable package and its parent were on `sys.path`, `import gradio` inside
these modules would resolve to *this directory* rather than the library — and the
failure would surface as a baffling `AttributeError` on `gradio.Blocks`. The modules
are flat, and `main.py` puts *this* directory on `sys.path`, never its parent.

**There is no authentication.** "Acting as" in the advisor tab is a text box, not a
session. A real deployment puts an identity provider in front and takes `advisor_id`
from the token — which is exactly why the API accepts it as a parameter rather than
inferring it, and why the audit trail records whoever the caller claimed to be. Stated
here rather than left for someone to discover.

**`demo.queue()` is load-bearing.** Without it Gradio collects a generator's output
and sends it once, which silently turns the streaming tab back into the blocking one —
the same class of bug as the missing `await asyncio.sleep(0)` in the Bento service.

**Errors are sentences, not tracebacks.** `api_client.py` is the only module that
knows HTTP exists, and it translates every failure into something actionable: 404
means a missing ticket, 409 means either an illegal transition *or* that another
advisor got there first, and a connection error names the command that starts the
stack. A UI that renders `httpx.ConnectError` teaches the viewer nothing.

## Configuration

All optional; defaults assume the compose stack on localhost.

| Variable | Default | What it does |
|---|---|---|
| `UI_API_BASE_URL` | `http://127.0.0.1:8000` | the gated FastAPI app |
| `UI_STREAM_BASE_URL` | `http://127.0.0.1:3001` | the BentoML service (optional — the UI degrades rather than breaking) |
| `UI_HOST` / `UI_PORT` | `127.0.0.1` / `7860` | where this console listens |
| `UI_REQUEST_TIMEOUT` | `180` | seconds. A cold question runs retrieval plus up to three judge calls |
| `UI_GRAFANA_URL`, `UI_PROMETHEUS_URL`, `UI_LANGFUSE_URL`, `UI_MLFLOW_URL` | localhost ports | dashboard links on the System tab |
| `UI_ADVISOR_ID`, `UI_STUDENT_ID` | `advisor_demo`, `student_demo` | prefilled ids |

## Streaming needs the Bento service

The streaming path is a *different process on a different port*. Start it separately:

```bash
uv run --extra dev bentoml serve customer_support.bento_service:HandbookAssistant --port 3001
```

Without it, the streaming radio option reports that the service is unreachable and
names this command. The verified path does not need it.
