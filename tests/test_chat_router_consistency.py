"""Two reported defects, both of the same kind: a value derived once and recorded twice.

F1 — the Prometheus counter and the Langfuse trace were labelled with
`payload.department`, the department the system had just *rejected*, while retrieval
used the effective one. A returning student with a stored profile and no department in
the body was answered from their real handbook and counted under "none" — which is
most traffic once memory has learned a few profiles.

F2 — `POST /chat` decided "escalated" from the gate verdict and
`GET /chat/{thread_id}` from the presence of a ticket id. They agree until a path sets
`escalate` without writing a ticket.
"""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from customer_support.routers.chat import ask_router, chat_router, was_escalated
from customer_support.stores.llm.templates import TemplateParser

# ------------------------------------------------------------------------ F2


def test_escalated_reads_the_gate_verdict_not_the_ticket_id():
    """The ticket id is the artefact; `escalate` is the decision.

    This is the case the two definitions disagreed on: the gates failed, so the turn
    WAS escalated, but create_ticket raised before an id was written.
    """
    assert was_escalated({"escalate": True, "ticket_id": None}) is True
    assert was_escalated({"escalate": False, "ticket_id": None}) is False


def test_a_ticket_id_alone_does_not_make_a_turn_escalated():
    """The inverse: state carrying a stale id from an earlier turn of the thread.

    The checkpointer MERGES each run into the thread's state, so `ticket_id` survives
    into later runs. Reading it as the escalation signal reports a previous turn's
    outcome as this one's.
    """
    assert was_escalated({"escalate": False, "ticket_id": 42}) is False


@pytest.mark.parametrize("state", [None, {}, {"answer": "x"}])
def test_missing_state_is_not_escalated_rather_than_an_error(state):
    """GET runs against a persisted snapshot that may predate the field."""
    assert was_escalated(state) is False


# ------------------------------------------------------------------------ F1


class _Recorder:
    """Captures what the router recorded, so the test asserts on the label itself."""

    def __init__(self):
        self.questions = []
        self.traces = []


@pytest.fixture
def harness(monkeypatch):
    recorder = _Recorder()

    import customer_support.routers.chat as chat_module

    monkeypatch.setattr(
        chat_module,
        "record_question",
        lambda outcome, failed_gate, department: recorder.questions.append(
            {"outcome": outcome, "failed_gate": failed_gate, "department": department}
        ),
    )
    monkeypatch.setattr(
        chat_module,
        "start_trace",
        lambda **kwargs: recorder.traces.append(kwargs) or None,
    )

    def _build(stored_department):
        class _Graph:
            async def ainvoke(self, state, config=None):
                # Echo back what the graph was actually given, so the test can also
                # confirm retrieval received the same value that got recorded.
                recorder.graph_department = state.get("department")
                return {"answer": "ok", "escalate": False, "retrieved_chunks": []}

        class _Memory:
            async def load_profile(self, _student_id):
                return SimpleNamespace(preferred_language=None, department=stored_department)

        app = FastAPI()
        app.include_router(chat_router)
        app.include_router(ask_router)
        app.state.graph = _Graph()
        app.state.memory = _Memory()
        app.state.templates = TemplateParser(primary_language="en", default_language="en")
        app.state.settings = SimpleNamespace(PII_REDACTION_ENABLED=False, MEMORY_ENABLED=False)
        return TestClient(app), recorder

    return _build


def test_the_metric_records_the_department_actually_used(harness):
    """The reported bug, exactly: profile says CS, body claims IS.

    The answer comes from CS. The counter must say CS, not IS — otherwise
    customer_support_questions_total{department=...} does not describe which handbook
    was used, which is the only reason the label exists.
    """
    client, recorder = harness(stored_department="CS")

    client.post(
        "/api/v1/chat",
        json={"question": "how many hours?", "student_id": "s1", "department": "IS"},
    )

    assert recorder.graph_department == "CS", "retrieval should use the stored department"
    assert recorder.questions[0]["department"] == "CS", (
        "the counter recorded the REJECTED department"
    )


def test_a_returning_student_with_no_claimed_department_is_not_counted_as_none(harness):
    """The common case, and the one that quietly skews the whole series.

    Memory has learned the student is CS; the client sends no department at all. The
    old code recorded `None`.
    """
    client, recorder = harness(stored_department="CS")

    client.post("/api/v1/chat", json={"question": "how many hours?", "student_id": "s1"})

    assert recorder.graph_department == "CS"
    assert recorder.questions[0]["department"] == "CS"


def test_the_body_is_still_the_fallback_for_an_unknown_student(harness):
    """Precedence must not invert: with no stored profile, the body is all there is."""
    client, recorder = harness(stored_department=None)

    client.post(
        "/api/v1/chat",
        json={"question": "how many hours?", "student_id": "new", "department": "IS"},
    )

    assert recorder.graph_department == "IS"
    assert recorder.questions[0]["department"] == "IS"


def test_the_trace_records_both_the_effective_and_the_claimed_department(harness):
    """Trace metadata is free-form, so the discrepancy is worth keeping there.

    It must NOT reach the metric, where a second department dimension would multiply
    the time series — the cardinality rule this project has already been bitten by.
    """
    client, recorder = harness(stored_department="CS")

    client.post(
        "/api/v1/chat",
        json={"question": "how many hours?", "student_id": "s1", "department": "IS"},
    )

    trace = recorder.traces[0]
    assert trace["department"] == "CS"
    assert trace["claimed_department"] == "IS"
    assert "claimed_department" not in recorder.questions[0]


def test_no_claimed_department_is_recorded_when_the_two_agree(harness):
    """Noise suppression: only record the discrepancy when there is one."""
    client, recorder = harness(stored_department="CS")

    client.post(
        "/api/v1/chat",
        json={"question": "how many hours?", "student_id": "s1", "department": "CS"},
    )

    assert recorder.traces[0]["claimed_department"] is None


def test_the_GET_endpoint_uses_the_shared_definition_not_the_ticket_id():
    """Asserted through the endpoint, not the helper.

    Testing `was_escalated` alone is not enough: someone can revert the call site to
    `bool(ticket_id)` and every unit test on the helper stays green. That is precisely
    the drift F2 reported, so the guard has to run the route.

    The state here is the disagreement case — gates failed, no ticket written.
    """
    from types import SimpleNamespace

    class _Graph:
        async def aget_state(self, _config):
            return SimpleNamespace(
                values={
                    "escalate": True,
                    "ticket_id": None,
                    "question": "q",
                    "answer": "holding message",
                    "messages": [],
                }
            )

    app = FastAPI()
    app.include_router(chat_router)
    app.state.graph = _Graph()

    body = TestClient(app).get("/api/v1/chat/some-thread").json()

    assert body["escalated"] is True, (
        "GET re-derived escalation from the ticket id and disagreed with POST"
    )
    assert body["ticket_id"] is None
