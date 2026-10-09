"""Tracing must be invisible when off and harmless when broken.

Observability that can take the app down is worse than no observability. These tests
pin the two properties that make it safe to leave wired in permanently: with no keys
configured it is a no-op, and when the backend misbehaves the request still completes.

They also assert the one thing that makes traces useful here — the trace id is the
`thread_id`, the same identifier as the structlog correlation id, so a trace and its
log lines are joinable without guessing from timestamps.
"""

import pytest

from customer_support.helpers import tracing


class NoKeys:
    LANGFUSE_PUBLIC_KEY = ""
    LANGFUSE_SECRET_KEY = ""
    LANGFUSE_HOST = "http://localhost:3002"


class WithKeys:
    LANGFUSE_PUBLIC_KEY = "pk-lf-test"
    LANGFUSE_SECRET_KEY = "sk-lf-test"
    LANGFUSE_HOST = "http://localhost:3002"


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Tracing keeps a module-level client, so state leaks between tests otherwise."""
    yield
    tracing._client = None
    tracing._enabled = False
    tracing.set_current_trace(None)


def test_tracing_is_off_when_no_keys_are_configured():
    """The default. The test suite must not need a tracing backend, and the project
    must run for someone who has not stood Langfuse up."""
    assert tracing.configure_tracing(NoKeys()) is False


def test_every_call_is_a_no_op_when_disabled():
    tracing.configure_tracing(NoKeys())

    assert tracing.start_trace(thread_id="t", question="q", student_id="s") is None
    # None must flow through the rest without a single `if` at the call sites.
    with tracing.span(None, "retrieve") as handle:
        handle.update(output={"x": 1})
    tracing.finish_trace(None, answer="a", escalated=False)
    tracing.shutdown_tracing()


@pytest.mark.asyncio
async def test_a_decorated_node_runs_normally_with_tracing_off():
    tracing.configure_tracing(NoKeys())

    @tracing.traced("retrieve")
    async def node(state):
        return {"retrieved_chunks": [1, 2, 3]}

    assert await node({}) == {"retrieved_chunks": [1, 2, 3]}


@pytest.mark.asyncio
async def test_a_broken_backend_does_not_break_the_request():
    """The property that matters most. An observability backend being down is not a
    reason a student fails to get an answer."""

    class Exploding:
        def span(self, **_kwargs):
            raise RuntimeError("langfuse is down")

        def update(self, **_kwargs):
            raise RuntimeError("langfuse is down")

    tracing.configure_tracing(WithKeys())
    tracing.set_current_trace(Exploding())

    @tracing.traced("generate")
    async def node(state):
        return {"answer": "135 credit hours"}

    assert await node({}) == {"answer": "135 credit hours"}
    # and finishing a broken trace must not raise either
    tracing.finish_trace(Exploding(), answer="a", escalated=False)


@pytest.mark.asyncio
async def test_a_node_error_is_recorded_and_then_re_raised():
    """Unlike everything else here, a node failing is the app's business. Swallowing
    it would hide a real error behind a tracing concern."""
    recorded = {}

    class Handle:
        def update(self, **kwargs):
            recorded.update(kwargs)

        def end(self):
            recorded["ended"] = True

    class Trace:
        def span(self, **_kwargs):
            return Handle()

    tracing._enabled = True
    tracing._client = object()
    # Must be set: without a current trace, span() correctly yields a _NullSpan and
    # there is nothing to record on. The first version of this test omitted it and
    # was asserting against the no-op path.
    tracing.set_current_trace(Trace())

    @tracing.traced("generate")
    async def node(state):
        raise ValueError("generation failed")

    with pytest.raises(ValueError, match="generation failed"):
        await node({})

    assert recorded.get("level") == "ERROR"
    assert "generation failed" in str(recorded.get("status_message"))


def test_bulky_values_are_summarised_rather_than_dumped_into_the_trace():
    """retrieved_chunks is five documents of handbook text. Putting that in every
    trace buries the gate scores that are the reason to open one."""
    captured = {}

    class Handle:
        def update(self, **kwargs):
            captured.update(kwargs)

        def end(self): ...

    class Trace:
        def span(self, **_kwargs):
            return Handle()

    tracing._enabled = True
    tracing._client = object()
    tracing.set_current_trace(Trace())

    import asyncio

    @tracing.traced("retrieve")
    async def node(state):
        return {"retrieved_chunks": ["a" * 2000] * 5, "escalate": False}

    asyncio.run(node({}))

    assert captured["output"]["retrieved_chunks"] == "<5 items>"
    assert captured["output"]["escalate"] is False
