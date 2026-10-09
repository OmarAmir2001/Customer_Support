"""Langfuse tracing — one trace per question, one span per graph node.

Deliberately thin. Three rules shaped it:

**Tracing must never break a request.** Every function here swallows its own errors.
An observability backend being down, misconfigured or slow is not a reason a student
fails to get an answer — and a tracing layer that can take the app down is worse than
no tracing at all.

**It must be optional.** With `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` unset,
every call here is a no-op and the app behaves exactly as before. That keeps the test
suite free of a service dependency and lets anyone run the project without standing
Langfuse up.

**The trace id is the `thread_id`.** The same value already used as the structlog
correlation id, so one identifier spans the student's question, the gate verdicts,
the ticket, the advisor's resolution days later — and now the trace. Without that,
correlating a trace against the logs means guessing from timestamps.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from customer_support.helpers.logging_config import get_logger

logger = get_logger(__name__)

_client: Any = None
_enabled = False


#: The trace for the request currently being served.
#:
#: A contextvar, not a parameter threaded through the graph, and not a field on the
#: graph state. State is serialised into the checkpointer, so a trace object there
#: would either fail to pickle or be persisted as junk. A contextvar also mirrors how
#: the correlation id already flows, so the two travel the same way.
_current_trace: ContextVar[Any] = ContextVar("current_trace", default=None)


def set_current_trace(trace: Any) -> None:
    _current_trace.set(trace)


def current_trace() -> Any:
    return _current_trace.get()


def configure_tracing(settings) -> bool:
    """Build the client once at startup. Returns whether tracing is on.

    Called from the lifespan rather than lazily per request: a misconfiguration
    should be visible in the boot logs, not discovered one request at a time.
    """
    global _client, _enabled

    public_key = getattr(settings, "LANGFUSE_PUBLIC_KEY", None)
    secret_key = getattr(settings, "LANGFUSE_SECRET_KEY", None)
    if not public_key or not secret_key:
        logger.info("tracing_disabled", reason="LANGFUSE_PUBLIC_KEY/SECRET_KEY not set")
        _enabled = False
        return False

    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=getattr(settings, "LANGFUSE_HOST", None) or "http://localhost:3002",
        )
        _enabled = True
        logger.info("tracing_enabled", host=getattr(settings, "LANGFUSE_HOST", None))
    except Exception as exc:
        # Import or construction failure is not fatal. The SDK is a dev dependency,
        # so a production image built without it should still serve.
        logger.warning("tracing_unavailable", error=str(exc))
        _enabled = False

    return _enabled


def shutdown_tracing() -> None:
    """Flush buffered events on the way out.

    The SDK batches in a background thread, so without this the last few traces of a
    run are simply lost when the process exits — most noticeably the ones you were
    looking for after a manual test.
    """
    if _enabled and _client is not None:
        try:
            _client.flush()
        except Exception as exc:
            logger.warning("tracing_flush_failed", error=str(exc))


def start_trace(*, thread_id: str, question: str, student_id: str, **metadata) -> Any:
    """Open a trace for one question, or return None when tracing is off."""
    if not _enabled or _client is None:
        return None
    try:
        return _client.trace(
            # The thread_id, so a trace and its log lines share one identifier.
            id=thread_id,
            name="chat",
            input={"question": question},
            user_id=student_id,
            session_id=thread_id,
            metadata=metadata or None,
        )
    except Exception as exc:
        logger.warning("tracing_start_failed", error=str(exc))
        return None


@contextmanager
def span(trace: Any, name: str, **inputs):
    """One graph node. A no-op when `trace` is None.

    Yields a handle whose `.update(...)` records the node's output. Exceptions are
    recorded on the span and then RE-RAISED — unlike everything else here, a node
    failing is the app's business, and swallowing it would hide a real error behind
    a tracing concern.
    """
    if trace is None:
        yield _NullSpan()
        return

    handle = None
    try:
        handle = trace.span(name=name, input=inputs or None)
    except Exception as exc:
        logger.warning("tracing_span_failed", span=name, error=str(exc))
        yield _NullSpan()
        return

    wrapper = _Span(handle)
    try:
        yield wrapper
    except Exception as exc:
        wrapper.update(level="ERROR", status_message=str(exc))
        wrapper.end()
        raise
    else:
        wrapper.end()


def traced(name: str):
    """Decorate an async graph node so it becomes one span.

    A decorator rather than a `with` block inside each node, because wrapping five
    existing function bodies would mean re-indenting all of them — a large, risky
    diff for a cross-cutting concern that should barely touch the code it observes.

    The node's return value is recorded as the span's output, which for this graph is
    exactly the right thing: every node returns the slice of state it computed.
    """

    def decorate(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            with span(current_trace(), name) as handle:
                result = await func(*args, **kwargs)
                if isinstance(result, dict):
                    # Keys only for anything bulky: retrieved_chunks is five
                    # documents of handbook text, and putting that in every trace
                    # would bury the gate scores that are the reason to look.
                    handle.update(
                        output={
                            key: (f"<{len(value)} items>" if isinstance(value, list) else value)
                            for key, value in result.items()
                        }
                    )
                return result

        return wrapper

    return decorate


class _NullSpan:
    """What callers get when tracing is off, so no call site needs an `if`."""

    def update(self, **_kwargs) -> None: ...
    def end(self) -> None: ...


class _Span:
    def __init__(self, handle: Any):
        self._handle = handle
        self._ended = False

    def update(self, **kwargs) -> None:
        try:
            self._handle.update(**kwargs)
        except Exception:
            # Silent: a failed span update must not become a second error on top of
            # whatever the caller was already doing.
            pass

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        try:
            self._handle.end()
        except Exception:
            pass


def finish_trace(trace: Any, *, answer: str | None, escalated: bool, **metadata) -> None:
    """Record the outcome. Safe to call with None."""
    if trace is None:
        return
    try:
        trace.update(
            output={"answer": answer, "escalated": escalated},
            metadata=metadata or None,
        )
    except Exception as exc:
        logger.warning("tracing_finish_failed", error=str(exc))
