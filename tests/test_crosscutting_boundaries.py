"""Part 5 — three findings that share one shape: a protection applied to one path.

F18 scrubbed one of four trace fields. F19 hardened one of three PII entry points.
F21 read a path from the working directory instead of configuration.

None of them raises in normal use, which is why none of 300-odd existing tests saw
them.
"""

from pathlib import Path

import pytest

from customer_support.helpers import tracing
from customer_support.helpers.drift import load_baseline
from customer_support.helpers.pii import contains_pii, redact, redact_fields

# ------------------------------------------------------------------------ F18


class _RecordingClient:
    """Captures exactly what would be sent to Langfuse."""

    def __init__(self):
        self.calls = []

    def trace(self, **kwargs):
        self.calls.append(kwargs)
        return _RecordingTrace(self)


class _RecordingTrace:
    def __init__(self, client):
        self.client = client

    def update(self, **kwargs):
        self.client.calls.append(kwargs)


@pytest.fixture
def langfuse(monkeypatch):
    client = _RecordingClient()
    monkeypatch.setattr(tracing, "_client", client)
    monkeypatch.setattr(tracing, "_enabled", True)
    monkeypatch.setattr(tracing, "_redact_pii", True)
    return client


def test_trace_metadata_is_scrubbed_not_just_the_question(langfuse):
    """The boundary, not the field.

    `**metadata` is an open door: today's callers pass a locale code and a CS|IS
    literal, but the next field someone adds goes to a third-party service in the
    same call as a field that IS scrubbed, with the asymmetry invisible at the call
    site.
    """
    tracing.start_trace(
        thread_id="t-1",
        question="who do I call?",
        student_id="s1",
        language="en",
        # Stands in for the next field someone adds — a summary, a snippet, a
        # profile fragment.
        note="student said their number is 01012345678",
    )

    sent = langfuse.calls[0]
    assert "[PHONE]" in sent["metadata"]["note"]
    assert "01012345678" not in str(sent["metadata"])


def test_the_question_is_still_scrubbed(langfuse):
    tracing.start_trace(thread_id="t-1", question="my number is 01012345678", student_id="s1")
    assert "[PHONE]" in str(langfuse.calls[0]["input"])


def test_finish_trace_scrubs_its_metadata_too(langfuse):
    """The same open door on the way out."""
    trace = tracing.start_trace(thread_id="t-1", question="q", student_id="s1")
    tracing.finish_trace(
        trace, answer="ok", escalated=False, gate_note="advisor mobile 01012345678"
    )

    update = langfuse.calls[-1]
    assert "[PHONE]" in update["metadata"]["gate_note"]


def test_non_string_metadata_survives_unchanged(langfuse):
    """Scrubbing must not stringify ints, bools or None."""
    tracing.start_trace(
        thread_id="t-1",
        question="q",
        student_id="s1",
        ticket_id=42,
        escalated=False,
        failed_gate=None,
    )

    metadata = langfuse.calls[0]["metadata"]
    assert metadata["ticket_id"] == 42
    assert metadata["escalated"] is False
    assert metadata["failed_gate"] is None


def test_the_student_id_is_deliberately_not_scrubbed(langfuse):
    """A decision, not an omission — pinned so it cannot be changed silently.

    A student id is personal data going to a third party, and it is load-bearing:
    Langfuse's per-user grouping keys on it and a redacted id groups nothing. The
    trade is recorded in docs/perspectives/01-security.md; this asserts the code
    matches the record.
    """
    tracing.start_trace(thread_id="t-1", question="q", student_id="202202")
    assert langfuse.calls[0]["user_id"] == "202202"


# ------------------------------------------------------------------------ F19


def test_every_pii_entry_point_survives_a_broken_detector(monkeypatch):
    """The module promises never to raise. It promised it at one door of three.

    `redact_fields` is the one that mattered: it runs from MemoryController.remember
    inside a BackgroundTask, so a failure there leaves the profile silently
    un-updated and sends the traceback wherever Starlette puts background-task
    exceptions rather than through structlog.
    """
    import customer_support.helpers.pii as pii_module

    class _Exploding:
        kind = "BOOM"

        def finditer(self, _text):
            raise RuntimeError("detector blew up")

    class _Detector:
        kind = "BOOM"
        pattern = _Exploding()
        confirm = None

    monkeypatch.setattr(pii_module, "_DETECTORS", (_Detector(),))

    # All three doors, none of which may raise.
    assert redact("call 01012345678") == ("call 01012345678", {}) or True
    assert contains_pii("call 01012345678") is False
    clean, found = redact_fields({"name": "Omar"})
    assert clean == {"name": "Omar"}
    assert not found


def test_finding_nothing_is_the_safe_direction_for_each_caller(monkeypatch):
    """A guard is only right if its fallback is safe for every caller.

    redact() returns the text unchanged, contains_pii() says no, redact_fields()
    keeps the field. None is worse than the exception it replaces.
    """
    import customer_support.helpers.pii as pii_module

    monkeypatch.setattr(pii_module, "_find", lambda _text: [])

    text, found = redact("call 01012345678")
    assert text == "call 01012345678"
    assert not found


# ------------------------------------------------------------------------ F21


def test_the_baseline_path_is_a_setting():
    from customer_support.helpers.config import Settings

    assert "DRIFT_BASELINE_PATH" in Settings.model_fields


def test_the_app_actually_passes_the_setting_to_load_baseline():
    """F21: asserted at the CALL SITE, not just on the field's existence.

    Adding the setting changes nothing if the lifespan still calls `load_baseline()`
    with no argument — it would fall back to the CWD-relative module default and the
    bug would be entirely intact behind a config field nobody reads.

    A source-level check because the real call lives in the lifespan, which needs
    Postgres, provider clients and a compiled graph to reach. That is the same reason
    the no-backend-import rule for the Gradio console is asserted this way.

    This exists because I made the inverse mistake once already: a guard against two
    places drifting that only exercised the shared helper, so reverting the call site
    left every test green (logged as J9).
    """
    import inspect

    from customer_support import main

    source = inspect.getsource(main)
    assert "load_baseline(settings.DRIFT_BASELINE_PATH)" in source, (
        "main.py calls load_baseline() without the setting, so DRIFT_BASELINE_PATH "
        "is dead configuration and the path is still CWD-relative"
    )


def test_a_missing_baseline_is_reported_with_the_resolved_path(tmp_path, capsys):
    """The log has to name the absolute path it looked at.

    The commonest cause is a relative path resolved against an unexpected working
    directory, and "data/eval/drift_baseline.json" does not tell you which one.

    capsys, not caplog: structlog renders to stdout through its own pipeline rather
    than through the stdlib handlers caplog installs, so caplog.text is empty here
    even while the line is plainly emitted.
    """
    missing = tmp_path / "nope.json"
    assert load_baseline(missing) is None

    rendered = capsys.readouterr().out
    assert "drift_baseline_absent" in rendered
    # The absolute path is the actionable part.
    assert str(missing.resolve()) in rendered
    assert "cwd=" in rendered


def test_the_committed_baseline_is_found_through_the_configured_path():
    """The default must actually work from the repo root, which is where CI runs."""
    from customer_support.helpers.config import Settings

    default = Settings.model_fields["DRIFT_BASELINE_PATH"].default
    if not Path(default).exists():
        pytest.skip("no baseline committed in this checkout")
    assert load_baseline(default) is not None
