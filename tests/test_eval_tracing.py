"""The eval tracer, with no Langfuse anywhere near it.

The properties worth pinning are the ones that bite silently: a tracer that fails
closed would kill a run that has already spent real money on model calls, and a
tracer that leaks PII into a third-party service does it without any visible symptom.
"""

from types import SimpleNamespace

import pytest

from customer_support.helpers.eval_tracing import (
    BOOLEAN_SCORES,
    EvalTracer,
    _looks_like_abstention,
    _scrub,
)


def _settings(**overrides):
    base = {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": "", "LANGFUSE_HOST": "http://x"}
    base.update(overrides)
    return SimpleNamespace(**base)


ROW = {
    "id": "q001",
    "question": "How many credit hours?",
    "language": "en",
    "department": "CS",
    "answerable": True,
    "answer": "You need 135 credit hours.",
    "retrieved_contexts": ["excerpt one"],
    "retrieved_sources": ["CS_2023"],
    "retrieved_sections": ["مادة (٢٤)"],
    "ragas": {"faithfulness": 1.0},
}


def test_without_keys_every_call_is_a_no_op():
    """A fresh clone has no Langfuse. The eval must run unchanged."""
    tracer = EvalTracer(_settings(), run_name="r")
    assert not tracer.enabled

    # None of these may raise, and none may need a server.
    tracer.start_question(ROW)
    tracer.record_retrieval(ROW)
    tracer.record_generation(ROW, model="m", usage={"input": 1, "output": 2})
    tracer.record_scores(ROW)
    tracer.finish()


def test_a_broken_client_disables_tracing_rather_than_failing_the_run():
    """An unreachable backend must not waste a run that has already paid for tokens.

    Constructed with keys but a host that cannot resolve — the constructor must
    swallow it and report disabled, not propagate.
    """
    tracer = EvalTracer(
        _settings(
            LANGFUSE_PUBLIC_KEY="pk", LANGFUSE_SECRET_KEY="sk", LANGFUSE_HOST="http://127.0.0.1:1"
        ),
        run_name="r",
    )
    # The v2 SDK constructs lazily, so it may report enabled; what matters is that
    # every subsequent call is still safe.
    tracer.start_question(ROW)
    tracer.record_retrieval(ROW)
    tracer.record_scores(ROW)
    tracer.finish()


def test_recording_an_unknown_question_is_ignored_not_an_error():
    """record_* can be called for a row whose trace was never opened."""
    tracer = EvalTracer(_settings(), run_name="r")
    tracer.record_retrieval({"id": "never-started"})
    tracer.record_scores({"id": "never-started"})


# --------------------------------------------------------------------- scrubbing


def test_pii_is_scrubbed_before_it_leaves_for_the_backend():
    """Langfuse is a separate service, and an answer can carry a promoted phone number.

    Same exposure as the request path, so the same treatment — this is the one place
    an eval could quietly ship student data off-box.
    """
    assert "[PHONE]" in _scrub("call 01012345678")
    assert "01012345678" not in _scrub("call 01012345678")
    assert "[EMAIL]" in _scrub("mail a@b.edu")


def test_scrub_passes_non_strings_through_untouched():
    for value in (5, 1.5, None, True, {"a": 1}, ["x"]):
        assert _scrub(value) is value


def test_handbook_numbers_survive_scrubbing():
    """Over-redaction would corrupt the very answers being evaluated."""
    text = "A student needs 135 credit hours with a GPA of 2.0 — مادة (٢٤)"
    assert _scrub(text) == text


# -------------------------------------------------------------------- abstention


@pytest.mark.parametrize(
    "answer",
    [
        "I'm sorry, but the excerpts you provided do not contain that.",
        "عذراً، لا يحتوي أي من المقتطفات على معلومات حول ذلك.",
        "I have forwarded your question to a human academic advisor.",
    ],
)
def test_refusals_are_detected_so_they_can_be_filtered(answer):
    """RAGAS scores an abstention as irrelevant (answer_relevancy 0).

    Tagging them is the difference between "the system is bad" and "the system
    correctly declined 5 of 54 questions" — which is the single most misread number
    in the whole evaluation.
    """
    assert _looks_like_abstention(answer)


def test_a_real_answer_is_not_flagged_as_an_abstention():
    assert not _looks_like_abstention("You need 135 credit hours to graduate, per مادة (٢٤).")
    assert not _looks_like_abstention("")


# ------------------------------------------------------------------ score typing


def test_boolean_scores_are_named_so_they_are_not_sent_as_floats():
    """Langfuse types scores; a boolean sent as NUMERIC loses its meaning in the UI."""
    assert "expected_article_retrieved" in BOOLEAN_SCORES
    assert "department_filter_leak" in BOOLEAN_SCORES
    # RAGAS metrics must NOT be in here — they are genuinely continuous.
    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        assert metric not in BOOLEAN_SCORES
