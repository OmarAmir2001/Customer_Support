"""What the metrics must be able to tell apart.

The HTTP metrics are generic. These two are not, and they exist because this system's
worst failure is invisible in HTTP terms: the judges fail closed, so a degraded model
provider turns every question into an escalation while /chat returns 200 with normal
latency and zero errors.

The test that matters most here is the last one — that a judge which FAILED and a
judge which merely scored low are counted differently. Downstream they are identical
by design (same score, same gate reason, same ticket), so if the counter cannot
separate them, nothing can.
"""

import pytest
from prometheus_client import CollectorRegistry, Counter

from customer_support.controllers.GradingController import GradingController
from customer_support.models.db_schemas import RetrievedDocument
from customer_support.stores.llm.templates import TemplateParser
from customer_support.utils import metrics

TEMPLATES = TemplateParser(primary_language="en", default_language="en")


class FakeSettings:
    GATE_CONTEXT_RELEVANCE_THRESHOLD = 0.5
    GATE_FAITHFULNESS_THRESHOLD = 0.8
    GATE_ANSWER_RELEVANCE_THRESHOLD = 0.7
    JUDGE_MAX_OUTPUT_TOKENS = 300
    JUDGE_TEMPERATURE = 0.0
    ASSETS_DIR = "assets"


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate_json(self, prompt, system_prompt=None, max_output_tokens=None, temperature=None):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.fixture
def counters(monkeypatch):
    """Fresh counters per test, on a private registry.

    The module-level counters are process-global, so without this one test's
    increments would leak into the next and the assertions would depend on ordering.
    """
    registry = CollectorRegistry()
    questions = Counter(
        "questions_total", "test", ["outcome", "failed_gate", "department"], registry=registry
    )
    failures = Counter("judge_failures_total", "test", ["gate"], registry=registry)
    monkeypatch.setattr(metrics, "QUESTIONS", questions)
    monkeypatch.setattr(metrics, "JUDGE_FAILURES", failures)

    def value(counter, **labels):
        return counter.labels(**labels)._value.get()

    return type(
        "C",
        (),
        {"questions": questions, "failures": failures, "value": staticmethod(value)},
    )


def chunk(text="Students need 135 credit hours.", score=0.9):
    return RetrievedDocument(text=text, score=score, metadata={"source": "CS_2023"})


def grader(response):
    return GradingController(
        generation_client=FakeLLM(response), templates=TEMPLATES, settings=FakeSettings()
    )


# ------------------------------------------------------------ question outcomes


def test_an_answered_question_is_labelled_none_not_dropped(counters):
    """failed_gate="none" rather than an absent series: escalation rate is then one
    expression over one metric, instead of a ratio of two that can drift apart."""
    metrics.record_question(outcome="answered", failed_gate=None, department="CS")

    assert (
        counters.value(counters.questions, outcome="answered", failed_gate="none", department="CS")
        == 1.0
    )


def test_an_escalation_records_which_gate_sent_it(counters):
    metrics.record_question(outcome="escalated", failed_gate="faithfulness", department="IS")

    assert (
        counters.value(
            counters.questions, outcome="escalated", failed_gate="faithfulness", department="IS"
        )
        == 1.0
    )


def test_a_missing_department_does_not_create_an_empty_label(counters):
    """A department is optional on the request. "none" keeps the series readable and,
    more importantly, keeps it a FIXED value rather than whatever arrived."""
    metrics.record_question(outcome="answered", failed_gate=None, department=None)

    assert (
        counters.value(
            counters.questions, outcome="answered", failed_gate="none", department="none"
        )
        == 1.0
    )


def test_labels_stay_bounded(counters):
    """Cardinality is the failure mode that kills Prometheus. Every label here is
    drawn from a closed set — outcomes, gates, departments — so the series count has
    a ceiling that does not depend on traffic."""
    for _ in range(50):
        metrics.record_question(
            outcome="escalated", failed_gate="context_relevance", department="CS"
        )

    assert len(list(counters.questions.collect()[0].samples)) <= 3  # total/created/_
    assert (
        counters.value(
            counters.questions,
            outcome="escalated",
            failed_gate="context_relevance",
            department="CS",
        )
        == 50.0
    )


# ------------------------------------------------- the distinction that matters


@pytest.mark.asyncio
async def test_a_judge_that_fails_is_counted_as_a_failure(counters):
    """A provider outage. The gate escalates, and the counter says why."""
    result = await grader(RuntimeError("provider is down")).check_faithfulness(
        answer="135 credit hours.", chunks=[chunk()]
    )

    assert not result.passed  # failed CLOSED
    assert counters.value(counters.failures, gate="faithfulness") == 1.0


@pytest.mark.asyncio
async def test_a_judge_that_merely_scores_low_is_NOT_counted_as_a_failure(counters):
    """The other half, and the reason the counter exists.

    Both paths produce an escalation with the same gate reason, so only this counter
    separates "retrieval is bad" from "the provider is down" — and those need
    opposite responses.
    """
    result = await grader('{"score": 0.1, "reason": "not supported"}').check_faithfulness(
        answer="135 credit hours.", chunks=[chunk()]
    )

    assert not result.passed  # escalates too...
    assert counters.value(counters.failures, gate="faithfulness") == 0.0  # ...but is not a failure


@pytest.mark.asyncio
async def test_unparseable_output_counts_once_not_once_per_retry(counters):
    """_judge retries before giving up. The counter must record one FAILURE, not one
    per attempt, or a retry policy change would silently rescale the metric."""
    llm = grader("this is not json")
    result = await llm.check_answer_relevance(question="q", answer="a")

    assert not result.passed
    assert counters.value(counters.failures, gate="answer_relevance") == 1.0
