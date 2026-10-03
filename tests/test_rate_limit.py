"""Rate-limit retry: what gets retried, how long it waits, and what it refuses to do.

Written against a measured failure, not a hypothetical. A ramped load test found the
judge model capped at 8,000 TPM, reached at five concurrent users: 94 questions, 5
answered, 110 failures, every one a 429 carrying "Please try again in 1.875s" that the
code ignored. Because the judges fail closed, each became an escalation while /chat
returned 200 OK.

The two properties that matter most here are the ones easiest to get wrong:

* **Only rate limits are retried.** A malformed response will not fix itself by
  waiting, and retrying it burns the budget a real 429 needs. More importantly, if
  this swallowed other exceptions the judges' fail-closed path would never see them.
* **The wait is bounded twice** — per sleep and in total. An unbounded wait honours a
  hostile Retry-After by parking a student's request, and three gates multiply it.

Sleep is injected throughout, so the waits are asserted rather than spent.
"""

import pytest

from customer_support.stores.llm.rate_limit import (
    call_with_rate_limit_retry,
    is_rate_limit,
    retry_after_seconds,
)

POLICY = {"max_retries": 3, "max_wait": 8.0, "total_budget": 12.0}


class FakeResponse:
    def __init__(self, headers=None, status_code=429):
        self.headers = headers or {}
        self.status_code = status_code


class RateLimitError(Exception):
    """Shaped like the OpenAI SDK's class of the same name. Duck-typed on purpose —
    see is_rate_limit."""

    def __init__(self, message="rate limited", response=None):
        super().__init__(message)
        self.response = response


class Boom(Exception):
    """Anything that is not a rate limit."""


class Recorder:
    """A sleep that records instead of sleeping."""

    def __init__(self):
        self.waits: list[float] = []

    def __call__(self, seconds):
        self.waits.append(round(seconds, 4))

    @property
    def total(self):
        return round(sum(self.waits), 4)


def flaky(fail_times, exc_factory=RateLimitError, result="ok"):
    """A call that fails `fail_times` times, then succeeds."""
    state = {"calls": 0}

    def call():
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise exc_factory()
        return result

    call.state = state
    return call


# ------------------------------------------------------- recognising a rate limit


def test_the_sdk_exception_class_is_recognised_without_importing_it():
    """Duck-typed so one helper serves both the OpenAI and Cohere clients, and so a
    rename in either SDK cannot break the import."""
    assert is_rate_limit(RateLimitError())


def test_a_429_status_is_recognised_whatever_the_class_is_called():
    class WeirdlyNamed(Exception):
        status_code = 429

    assert is_rate_limit(WeirdlyNamed())


def test_a_429_on_the_nested_response_is_recognised():
    """Some SDKs put the status on the exception, others only on the response it
    carries. Both shapes have to count, or a real 429 propagates as an outage."""

    class OnlyOnResponse(Exception):
        response = FakeResponse(status_code=429)

    assert is_rate_limit(OnlyOnResponse())


def test_an_ordinary_error_is_not_a_rate_limit():
    assert not is_rate_limit(Boom("connection reset"))
    assert not is_rate_limit(ValueError("bad json"))


# ------------------------------------------------------------- choosing the wait


def test_the_retry_after_header_wins_because_it_is_the_standard():
    exc = RateLimitError(response=FakeResponse({"retry-after": "3"}))
    assert retry_after_seconds(exc, attempt=1, max_wait=8.0) == 3.0


def test_the_hint_in_the_message_is_used_when_there_is_no_header():
    """Groq's exact wording. 1.875s is not arbitrary — it is how long until the token
    window refills, which is better information than any backoff curve."""
    exc = RateLimitError("Rate limit reached... Please try again in 1.875s. Need more tokens?")
    assert retry_after_seconds(exc, attempt=1, max_wait=8.0) == 1.875


def test_a_millisecond_hint_is_converted_not_taken_literally():
    """Some providers report milliseconds. Reading 500ms as 500 seconds would park a
    request for eight minutes — or, capped, for the full ceiling."""
    exc = RateLimitError("Please try again in 500ms")
    assert retry_after_seconds(exc, attempt=1, max_wait=8.0) == 0.5


def test_backoff_is_only_the_fallback_when_the_provider_said_nothing():
    exc = RateLimitError("too many requests")
    assert retry_after_seconds(exc, attempt=1, max_wait=8.0) == 1.0
    assert retry_after_seconds(exc, attempt=2, max_wait=8.0) == 2.0
    assert retry_after_seconds(exc, attempt=3, max_wait=8.0) == 4.0


def test_a_hostile_retry_after_cannot_park_the_request():
    exc = RateLimitError(response=FakeResponse({"retry-after": "600"}))
    assert retry_after_seconds(exc, attempt=1, max_wait=8.0) == 8.0


def test_a_garbage_header_falls_through_to_the_message_then_backoff():
    exc = RateLimitError("nothing useful", response=FakeResponse({"retry-after": "soon"}))
    assert retry_after_seconds(exc, attempt=1, max_wait=8.0) == 1.0


def test_a_zero_hint_still_waits_a_little():
    """An immediate retry would hit the same limit — which is precisely the bug this
    whole module exists to fix."""
    exc = RateLimitError("Please try again in 0s")
    assert retry_after_seconds(exc, attempt=1, max_wait=8.0) > 0


# -------------------------------------------------------------- the retry loop


def test_a_transient_rate_limit_becomes_a_short_wait_not_a_failure():
    """The headline behaviour. Before this, these two 429s were two escalations."""
    sleep = Recorder()
    call = flaky(fail_times=2)

    result = call_with_rate_limit_retry(call, operation="generate_json", sleep=sleep, **POLICY)

    assert result == "ok"
    assert call.state["calls"] == 3
    assert sleep.waits == [1.0, 2.0]


def test_a_non_rate_limit_error_propagates_immediately_and_untouched():
    """Load-bearing. The judges' fail-closed path depends on seeing these. If this
    retried or swallowed them, a malformed verdict would look like an outage and the
    gate would never record a real failure."""
    sleep = Recorder()
    call = flaky(fail_times=1, exc_factory=Boom)

    with pytest.raises(Boom):
        call_with_rate_limit_retry(call, operation="generate_json", sleep=sleep, **POLICY)

    assert call.state["calls"] == 1, "must not retry a non-rate-limit error"
    assert sleep.waits == []


def test_retries_are_capped_and_then_it_gives_up():
    sleep = Recorder()
    call = flaky(fail_times=99)

    with pytest.raises(RateLimitError):
        call_with_rate_limit_retry(call, operation="generate_json", sleep=sleep, **POLICY)

    assert call.state["calls"] == 4, "one initial attempt plus max_retries"
    assert len(sleep.waits) == 3


def test_the_total_budget_stops_it_before_the_wait_not_after():
    """Sleeping past the budget and then failing anyway would spend the student's
    patience and still escalate. The check comes first."""
    sleep = Recorder()
    call = flaky(fail_times=99)
    policy = {"max_retries": 5, "max_wait": 8.0, "total_budget": 3.0}

    with pytest.raises(RateLimitError):
        call_with_rate_limit_retry(call, operation="generate_json", sleep=sleep, **policy)

    assert sleep.total <= 3.0
    # 1.0 then 2.0 fits exactly; the next would be 4.0 and is refused rather than
    # trimmed, because a trimmed wait is too short to clear the limit anyway.
    assert sleep.waits == [1.0, 2.0]


def test_a_call_that_succeeds_first_time_never_sleeps():
    sleep = Recorder()
    call = flaky(fail_times=0)

    assert call_with_rate_limit_retry(call, operation="embed_text", sleep=sleep, **POLICY) == "ok"
    assert sleep.waits == []


def test_the_provider_is_actually_wired_to_the_retry():
    """The integration that could silently not exist.

    Every test above exercises the helper. None of them proves OpenAIProvider calls
    it — and a helper nothing invokes is the most expensive kind of dead code, because
    the tests are green and the bug is still in production.

    Drives a real OpenAIProvider with a fake SDK client that 429s once.
    """
    from customer_support.stores.llm.providers.OpenAIProvider import OpenAIProvider

    calls = {"n": 0}

    class FakeCompletions:
        def create(self, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RateLimitError("Please try again in 0.25s")

            class Message:
                content = '{"score": 1.0, "reason": "ok"}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    provider = OpenAIProvider(api_key="test-key", rate_limit_max_retries=2)
    provider.set_generation_model("some-judge-model")
    provider.client = type(
        "FakeClient", (), {"chat": type("Chat", (), {"completions": FakeCompletions()})()}
    )()

    raw = provider.generate_json(prompt="q", system_prompt="s")

    assert calls["n"] == 2, "the 429 was not retried — the provider is not wired up"
    assert raw == '{"score": 1.0, "reason": "ok"}'


def test_the_providers_own_hint_is_preferred_over_backoff_across_retries():
    """End to end with Groq's real message: three attempts should wait what Groq asked
    for, not 1-2-4."""
    sleep = Recorder()
    state = {"calls": 0}

    def call():
        state["calls"] += 1
        if state["calls"] <= 2:
            raise RateLimitError("Rate limit reached. Please try again in 1.5s.")
        return "ok"

    call_with_rate_limit_retry(call, operation="generate_json", sleep=sleep, **POLICY)

    assert sleep.waits == [1.5, 1.5]
