"""Rate-limit retry for synchronous provider SDK calls.

This lives at the provider layer rather than in the judges, and that placement is the
point. Rate limiting is provider knowledge — the 429, the `Retry-After` header, the
wording of the hint in the body — and every caller hits the same wall: the three judge
gates, generation, memory extraction and the promotion assessment all go to the same
Groq account. One implementation here beats four copies that drift.

**Why it exists.** A ramped load test against Groq's free tier found the judge model
capped at 8,000 tokens/minute, reached at *five* concurrent users. Of 94 questions,
5 were answered; all 110 failures were 429s. Groq had replied "Please try again in
1.875s" every time, and the code ignored it — retrying twice immediately, so both
retries hit the same limit. Because the judges fail closed, each one became an
escalation while `/chat` returned 200 OK throughout.

The app has since moved to a paid Cohere key, which removes that particular ceiling
but not the need for this: a separate Cohere trial-key 429 on embeddings killed an
evaluation sweep partway through, and the reports it left behind still carry
`"ragas": null`. Both providers route through here now.

**`time.sleep`, not `asyncio.sleep`, is correct here.** These SDKs are synchronous and
every call already runs inside `asyncio.to_thread`, so the sleep blocks a worker
thread rather than the event loop.

That is not free, and the trade is worth stating: a sleeping thread still holds one of
the executor's slots (capped at `min(32, cpu_count + 4)`), so enough concurrent
backoff can starve the pool. It is still strictly better than failing closed — a slow
answer beats a wrongly escalated one — but the real fix for slot contention is a
semaphore at this same boundary, which is separate work.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable

from customer_support.helpers.logging_config import get_logger

logger = get_logger(__name__)

#: Providers word the hint differently and not all of them send a header. Groq uses
#: "Please try again in 1.875s"; others use milliseconds or a bare number of seconds.
_HINT_PATTERNS = (
    re.compile(r"try again in\s+([0-9]*\.?[0-9]+)\s*ms", re.I),
    re.compile(r"try again in\s+([0-9]*\.?[0-9]+)\s*s", re.I),
    re.compile(r"retry[- ]after[:\s]+([0-9]*\.?[0-9]+)", re.I),
)


def is_rate_limit(exc: BaseException) -> bool:
    """Whether this exception is the provider refusing us for rate reasons.

    Deliberately duck-typed rather than importing `openai.RateLimitError`: the same
    helper serves the Cohere client, and a hard import would couple this module to
    one SDK and break if either renames its exception class.
    """
    if exc.__class__.__name__ in {"RateLimitError", "TooManyRequestsError"}:
        return True

    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    return status == 429


def retry_after_seconds(exc: BaseException, attempt: int, max_wait: float) -> float:
    """How long to wait, preferring what the provider actually told us.

    Order matters. The `Retry-After` header is the standard and the most trustworthy.
    The hint in the message body is provider-specific but precise — Groq's 1.875s is
    exactly how long until the token window refills. Exponential backoff is the last
    resort, used only when the provider said nothing: it is a guess, and a guess that
    waits too long is a student staring at a spinner.
    """
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            header = response.headers.get("retry-after")
        except Exception:  # a mock, or an SDK that shapes this differently
            header = None
        if header:
            try:
                return min(float(header), max_wait)
            except (TypeError, ValueError):
                pass

    message = str(exc)
    for pattern in _HINT_PATTERNS:
        match = pattern.search(message)
        if match:
            value = float(match.group(1))
            # The millisecond pattern is first, so a match on it means milliseconds.
            if pattern is _HINT_PATTERNS[0]:
                value /= 1000.0
            # A hint of 0 would make the retry immediate and pointless; floor it.
            return min(max(value, 0.05), max_wait)

    return min(2.0 ** (attempt - 1), max_wait)


def call_with_rate_limit_retry[T](
    call: Callable[[], T],
    *,
    operation: str,
    max_retries: int,
    max_wait: float,
    total_budget: float,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `call`, retrying only on rate limits, honouring the provider's own hint.

    **Only rate limits are retried.** A malformed response or a bad request will not
    fix itself by waiting, and retrying it burns the budget that a genuine 429 needs.
    Anything that is not a rate limit propagates immediately, so the caller's own
    error handling — including the judges' fail-closed path — still sees it unchanged.

    `sleep` is injected so tests can assert the waits without spending them.
    """
    spent = 0.0

    for attempt in range(1, max_retries + 2):  # max_retries retries after the first try
        try:
            return call()
        except Exception as exc:
            if not is_rate_limit(exc) or attempt > max_retries:
                raise

            wait = retry_after_seconds(exc, attempt, max_wait)

            # Stop before exceeding the budget rather than after. Sleeping past it
            # and then failing would spend the student's patience and still escalate.
            if spent + wait > total_budget:
                logger.warning(
                    "provider_rate_limit_budget_exhausted",
                    operation=operation,
                    attempt=attempt,
                    would_wait=round(wait, 3),
                    already_spent=round(spent, 3),
                    budget=total_budget,
                )
                raise

            logger.warning(
                "provider_rate_limited_retrying",
                operation=operation,
                attempt=attempt,
                waiting=round(wait, 3),
                # Whether the number came from the provider or from our own backoff.
                # Worth distinguishing: a run full of guessed waits means the provider
                # stopped telling us, and the numbers are no longer grounded.
                source="provider_hint" if wait != 2.0 ** (attempt - 1) else "backoff",
            )
            sleep(wait)
            spent += wait

    # Unreachable: the loop either returns or raises.
    raise RuntimeError(f"rate limit retry loop exited without result for {operation}")
