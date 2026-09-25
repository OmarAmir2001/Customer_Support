"""Prometheus instrumentation.

Three things here are not the obvious implementation, and each one is the difference
between a dashboard you can trust and one that quietly lies.

**Labels use the matched ROUTE TEMPLATE, never the request path.** `request.url.path`
would make `/api/v1/chat/<uuid>` a distinct time series for every conversation, and
`/tickets/3/claim` a distinct one for every ticket — unbounded, permanent, and
invisible until Prometheus runs out of memory. The app has eleven routes; the naive
version has as many series as it has ever had requests.

**The registry is multiprocess-aware.** uvicorn runs with `--workers`, and
prometheus_client keeps counters in process memory. A scrape lands on whichever
worker the OS gives it, so a plain registry reports that worker's slice — about a
fifth of real traffic here, and a different fifth each scrape.

**Recording happens in a `finally`.** An unhandled exception must still be counted,
or the error rate under-reports exactly the failures worth alerting on.
"""

import os
import time

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    multiprocess,
)
from starlette.middleware.base import BaseHTTPMiddleware

#: Where the scrape endpoint lives. Obscure, but treat that as tidiness rather than
#: security — nginx must still refuse it from outside, because Prometheus reaches the
#: app directly over the compose network and never needs the public route.
METRICS_PATH = "/TrhBVe"

#: Anything that matched no route. Bucketed under one label rather than recorded as
#: the URL that was probed: a public port gets scanned, and `/wp-login.php` must not
#: be able to mint a permanent time series.
UNMATCHED = "__unmatched__"

REQUEST_COUNT = Counter(
    "customer_support_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "customer_support_request_latency_seconds",
    "Request latency",
    ["method", "endpoint"],
)


# --------------------------------------------------------------- domain metrics
#
# The HTTP metrics above describe any web app. These two describe THIS one, and they
# exist because the system's worst failure is invisible in HTTP terms: the judges fail
# closed, so when the model provider degrades, every question escalates while /chat
# keeps returning 200 with normal latency and no errors. Request-rate and error-rate
# panels would look perfect throughout.

QUESTIONS = Counter(
    "customer_support_questions_total",
    "Questions by outcome, and which gate sent them to a human",
    # failed_gate is "none" when answered, so escalation RATE is one expression over
    # one metric rather than a ratio of two that can drift apart.
    ["outcome", "failed_gate", "department"],
)

JUDGE_FAILURES = Counter(
    "customer_support_judge_failures_total",
    "Judges that gave up after their retries and failed closed",
    ["gate"],
)


def record_question(outcome: str, failed_gate: str | None, department: str | None) -> None:
    """One completed question. Called once per run, from the only entry point."""
    QUESTIONS.labels(outcome, failed_gate or "none", department or "none").inc()


def record_judge_failure(gate: str) -> None:
    """A judge that could not be reached or understood, and so escalated by default.

    Separate from the escalation counter, and that separation is the entire point. A
    fail-closed judge returns the same gate-specific reason as a judge that genuinely
    scored the answer low, so nothing downstream — not the ticket, not the logs' gate
    reason, not the escalation count — can tell "retrieval is bad" from "the provider
    is down". They need opposite responses, so they need separate counters.
    """
    JUDGE_FAILURES.labels(gate).inc()


def _route_template(request: Request) -> str:
    """The matched route's path, e.g. ``/api/v1/chat/{thread_id}``.

    Only readable AFTER routing, which is why the caller reads it once the downstream
    app has run rather than up front — Starlette sets ``scope["route"]`` when it picks
    the endpoint, and before that there is nothing to read.
    """
    return getattr(request.scope.get("route"), "path", None) or UNMATCHED


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Don't measure the scrape. It would otherwise be the busiest "endpoint" on
        # the dashboard and tell you nothing about the app.
        if request.url.path == METRICS_PATH:
            return await call_next(request)

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            REQUEST_LATENCY.labels(request.method, _route_template(request)).observe(
                time.perf_counter() - start
            )
            REQUEST_COUNT.labels(request.method, _route_template(request), status_code).inc()


def _registry() -> CollectorRegistry | None:
    """The registry a scrape should read, or None when the default one is correct.

    With PROMETHEUS_MULTIPROC_DIR set, every worker writes counters to mmap files in
    that directory and this collector sums them. Without it — a single process, or a
    test — the default in-process registry already holds everything.
    """
    if not os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        return None

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return registry


def setup_metrics(app: FastAPI) -> None:
    """Install the middleware and the scrape endpoint."""
    app.add_middleware(PrometheusMiddleware)

    @app.get(METRICS_PATH, include_in_schema=False)
    async def metrics() -> Response:
        registry = _registry()
        payload = generate_latest(registry) if registry is not None else generate_latest()
        return Response(payload, media_type=CONTENT_TYPE_LATEST)
