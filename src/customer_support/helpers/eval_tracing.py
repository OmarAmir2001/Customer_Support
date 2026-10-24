"""Langfuse tracing for OFFLINE evaluation runs.

Separate from `helpers/tracing.py`, and the difference is structural rather than
cosmetic. That module traces a live request: one trace per question, discovered
through a `ContextVar` because the graph's nodes have no way to pass a handle to each
other. This one traces a batch job, where:

* the caller holds every trace at once and attaches scores to them in a *later* phase
  (RAGAS scores one metric across all questions before moving to the next), so a
  contextvar is useless and a dict keyed on question id is exactly right;
* every trace belongs to one run, so they share a `session_id` and show up together
  in Langfuse's Sessions view — which is what makes "how did run B differ from run A"
  a thing you can look at rather than a diff of two JSON files;
* scores are the point. A live trace records what happened; an eval trace exists to
  carry a number you can compare across runs.

**Why trace an eval at all.** Until now the three systems were disconnected: RAGAS
scores landed in `reports/*.json`, chunking sweeps landed in MLflow, and Langfuse only
ever saw live `/chat` traffic. So the question "answer_relevancy is 0.57 — which
answers, and what were they shown?" meant opening a JSON file and reading arrays by
hand. With this, a bad score is a click into the exact prompt and excerpts that
produced it.

Same safety rule as the request-path tracer: **tracing must never fail the run.** An
evaluation that dies because an observability backend is unreachable has wasted real
money on model calls. Every method here swallows its own errors.
"""

from __future__ import annotations

from typing import Any

from customer_support.helpers.logging_config import get_logger
from customer_support.helpers.pii import redact

logger = get_logger(__name__)

#: Scores that are not in [0, 1] or are not numbers at all. Langfuse types scores,
#: and a boolean sent as a float loses the thing that makes it readable in the UI.
BOOLEAN_SCORES = (
    "expected_article_retrieved",
    "department_filter_leak",
    "gate_would_escalate",
)


def _scrub(value: Any) -> Any:
    """Redact strings on the way out, as the request path does.

    The eval questions are ours, but the ANSWERS are generated from retrieved
    chunks — and a promoted ticket answer is advisor-written free text that can carry
    a phone number. The same exposure as production, so the same treatment.
    """
    if not isinstance(value, str):
        return value
    redacted, _found = redact(value)
    return redacted


class EvalTracer:
    """One trace per evaluated question, grouped into one run.

    A no-op when the Langfuse keys are unset, so the eval scripts run unchanged for
    anyone who has not stood Langfuse up — exactly like the request-path tracer.
    """

    def __init__(
        self,
        settings,
        *,
        run_name: str,
        metadata: dict | None = None,
        dataset_name: str | None = None,
    ) -> None:
        self._client = None
        self._traces: dict[str, Any] = {}
        #: question id -> Langfuse dataset item, when a dataset is configured.
        self._items: dict[str, Any] = {}
        self.run_name = run_name
        self.metadata = metadata or {}
        self.dataset_name = dataset_name

        public_key = getattr(settings, "LANGFUSE_PUBLIC_KEY", None)
        secret_key = getattr(settings, "LANGFUSE_SECRET_KEY", None)
        if not public_key or not secret_key:
            logger.info("eval_tracing_disabled", reason="LANGFUSE keys not set")
            return

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=getattr(settings, "LANGFUSE_HOST", None) or "http://localhost:3002",
            )
            logger.info("eval_tracing_enabled", run=run_name)
        except Exception as exc:
            logger.warning("eval_tracing_unavailable", error=str(exc))
            self._client = None
            return

        self._load_dataset()

    def _load_dataset(self) -> None:
        """Map question id -> dataset item, so each trace can join the dataset run.

        Optional and non-fatal. Tracing alone makes ONE run browsable; the dataset is
        what makes two runs comparable, and an eval must not refuse to run because
        nobody has uploaded one yet (scripts/upload_eval_dataset.py does that).
        """
        if not self.dataset_name:
            return
        try:
            dataset = self._client.get_dataset(self.dataset_name)
            for item in dataset.items:
                question_id = (item.metadata or {}).get("question_id")
                if question_id:
                    self._items[question_id] = item
            logger.info("eval_dataset_loaded", dataset=self.dataset_name, items=len(self._items))
        except Exception as exc:
            logger.warning("eval_dataset_unavailable", dataset=self.dataset_name, error=str(exc))

    @property
    def enabled(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------- per question

    def start_question(self, row: dict) -> None:
        """Open the trace for one evaluated question.

        Keyed on the question id rather than returned, because the scoring phase runs
        much later and in a different loop over the same rows.
        """
        if not self.enabled:
            return
        try:
            trace = self._client.trace(
                name="eval-question",
                # Groups every question of this run. This is what turns the Sessions
                # view into a per-run view.
                session_id=self.run_name,
                input={"question": _scrub(row.get("question"))},
                metadata={
                    "question_id": row.get("id"),
                    "language": row.get("language"),
                    "department": row.get("department"),
                    "answerable": row.get("answerable"),
                    "expected_article": row.get("expected_article"),
                    "expected_sources": row.get("expected_sources"),
                    **self.metadata,
                },
                tags=[
                    "eval",
                    f"lang:{row.get('language')}",
                    f"dept:{row.get('department') or 'none'}",
                    "answerable" if row.get("answerable") else "unanswerable",
                ],
            )
            self._traces[row["id"]] = trace
        except Exception as exc:
            logger.warning("eval_trace_start_failed", question_id=row.get("id"), error=str(exc))
            return

        # Joining the trace to its dataset item is what creates a "dataset run item",
        # and that is the whole mechanism behind run-over-run comparison: Langfuse can
        # only line up two runs if both point at the same item.
        item = self._items.get(row.get("id"))
        if item is None:
            return
        try:
            item.link(trace, self.run_name, run_metadata=self.metadata)
        except Exception as exc:
            logger.warning("eval_dataset_link_failed", question_id=row.get("id"), error=str(exc))

    def record_retrieval(self, row: dict) -> None:
        """A span for what retrieval returned, with provenance.

        A span rather than a generation: nothing was generated. (Langfuse v2 has no
        `retriever` observation type — that arrived in v3 — so a named span is the
        most specific thing available here.)
        """
        trace = self._traces.get(row.get("id"))
        if trace is None:
            return
        try:
            span = trace.span(
                name="retrieve",
                input={
                    "question": _scrub(row.get("question")),
                    "department": row.get("department"),
                },
                output={
                    "chunks": len(row.get("retrieved_contexts") or []),
                    "sources": row.get("retrieved_sources"),
                    "sections": row.get("retrieved_sections"),
                },
                metadata={
                    "expected_article": row.get("expected_article"),
                    # The whole point of the eval: did retrieval find the article the
                    # ground truth says answers this question?
                    "expected_article_retrieved": any(
                        row.get("expected_article") and row["expected_article"] in section
                        for section in (row.get("retrieved_sections") or [])
                    ),
                },
            )
            span.end()
        except Exception as exc:
            logger.warning("eval_span_failed", span="retrieve", error=str(exc))

    def record_generation(
        self, row: dict, *, model: str | None, usage: dict | None, params: dict | None = None
    ) -> None:
        """A GENERATION observation, so Langfuse computes cost and model analytics.

        `usage` comes from the provider's own reported numbers (`last_usage`), not an
        estimate: Arabic and English tokenise very differently in these models, so a
        character-based guess is wrong by a language-dependent factor.
        """
        trace = self._traces.get(row.get("id"))
        if trace is None:
            return
        try:
            # `usage`, NOT `usage_details`.
            #
            # The SDK marks `usage` deprecated in favour of `usage_details`, and the
            # v2 SERVER silently ignores the newer field: it accepts the write, stores
            # promptTokens=0/completionTokens=0, and reports
            # `{"unit": null, "input": 0, "output": 0}` on read. No error anywhere —
            # the generation just shows zero tokens and no computed cost, which looks
            # like a provider that reports no usage rather than a field name mismatch.
            # Verified by writing with usage_details and reading the observation back.
            # Switch to usage_details if this is ever pointed at a v3+ server.
            token_usage = None
            if usage and (usage.get("input") or usage.get("output")):
                token_usage = {
                    "input": usage.get("input") or 0,
                    "output": usage.get("output") or 0,
                    "unit": "TOKENS",
                }

            # The EXCERPTS, not a count of them. The reason to trace an eval at all is
            # that a bad score becomes a click into what the model was actually shown;
            # `{"excerpts": 5}` answers none of the questions you would open it to ask.
            contexts = row.get("retrieved_contexts") or []
            generation = trace.generation(
                name="generate",
                model=model,
                model_parameters=params or {},
                input={
                    "question": _scrub(row.get("question")),
                    "excerpt_count": len(contexts),
                    "excerpts": [
                        {
                            "source": source,
                            "section": section,
                            # Truncated: five full handbook sections per trace is a lot
                            # of storage for text that is identical across runs, and
                            # the first 600 characters are what you read to judge
                            # whether retrieval found the right thing.
                            "text": _scrub(text)[:600],
                        }
                        for text, source, section in zip(
                            contexts,
                            row.get("retrieved_sources") or [""] * len(contexts),
                            row.get("retrieved_sections") or [""] * len(contexts),
                            strict=False,
                        )
                    ],
                },
                output=_scrub(row.get("answer")),
                usage=token_usage,
                metadata={
                    "generation_skipped": row.get("generation_skipped"),
                    "reference": _scrub(row.get("reference")),
                },
                level="ERROR" if row.get("generation_error") else "DEFAULT",
                status_message=row.get("generation_error"),
            )
            generation.end()
        except Exception as exc:
            logger.warning("eval_span_failed", span="generate", error=str(exc))

    def record_scores(self, row: dict) -> None:
        """Attach every number this row earned, typed.

        Called after the scoring phase, which is why the traces are held in a dict:
        RAGAS evaluates one metric across all questions before starting the next, so
        the scores for a given question arrive long after its trace was opened.
        """
        trace = self._traces.get(row.get("id"))
        if trace is None:
            return

        numeric = dict(row.get("ragas") or {})
        for key in (
            "ground_truth_recall",
            "gate_faithfulness",
            "gate_context_relevance",
            "gate_answer_relevance",
        ):
            if isinstance(row.get(key), int | float):
                numeric[key] = row[key]

        try:
            for name, value in numeric.items():
                if isinstance(value, int | float):
                    trace.score(name=name, value=float(value), data_type="NUMERIC")

            for name in BOOLEAN_SCORES:
                if name in row and row[name] is not None:
                    trace.score(name=name, value=1 if row[name] else 0, data_type="BOOLEAN")

            # The answer the row ended up with, now that it exists.
            trace.update(
                output={
                    "answer": _scrub(row.get("answer")),
                    "abstained": bool(row.get("answer")) and _looks_like_abstention(row["answer"]),
                }
            )
        except Exception as exc:
            logger.warning("eval_score_failed", question_id=row.get("id"), error=str(exc))

    # ------------------------------------------------------------------ teardown

    def finish(self) -> None:
        """Flush. Without this a script exits before the batching thread sends anything.

        The single most common way to instrument a script and see nothing in the UI.
        """
        if not self.enabled:
            return
        try:
            self._client.flush()
            logger.info("eval_tracing_flushed", run=self.run_name, traces=len(self._traces))
        except Exception as exc:
            logger.warning("eval_tracing_flush_failed", error=str(exc))


#: Phrases a refusal uses in either locale. Used only to TAG a trace, never to decide
#: anything — RAGAS scores an abstention as irrelevant (answer_relevancy 0), so being
#: able to filter them out in the UI is the difference between "the system is bad" and
#: "the system correctly declined 5 questions".
_ABSTENTION_MARKERS = (
    "sorry",
    "do not specify",
    "do not contain",
    "forwarded your question",
    "عذرا",
    "عذراً",
    "لا يحتوي",
    "لا تتضمن",
    "لا توجد",
    "المرشد الأكاديمي",
)


def _looks_like_abstention(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in _ABSTENTION_MARKERS)
