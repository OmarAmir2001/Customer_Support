"""Section 2 — the confidence decision, as three orthogonal single-purpose checks.

Why not one "confidence" number: LLMs are badly calibrated at self-reported
confidence, and averaging several prompts of the same model over the same chunks
adds no independent information — those judges share blind spots. Three checks that
each ask a different question about a different part of the pipeline do not.

This controller knows nothing about the graph. Every method can be called directly
from a pytest.
"""

import asyncio
import json

from pydantic import ValidationError

from customer_support.helpers.logging_config import get_logger
from customer_support.models.db_schemas import RetrievedDocument
from customer_support.models.enums.GateEnum import GateEnum, GateFailureReason
from customer_support.models.llm_schemas.gate_result import GateResult, JudgeVerdict

from .BaseController import BaseController
from ..stores.llm.templates.judge_prompts import (
    ANSWER_RELEVANCE_PROMPT,
    CONTEXT_RELEVANCE_PROMPT,
    FAITHFULNESS_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    format_excerpts,
)


class GradingController(BaseController):
    def __init__(self, generation_client, settings=None):
        super().__init__(settings)
        self.generation_client = generation_client
        self.logger = get_logger(__name__)

    # ------------------------------------------------------------------ gates

    async def check_context_relevance(
        self, question: str, chunks: list[RetrievedDocument]
    ) -> GateResult:
        """Gate 1, pre-generation. Cheapest check first: if the handbook does not
        cover the question, escalate without paying for a generation call."""

        if not chunks:
            # No LLM call needed: empty retrieval is already the answer.
            return GateResult(
                gate=GateEnum.CONTEXT_RELEVANCE.value,
                score=0.0,
                threshold=self.app_settings.GATE_CONTEXT_RELEVANCE_THRESHOLD,
                passed=False,
                reason=GateFailureReason.NO_RELEVANT_CONTEXT.value,
            )

        verdict = await self._judge(
            gate=GateEnum.CONTEXT_RELEVANCE,
            prompt=CONTEXT_RELEVANCE_PROMPT.format(
                question=question, chunks=self._format_chunks(chunks)
            ),
            fallback_reason=GateFailureReason.NO_RELEVANT_CONTEXT,
        )
        return self._apply_threshold(
            GateEnum.CONTEXT_RELEVANCE,
            verdict,
            self.app_settings.GATE_CONTEXT_RELEVANCE_THRESHOLD,
        )

    async def check_faithfulness(self, answer: str, chunks: list[RetrievedDocument]) -> GateResult:
        """Gate 2a, post-generation. Claim decomposition: score = supported / total."""

        verdict = await self._judge(
            gate=GateEnum.FAITHFULNESS,
            prompt=FAITHFULNESS_PROMPT.format(answer=answer, chunks=self._format_chunks(chunks)),
            fallback_reason=GateFailureReason.NOT_GROUNDED,
        )
        return self._apply_threshold(
            GateEnum.FAITHFULNESS, verdict, self.app_settings.GATE_FAITHFULNESS_THRESHOLD
        )

    async def check_answer_relevance(self, question: str, answer: str) -> GateResult:
        """Gate 2b, post-generation. Sees no chunks on purpose: a grounded answer to
        the wrong question must fail here, and faithfulness will never catch it."""

        verdict = await self._judge(
            gate=GateEnum.ANSWER_RELEVANCE,
            prompt=ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer),
            fallback_reason=GateFailureReason.OFF_TOPIC,
        )
        return self._apply_threshold(
            GateEnum.ANSWER_RELEVANCE, verdict, self.app_settings.GATE_ANSWER_RELEVANCE_THRESHOLD
        )

    async def check_post_generation(
        self, question: str, answer: str, chunks: list[RetrievedDocument]
    ) -> list[GateResult]:
        """Both post-generation gates, concurrently: they are independent, so running
        them in sequence would double the latency for no benefit."""

        return list(
            await asyncio.gather(
                self.check_faithfulness(answer=answer, chunks=chunks),
                self.check_answer_relevance(question=question, answer=answer),
            )
        )

    # -------------------------------------------------------------- internals

    async def _judge(
        self, gate: GateEnum, prompt: str, fallback_reason: GateFailureReason
    ) -> JudgeVerdict:
        """Run one judge with a single retry, then fail CLOSED.

        Fail closed means: an unparseable or unavailable judge escalates to a human.
        Failing open would let unverified answers reach students precisely when the
        verification machinery is broken.
        """

        for attempt in (1, 2):
            try:
                # The provider SDK is synchronous; to_thread keeps it off the event
                # loop so concurrent requests are not blocked by a judge call.
                raw = await asyncio.to_thread(
                    self.generation_client.generate_json,
                    prompt,
                    JUDGE_SYSTEM_PROMPT,
                    self.app_settings.JUDGE_MAX_OUTPUT_TOKENS,
                    self.app_settings.JUDGE_TEMPERATURE,
                )
                if not raw:
                    raise ValueError("judge returned an empty response")

                return JudgeVerdict.model_validate_json(raw)

            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                self.logger.warning(
                    "judge_output_rejected", gate=gate.value, attempt=attempt, error=str(exc)
                )
            except Exception as exc:  # network error, rate limit, provider outage
                self.logger.error(
                    "judge_call_failed",
                    gate=gate.value,
                    attempt=attempt,
                    error=str(exc),
                    exc_info=True,
                )

        self.logger.error("judge_unavailable_failing_closed", gate=gate.value)
        return JudgeVerdict(score=0.0, reason=fallback_reason.value)

    def _apply_threshold(
        self, gate: GateEnum, verdict: JudgeVerdict, threshold: float
    ) -> GateResult:
        passed = verdict.score >= threshold

        # One line per gate: these are the numbers you later replay into MLflow when
        # tuning thresholds, so log the score even when the gate passes.
        self.logger.info(
            "gate_evaluated",
            gate=gate.value,
            score=round(verdict.score, 3),
            threshold=threshold,
            passed=passed,
        )

        return GateResult(
            gate=gate.value,
            score=verdict.score,
            threshold=threshold,
            passed=passed,
            reason=verdict.reason,
        )

    @staticmethod
    def _format_chunks(chunks: list[RetrievedDocument]) -> str:
        """Numbered, provenance-labelled excerpts, formatted the same way the generator
        formats them — see ``judge_prompts.format_excerpts``."""
        return format_excerpts(chunks)