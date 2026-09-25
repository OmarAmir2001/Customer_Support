"""Section 5 — the learning loop's quality gate.

The knowledge base has two ingestion paths: a curated one (handbooks, vetted before
they go in) and an automatic one (resolved tickets flowing back as
``instructor_resolved`` chunks). The second is an attack surface on your own KB, and
its failures are quiet — nobody notices until a student gets a confidently wrong
answer sourced from a bad past resolution.

This controller runs the two checks that stand in front of that path. It decides
nothing on its own:

* **Generalizability** sets the default state of the advisor's checkbox. Advice.
* **Contradiction** holds promotion and flags the handbook for review. A veto.

Why only one of them can block: a safety-critical call should not be handed to an
imperfect LLM judge. If the generalizability judge is wrong, the advisor flips the
box and nothing is lost. If it could reject, a wrong call would silently discard
good knowledge. A contradiction is different in kind — it does not mean "do not add
this", it means "one of these two sources is wrong", which is a question only a
human looking at the handbook can settle.
"""

import asyncio
import json

from pydantic import ValidationError

from customer_support.helpers.logging_config import get_logger
from customer_support.models.db_schemas import RetrievedDocument
from customer_support.models.llm_schemas.gate_result import JudgeVerdict
from customer_support.models.llm_schemas.promotion import PromotionAssessment
from customer_support.stores.llm.templates import JUDGE_LANGUAGE, TemplateParser, format_excerpts
from customer_support.stores.llm.templates.promotion import (
    CONTRADICTION_TEMPLATE,
    GENERALIZABILITY_TEMPLATE,
    PROMOTION_SYSTEM_PROMPT,
)

from .BaseController import BaseController

#: Sources a contradiction is judged against. Only the curated handbook counts:
#: "does this conflict with an earlier advisor's answer?" is a different question
#: with a different remedy, and mixing them would let one resolved answer veto
#: another.
AUTHORITATIVE_SOURCES = ("CS_2023", "IS_2023")


class PromotionController(BaseController):
    def __init__(self, judge_client, retrieval, templates: TemplateParser, settings=None):
        super().__init__(settings)
        self.judge_client = judge_client
        self.retrieval = retrieval
        self.templates = templates
        self.logger = get_logger(__name__)

    async def assess(
        self, question: str, answer: str, department: str | None = None
    ) -> PromotionAssessment:
        """Run both checks and return the advice. Never writes anything.

        The two are independent, so they run concurrently — the advisor is waiting
        on this before the checkbox renders, and sequencing them would double that
        wait for nothing.
        """
        handbook_excerpts = await self._handbook_context(
            question=question, answer=answer, department=department
        )

        generalizability, contradiction = await asyncio.gather(
            self._judge(
                name="generalizability",
                prompt=GENERALIZABILITY_TEMPLATE.substitute(question=question, answer=answer),
                # Fail SAFE, not closed: an unavailable judge must not silently
                # un-tick the box and lose knowledge, so an unknown generalizability
                # defaults to "probably general" and leaves the advisor to decide.
                fallback=JudgeVerdict(
                    score=1.0, reason="Generalizability could not be assessed automatically."
                ),
            ),
            self._judge(
                name="contradiction",
                prompt=CONTRADICTION_TEMPLATE.substitute(
                    answer=answer,
                    chunks=self._render(handbook_excerpts) or "(no handbook excerpts found)",
                ),
                # Also fail safe, and here safe means "do not hold". A hold is a veto
                # a human has to clear; manufacturing one from a broken judge would
                # block real answers on an outage.
                fallback=JudgeVerdict(
                    score=0.0, reason="Contradiction could not be assessed automatically."
                ),
            ),
        )

        assessment = PromotionAssessment(
            generalizable=(
                generalizability.score >= self.app_settings.PROMOTION_GENERALIZABILITY_THRESHOLD
            ),
            generalizability_score=generalizability.score,
            generalizability_reason=generalizability.reason,
            contradicts_handbook=(
                contradiction.score >= self.app_settings.PROMOTION_CONTRADICTION_THRESHOLD
            ),
            contradiction_score=contradiction.score,
            contradiction_reason=contradiction.reason,
            compared_against=[excerpt.text[:300] for excerpt in handbook_excerpts],
        )

        self.logger.info(
            "promotion_assessed",
            generalizable=assessment.generalizable,
            generalizability_score=round(generalizability.score, 3),
            contradicts=assessment.contradicts_handbook,
            contradiction_score=round(contradiction.score, 3),
            suggested_promote=assessment.suggested_promote,
            held=assessment.held,
            compared_against=len(handbook_excerpts),
        )
        return assessment

    # -------------------------------------------------------------- internals

    async def _handbook_context(
        self, question: str, answer: str, department: str | None
    ) -> list[RetrievedDocument]:
        """Handbook excerpts to judge the answer against.

        Retrieved on the QUESTION AND THE ANSWER TOGETHER, not the question alone.
        A contradiction is a conflict between what the ANSWER asserts and what the
        handbook says, so the claims in the answer are what has to drive retrieval.

        Searching on the question alone finds what the student asked about, which is
        subtly the wrong thing: an answer claiming "200 credit hours to graduate" was
        compared against excerpts on prerequisites and summer training — topically
        adjacent, none of them stating the real figure — so the check correctly
        reported no conflict with evidence that could not contain one. The question
        still contributes the topic; the answer contributes the claims.

        Deliberately excludes ``instructor_resolved`` rows: the question is whether
        this answer conflicts with the CURATED source, and letting one promoted
        answer contradict another would turn the KB's own history into a veto.
        """
        try:
            retrieved = await self.retrieval.retrieve(
                question=f"{question}\n{answer}",
                department=department,
                limit=self.app_settings.PROMOTION_CONTEXT_TOP_K,
            )
        except Exception as exc:
            # No context means the contradiction judge sees nothing and scores 0.0 —
            # it cannot invent a conflict, so a retrieval failure degrades to "no
            # hold" rather than blocking the advisor.
            self.logger.error("promotion_context_failed", error=str(exc), exc_info=True)
            return []

        return [
            document
            for document in retrieved
            if str((document.metadata or {}).get("source", "")).startswith(AUTHORITATIVE_SOURCES)
        ]

    def _render(self, excerpts: list[RetrievedDocument]) -> str:
        return format_excerpts(excerpts, self.templates, language=JUDGE_LANGUAGE)

    async def _judge(self, name: str, prompt: str, fallback: JudgeVerdict) -> JudgeVerdict:
        """One judge call, with no retry.

        Unlike the answer gates, neither verdict here is on the student's critical
        path — the advisor's answer reaches them regardless. So a failure costs a
        default, not a wrong answer, and the fallback each caller passes encodes
        which direction is safe for that particular check.
        """
        try:
            raw = await asyncio.to_thread(
                self.judge_client.generate_json,
                prompt,
                PROMOTION_SYSTEM_PROMPT,
                self.app_settings.JUDGE_MAX_OUTPUT_TOKENS,
                self.app_settings.JUDGE_TEMPERATURE,
            )
            if not raw:
                raise ValueError("promotion judge returned an empty response")
            return JudgeVerdict.model_validate_json(raw)

        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            self.logger.warning("promotion_judge_rejected", check=name, error=str(exc))
        except Exception as exc:
            self.logger.error("promotion_judge_failed", check=name, error=str(exc), exc_info=True)

        return fallback
