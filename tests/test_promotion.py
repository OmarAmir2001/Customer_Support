"""Section 5's quality gate: what may block promotion, and what may only advise.

The asymmetry between the two checks is the design, so it is what these tests pin
down. Generalizability moves a checkbox and nothing more; contradiction is the one
veto. Getting that backwards would either silently discard good knowledge or let a
conflicting answer become a competing chunk.
"""

import json

import pytest

from customer_support.controllers.PromotionController import PromotionController
from customer_support.models.db_schemas import RetrievedDocument
from customer_support.models.llm_schemas.promotion import PromotionAssessment
from customer_support.stores.llm.templates import TemplateParser

TEMPLATES = TemplateParser(primary_language="en", default_language="en")


class FakeSettings:
    PROMOTION_GENERALIZABILITY_THRESHOLD = 0.6
    PROMOTION_CONTRADICTION_THRESHOLD = 0.6
    PROMOTION_CONTEXT_TOP_K = 5
    JUDGE_MAX_OUTPUT_TOKENS = 1200
    JUDGE_TEMPERATURE = 0.0
    ASSETS_DIR = "assets"


class FakeJudge:
    """Answers each check by looking at which rubric it was handed."""

    def __init__(self, generalizability=1.0, contradiction=0.0, raise_on=None):
        self.generalizability = generalizability
        self.contradiction = contradiction
        self.raise_on = raise_on or set()
        self.calls = []

    def generate_json(self, prompt, system_prompt=None, max_output_tokens=None, temperature=None):
        check = "contradiction" if "CONTRADICTS" in prompt else "generalizability"
        self.calls.append(check)
        if check in self.raise_on:
            raise RuntimeError("judge is down")
        score = self.contradiction if check == "contradiction" else self.generalizability
        return json.dumps({"score": score, "reason": f"{check} verdict"})


class FakeRetrieval:
    def __init__(self, documents=None, fail=False):
        self.documents = documents or []
        self.fail = fail

    async def retrieve(self, question, department=None, limit=5):
        if self.fail:
            raise RuntimeError("vector store is down")
        return self.documents


def handbook(text, source="CS_2023"):
    return RetrievedDocument(
        text=text, score=0.9, metadata={"source": source, "section": "4.2", "department": "CS"}
    )


def promoted(text):
    return RetrievedDocument(
        text=text, score=0.95, metadata={"source": "instructor_resolved", "ticket_id": "7"}
    )


def controller(judge=None, retrieval=None):
    return PromotionController(
        judge_client=judge or FakeJudge(),
        retrieval=retrieval or FakeRetrieval(),
        templates=TEMPLATES,
        settings=FakeSettings(),
    )


# ------------------------------------------------------- generalizability advises


@pytest.mark.asyncio
async def test_a_general_answer_is_suggested_for_promotion():
    assessment = await controller(FakeJudge(generalizability=0.9)).assess(
        question="how long to withdraw?", answer="Students may withdraw within two weeks."
    )

    assert assessment.generalizable
    assert assessment.suggested_promote
    assert not assessment.held


@pytest.mark.asyncio
async def test_a_student_specific_answer_is_not_suggested_but_is_not_blocked():
    """The whole point of 'suggests, never rejects': the box defaults to unticked,
    and the advisor can still tick it."""
    assessment = await controller(FakeJudge(generalizability=0.1)).assess(
        question="can I get an extension?",
        answer="Your deadline is extended to Friday because of your medical excuse.",
    )

    assert not assessment.generalizable
    assert not assessment.suggested_promote
    # Not held — nothing stops the advisor overriding it.
    assert not assessment.held


# --------------------------------------------------------- contradiction vetoes


@pytest.mark.asyncio
async def test_a_contradiction_holds_promotion():
    assessment = await controller(
        FakeJudge(generalizability=1.0, contradiction=0.9),
        FakeRetrieval([handbook("Students may withdraw within two weeks.")]),
    ).assess(question="how long to withdraw?", answer="You may withdraw within six weeks.")

    assert assessment.contradicts_handbook
    assert assessment.held
    # Held even though the answer is perfectly general — this is the one check that
    # overrides the advisor rather than advising them.
    assert assessment.generalizable
    assert not assessment.suggested_promote
    assert assessment.hold_reason()


@pytest.mark.asyncio
async def test_adding_information_the_handbook_lacks_is_not_a_contradiction():
    """Filling a gap is the entire point of the learning loop. If gap-filling
    registered as a conflict, nothing would ever be promoted."""
    assessment = await controller(
        FakeJudge(generalizability=1.0, contradiction=0.0),
        FakeRetrieval([handbook("Withdrawal is covered by the academic regulations.")]),
    ).assess(question="what is the wifi password?", answer="The lab WiFi password is CS-Labs-2023.")

    assert not assessment.held
    assert assessment.suggested_promote


@pytest.mark.asyncio
async def test_only_handbook_sources_are_compared_against():
    """A previously promoted answer must not be able to veto a new one — that would
    let the KB's own history block its corrections."""
    assessment = await controller(
        FakeJudge(),
        FakeRetrieval([promoted("An older advisor answer."), handbook("A real handbook rule.")]),
    ).assess(question="q", answer="a")

    assert len(assessment.compared_against) == 1
    assert "handbook rule" in assessment.compared_against[0]


# -------------------------------------------------------------- failure modes


@pytest.mark.asyncio
async def test_a_broken_generalizability_judge_fails_towards_keeping_knowledge():
    """Fail SAFE, not closed. An unavailable judge must not silently un-tick the box:
    the failure would look exactly like an answer nobody chose to promote."""
    assessment = await controller(FakeJudge(raise_on={"generalizability"})).assess(
        question="q", answer="a"
    )

    assert assessment.generalizable
    assert assessment.suggested_promote


@pytest.mark.asyncio
async def test_a_broken_contradiction_judge_does_not_manufacture_a_hold():
    """A hold is a veto a human has to clear. Inventing one during an outage would
    block real answers for a reason nobody can act on."""
    assessment = await controller(FakeJudge(raise_on={"contradiction"})).assess(
        question="q", answer="a"
    )

    assert not assessment.contradicts_handbook
    assert not assessment.held


@pytest.mark.asyncio
async def test_a_retrieval_failure_degrades_to_no_hold():
    assessment = await controller(FakeJudge(), FakeRetrieval(fail=True)).assess(
        question="q", answer="a"
    )

    assert assessment.compared_against == []
    assert not assessment.held


@pytest.mark.asyncio
async def test_both_checks_run():
    judge = FakeJudge()
    await controller(judge).assess(question="q", answer="a")

    assert sorted(judge.calls) == ["contradiction", "generalizability"]


# ------------------------------------------------------------- the assessment


def test_hold_reason_is_only_present_when_held():
    clean = PromotionAssessment(
        generalizable=True,
        generalizability_score=0.9,
        generalizability_reason="general",
        contradicts_handbook=False,
        contradiction_score=0.1,
        contradiction_reason="no conflict",
    )
    assert clean.hold_reason() is None
    assert clean.suggested_promote

    conflicted = clean.model_copy(
        update={
            "contradicts_handbook": True,
            "contradiction_reason": "says six weeks, handbook says two",
        }
    )
    assert conflicted.held
    assert conflicted.hold_reason() == "says six weeks, handbook says two"
    # A held answer is never suggested, however general it is.
    assert not conflicted.suggested_promote
