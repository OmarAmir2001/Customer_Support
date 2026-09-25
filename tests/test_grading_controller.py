"""Judge tests with a fake LLM client.

This is the payoff of keeping logic in controllers: no graph is involved, so a
judge can be tested against fixed inputs in milliseconds.
"""

import pytest

from customer_support.controllers.GradingController import GradingController
from customer_support.models.db_schemas import RetrievedDocument
from customer_support.models.enums.GateEnum import GateEnum
from customer_support.stores.llm.templates import TemplateParser


class FakeLLM:
    """Returns a canned JSON string; records the prompts it was given."""

    def __init__(self, response: str | None):
        self.response = response
        self.prompts: list[str] = []

    def generate_json(self, prompt, system_prompt=None, max_output_tokens=None, temperature=None):
        self.prompts.append(prompt)
        return self.response


class FakeSettings:
    GATE_CONTEXT_RELEVANCE_THRESHOLD = 0.5
    GATE_FAITHFULNESS_THRESHOLD = 0.8
    GATE_ANSWER_RELEVANCE_THRESHOLD = 0.7
    JUDGE_MAX_OUTPUT_TOKENS = 300
    JUDGE_TEMPERATURE = 0.0
    ASSETS_DIR = "assets"


TEMPLATES = TemplateParser(primary_language="en", default_language="en")


def chunk(text: str, score: float = 0.9) -> RetrievedDocument:
    return RetrievedDocument(text=text, score=score, metadata={"source": "CS_2023"})


@pytest.mark.asyncio
async def test_empty_retrieval_fails_without_calling_the_llm():
    llm = FakeLLM(response=None)
    controller = GradingController(
        generation_client=llm, templates=TEMPLATES, settings=FakeSettings()
    )

    result = await controller.check_context_relevance(question="anything", chunks=[])

    assert result.passed is False
    assert result.gate == GateEnum.CONTEXT_RELEVANCE.value
    assert llm.prompts == []  # no chunks means no judgement is needed


@pytest.mark.asyncio
async def test_grounded_answer_passes_faithfulness():
    llm = FakeLLM('{"score": 1.0, "reason": "every claim appears in the excerpts"}')
    controller = GradingController(
        generation_client=llm, templates=TEMPLATES, settings=FakeSettings()
    )

    result = await controller.check_faithfulness(
        answer="The withdrawal window is two weeks.",
        chunks=[chunk("Students may withdraw within two weeks of the term start.")],
    )

    assert result.passed is True
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_unsupported_claim_fails_faithfulness():
    llm = FakeLLM('{"score": 0.5, "reason": "the claim about a fee is not in the excerpts"}')
    controller = GradingController(
        generation_client=llm, templates=TEMPLATES, settings=FakeSettings()
    )

    result = await controller.check_faithfulness(
        answer="You may withdraw within two weeks, and the fee is 500 EGP.",
        chunks=[chunk("Students may withdraw within two weeks of the term start.")],
    )

    assert result.passed is False
    assert "fee" in result.reason


@pytest.mark.asyncio
async def test_broken_judge_fails_closed():
    """An unparseable judge must escalate, never wave the answer through.

    Failing open would let unverified answers reach students exactly when the
    verification machinery is broken.
    """
    llm = FakeLLM("I think the answer looks pretty good to me!")
    controller = GradingController(
        generation_client=llm, templates=TEMPLATES, settings=FakeSettings()
    )

    result = await controller.check_answer_relevance(question="q", answer="a")

    assert result.passed is False
    assert result.score == 0.0
    assert len(llm.prompts) == 2  # one retry before giving up
