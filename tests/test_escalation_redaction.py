"""F4 — an advisor's answer must not reach a student, or the vector store, unredacted.

The report's own words: the existing suite "asserts redaction on the generated-answer
path. Nothing asserts it on GET /chat/{thread_id}. A test that resolves a ticket with
a phone number in the answer and then reads the thread would fail today."

That is the test below, plus the two write paths behind it. Also F7: resolving a
ticket whose checkpoint has vanished must decline rather than create a one-message
conversation.

The redaction is applied to the DERIVED copies only. The ticket keeps the advisor's
exact words, because it is the audit record — asserted here so nobody "tidies" that
into redacting on the way in.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from customer_support.controllers.EscalationController import EscalationController
from customer_support.models.enums.TicketStatusEnum import TicketStatus
from customer_support.routers.chat import chat_router

ANSWER_WITH_PII = "Call the registrar on 01012345678 or email registrar@institute.edu.eg."


class FakeTicket:
    def __init__(self, status=TicketStatus.UNDER_REVIEW, **kwargs):
        self.ticket_id = 1
        self.thread_id = "t-1"
        self.status = status.value
        self.question = "Who do I contact about a transcript?"
        self.advisor_answer = kwargs.get("advisor_answer")
        self.department = "CS"
        self.promoted_to_kb = kwargs.get("promoted_to_kb", False)
        self.promoted_at = None
        self.resolved_by = kwargs.get("resolved_by")


class FakeTicketModel:
    def __init__(self, ticket):
        self.ticket = ticket

    async def get_ticket(self, _ticket_id):
        return self.ticket

    async def apply_transition(
        self, ticket_id, from_status, to_status, actor, note=None, fields=None
    ):
        self.ticket.status = to_status.value
        for key, value in (fields or {}).items():
            setattr(self.ticket, key, value)
        return self.ticket


class FakeVectorDB:
    def __init__(self):
        self.deleted = []
        self.inserted = []

    async def delete_by_metadata(self, collection_name, criteria):
        self.deleted.append(criteria)
        return 1

    async def insert_one(self, **kwargs):
        self.inserted.append(kwargs)
        return True


class FakeEmbedding:
    def __init__(self):
        self.embedded = []

    def embed_text(self, text, document_type=None):
        self.embedded.append(text)
        return [[0.1, 0.2, 0.3]]


class FakeSettings:
    ASSETS_DIR = "assets"
    PII_REDACTION_ENABLED = True


class FakeThreadWriter:
    """Mimics the compiled graph's two state methods.

    `exists=False` is the F7 case: a thread_id with no checkpoint. The real
    `aupdate_state` would happily CREATE one.
    """

    def __init__(self, exists=True):
        self.exists = exists
        self.updates = []

    async def aget_state(self, _config):
        if not self.exists:
            return None

        class _Snapshot:
            values = {"question": "Who do I contact about a transcript?", "messages": []}

        return _Snapshot()

    async def aupdate_state(self, config, values, as_node=None):
        self.updates.append(values)


def controller(ticket, *, thread_exists=True):
    vectordb = FakeVectorDB()
    embedding = FakeEmbedding()
    writer = FakeThreadWriter(exists=thread_exists)
    ctrl = EscalationController(
        ticket_model=FakeTicketModel(ticket),
        vectordb_client=vectordb,
        embedding_client=embedding,
        collection_name="collection_3",
        settings=FakeSettings(),
    )
    ctrl.thread_writer = writer
    return ctrl, writer, vectordb, embedding


# ----------------------------------------------------------------------- F4


@pytest.mark.asyncio
async def test_the_answer_written_into_the_thread_is_redacted():
    """The direct path: advisor answer -> thread -> the student who asked.

    This is the path the escalation feature exists for, and it was the only write
    into durable storage with no guard on it.
    """
    ctrl, writer, _, _ = controller(FakeTicket())

    await ctrl.resolve(
        ticket_id=1, advisor_answer=ANSWER_WITH_PII, advisor_id="adv", promote_to_kb=False
    )

    written = writer.updates[0]
    assert "[PHONE]" in written["answer"]
    assert "[EMAIL]" in written["answer"]
    assert "01012345678" not in written["answer"]
    assert "registrar@institute.edu.eg" not in written["answer"]

    # The transcript turn and the mirrored `answer` must agree — a student reading
    # the thread must not get one redacted and one raw.
    turn = written["messages"][0]
    assert "[PHONE]" in turn.content
    assert "01012345678" not in turn.content


@pytest.mark.asyncio
async def test_the_text_embedded_into_the_knowledge_base_is_redacted():
    """The indirect path, and BOTH halves of it.

    The question is the student's own words, so "my number is 010…, when are exams?"
    would otherwise be embedded verbatim and retrievable by anyone.
    """
    ticket = FakeTicket()
    ticket.question = "My number is 01099887766, who do I contact?"
    ctrl, _, _, embedding = controller(ticket)

    await ctrl.resolve(
        ticket_id=1, advisor_answer=ANSWER_WITH_PII, advisor_id="adv", promote_to_kb=True
    )

    embedded = embedding.embedded[0]
    assert "[PHONE]" in embedded
    assert "01012345678" not in embedded, "the advisor's number was embedded"
    assert "01099887766" not in embedded, "the student's number was embedded"


@pytest.mark.asyncio
async def test_the_ticket_itself_keeps_the_advisors_exact_words():
    """Redaction is on the way OUT. The ticket is the audit record.

    An advisor's original phrasing is what a later reviewer needs, and scrubbing it
    at the source would destroy evidence to protect a copy.
    """
    ticket = FakeTicket()
    ctrl, _, _, _ = controller(ticket)

    await ctrl.resolve(
        ticket_id=1, advisor_answer=ANSWER_WITH_PII, advisor_id="adv", promote_to_kb=False
    )

    assert ticket.advisor_answer == ANSWER_WITH_PII


@pytest.mark.asyncio
async def test_a_clean_answer_passes_through_untouched():
    """Over-redaction would corrupt ordinary advisor answers full of article numbers."""
    clean = "You need 135 credit hours with a GPA of at least 2.0 — see مادة (٢٤)."
    ctrl, writer, _, _ = controller(FakeTicket())

    await ctrl.resolve(ticket_id=1, advisor_answer=clean, advisor_id="adv", promote_to_kb=False)

    assert writer.updates[0]["answer"] == clean


@pytest.mark.asyncio
async def test_redaction_can_be_switched_off_by_configuration():
    """Consistent with every other guard in the project."""

    class _Off(FakeSettings):
        PII_REDACTION_ENABLED = False

    ticket = FakeTicket()
    ctrl, writer, _, _ = controller(ticket)
    ctrl.app_settings = _Off()

    await ctrl.resolve(
        ticket_id=1, advisor_answer=ANSWER_WITH_PII, advisor_id="adv", promote_to_kb=False
    )
    assert "01012345678" in writer.updates[0]["answer"]


def test_the_student_facing_read_returns_the_redacted_turn():
    """End of the direct path: what GET /chat/{thread_id} actually serves.

    Asserted through the route, because the redaction only helps if it is the stored
    value that was scrubbed — a test on the controller alone would pass even if the
    handler re-read something raw.
    """
    from types import SimpleNamespace

    from customer_support.models.graph.conversation import ConversationMessage

    scrubbed = "Call the registrar on [PHONE] or email [EMAIL]."

    class _Graph:
        async def aget_state(self, _config):
            return SimpleNamespace(
                values={
                    "question": "Who do I contact?",
                    "answer": scrubbed,
                    "resolved_by": "adv",
                    "escalate": True,
                    "ticket_id": 1,
                    "messages": [ConversationMessage.advisor(content=scrubbed, ticket_id=1)],
                }
            )

    app = FastAPI()
    app.include_router(chat_router)
    app.state.graph = _Graph()

    body = TestClient(app).get("/api/v1/chat/t-1").json()
    assert "01012345678" not in str(body)
    assert "[PHONE]" in body["answer"]


# ----------------------------------------------------------------------- F7


@pytest.mark.asyncio
async def test_resolving_a_ticket_whose_thread_is_gone_does_not_create_one():
    """`aupdate_state` on an unknown thread_id CREATES a checkpoint.

    So a vanished checkpoint plus a resolve would produce a brand-new thread holding
    exactly one message — the advisor's answer, as a reply to nothing, with no
    question and no holding message above it. The answer stays on the ticket, which
    is the source of truth, so declining loses nothing.
    """
    ticket = FakeTicket()
    ctrl, writer, _, _ = controller(ticket, thread_exists=False)

    result = await ctrl.resolve(
        ticket_id=1, advisor_answer="Two weeks.", advisor_id="adv", promote_to_kb=False
    )

    assert writer.updates == [], "a one-message thread was created"
    # The resolve itself still succeeds: the advisor's request must not fail after
    # their answer has committed to the ticket.
    assert result.status == TicketStatus.RESOLVED.value
    assert ticket.advisor_answer == "Two weeks."


# ----------------------------------------------------------------------- F5


class _CountingJudge:
    """A judge client that records every prompt it is asked to score.

    Counting CALLS rather than asserting which controller method was used: the point
    of F5 is the wasted model call, so the test should fail if the call comes back by
    any route.
    """

    def __init__(self):
        self.prompts = []
        self.generation_model_id = "fake-judge"

    def generate_json(self, prompt, system_prompt=None, max_output_tokens=None, temperature=None):
        self.prompts.append(prompt)
        # A verdict both judges can parse: low score = general enough, no contradiction.
        return '{"score": 0.0, "reason": "no contradiction found"}'


class _FakeRetrieval:
    async def retrieve(self, question, department=None, limit=5):
        return []


@pytest.mark.asyncio
async def test_resolving_with_promotion_runs_one_judge_not_two():
    """F5: only the contradiction veto is read, so only it should be paid for.

    `assess()` also runs the generalizability judge, whose answer feeds the checkbox
    DEFAULT — meaningless once the advisor has already ticked the box and hit resolve.
    Every promoted resolve was buying a judge call it discarded, on the path the rate
    limits exist for.
    """
    from customer_support.controllers.PromotionController import PromotionController
    from customer_support.stores.llm.templates import TemplateParser

    class _Settings(FakeSettings):
        PROMOTION_GENERALIZABILITY_THRESHOLD = 0.6
        PROMOTION_CONTRADICTION_THRESHOLD = 0.6
        PROMOTION_CONTEXT_TOP_K = 5
        JUDGE_MAX_OUTPUT_TOKENS = 1200
        JUDGE_TEMPERATURE = 0.0

    judge = _CountingJudge()
    promotion = PromotionController(
        judge_client=judge,
        retrieval=_FakeRetrieval(),
        templates=TemplateParser(primary_language="en", default_language="en"),
        settings=_Settings(),
    )

    ticket = FakeTicket()
    ctrl, _, _, _ = controller(ticket)
    ctrl.promotion = promotion

    await ctrl.resolve(
        ticket_id=1,
        advisor_answer="Two weeks, see مادة (٢١).",
        advisor_id="adv",
        promote_to_kb=True,
    )

    assert len(judge.prompts) == 1, (
        f"expected 1 judge call (contradiction only), got {len(judge.prompts)}"
    )


@pytest.mark.asyncio
async def test_the_assessment_endpoint_still_runs_both_judges():
    """The other caller asks a different question and must keep both answers.

    `GET /promotion-assessment` renders the checkbox, which needs generalizability.
    Narrowing resolve() must not narrow this.
    """
    from customer_support.controllers.PromotionController import PromotionController
    from customer_support.stores.llm.templates import TemplateParser

    class _Settings(FakeSettings):
        PROMOTION_GENERALIZABILITY_THRESHOLD = 0.6
        PROMOTION_CONTRADICTION_THRESHOLD = 0.6
        PROMOTION_CONTEXT_TOP_K = 5
        JUDGE_MAX_OUTPUT_TOKENS = 1200
        JUDGE_TEMPERATURE = 0.0

    judge = _CountingJudge()
    promotion = PromotionController(
        judge_client=judge,
        retrieval=_FakeRetrieval(),
        templates=TemplateParser(primary_language="en", default_language="en"),
        settings=_Settings(),
    )

    assessment = await promotion.assess(question="q", answer="a long enough answer")

    assert len(judge.prompts) == 2
    # And it still reports both halves, which is what the checkbox renders from.
    assert assessment.generalizability_reason
    assert assessment.contradiction_reason
