"""The lifecycle moves the advisor dashboard is built on.

Before these, only claim and resolve were reachable over HTTP: 3 of the state
machine's 11 legal transitions. The dashboard is a queue with action buttons, so the
rest were buttons with nothing to call.

The tests that matter most here are the knowledge-base ones. Three of these four
transitions touch whether a promoted answer stays in pgvector, and every way of
getting that wrong is silent — the loop keeps working and the knowledge quietly
drains away.
"""

import pytest

from customer_support.controllers.EscalationController import EscalationController
from customer_support.models.enums.TicketStatusEnum import (
    ConcurrentTicketUpdate,
    InvalidTicketTransition,
    TicketStatus,
)


class FakeTicket:
    def __init__(self, status, promoted_to_kb=False, advisor_answer=None):
        self.ticket_id = 1
        self.thread_id = "t-1"
        self.status = status.value
        self.question = "how long to withdraw?"
        self.advisor_answer = advisor_answer
        self.department = "CS"
        self.promoted_to_kb = promoted_to_kb
        self.promoted_at = None


class FakeTicketModel:
    """Mimics apply_transition's contract: optimistic WHERE on from_status, and the
    fields dict applied atomically with the status."""

    def __init__(self, ticket):
        self.ticket = ticket
        self.transitions = []

    async def get_ticket(self, ticket_id):
        return self.ticket

    async def apply_transition(
        self, ticket_id, from_status, to_status, actor, note=None, fields=None
    ):
        if self.ticket.status != from_status.value:
            raise ConcurrentTicketUpdate(ticket_id, from_status)
        self.transitions.append((from_status, to_status, actor, note, fields or {}))
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
    def embed_text(self, text, document_type=None):
        return [[0.1, 0.2, 0.3]]


def controller(ticket):
    model = FakeTicketModel(ticket)
    vectordb = FakeVectorDB()
    ctrl = EscalationController(
        ticket_model=model,
        vectordb_client=vectordb,
        embedding_client=FakeEmbedding(),
        collection_name="collection_3",
    )
    return ctrl, model, vectordb


# --------------------------------------------------------------------- release


@pytest.mark.asyncio
async def test_release_returns_a_claimed_ticket_to_the_queue():
    """Claim used to be a one-way door: an advisor who claimed and walked away left
    the ticket in under_review forever, missing from the pending queue."""
    ctrl, model, _ = controller(FakeTicket(TicketStatus.UNDER_REVIEW))

    ticket = await ctrl.release(ticket_id=1, actor="advisor-7", note="handing over")

    assert ticket.status == TicketStatus.PENDING.value
    assert model.transitions[0][2] == "advisor-7"


@pytest.mark.asyncio
async def test_release_refuses_a_ticket_nobody_claimed():
    ctrl, _, _ = controller(FakeTicket(TicketStatus.PENDING))

    with pytest.raises(InvalidTicketTransition):
        await ctrl.release(ticket_id=1, actor="advisor-7")


# ---------------------------------------------------------------------- reject


@pytest.mark.asyncio
async def test_reject_requires_a_reason():
    """The status history is the only record of the decision."""
    ctrl, _, _ = controller(FakeTicket(TicketStatus.PENDING))

    with pytest.raises(ValueError):
        await ctrl.reject(ticket_id=1, actor="advisor-7", reason="   ")


@pytest.mark.asyncio
async def test_reject_records_the_reason_and_leaves_no_vector():
    ctrl, model, vectordb = controller(FakeTicket(TicketStatus.PENDING))

    await ctrl.reject(ticket_id=1, actor="advisor-7", reason="not a handbook question")

    assert model.transitions[0][1] is TicketStatus.REJECTED
    assert model.transitions[0][3] == "not a handbook question"
    # The repair path: reopen's sync is best-effort, so a vector can outlive the flag
    # that justified it. Rejection is the last state that can clean that up.
    assert vectordb.deleted == [{"ticket_id": "1"}]
    assert vectordb.inserted == []


# ----------------------------------------------------------------------- close


@pytest.mark.asyncio
async def test_close_keeps_the_promoted_answer_in_the_knowledge_base():
    """The trap this whole file was worth writing for.

    Indexing used to require status == RESOLVED exactly. Closing a ticket is the
    HAPPY ending — the answer stood — but it moves the status off RESOLVED, so the
    next sync from anywhere would have deleted the vector. The learning loop would
    have worked perfectly right up until an advisor ticked "this went well".
    """
    ctrl, _, vectordb = controller(
        FakeTicket(TicketStatus.RESOLVED, promoted_to_kb=True, advisor_answer="Two weeks.")
    )

    ticket = await ctrl.close(ticket_id=1, actor="advisor-7")
    assert ticket.status == TicketStatus.CLOSED.value

    # close itself must not touch the index...
    assert vectordb.deleted == []

    # ...and a later sync must still consider a closed ticket indexable.
    await ctrl.sync_ticket_to_vectors(ticket_id=1)
    assert len(vectordb.inserted) == 1
    assert vectordb.inserted[0]["metadata"]["ticket_id"] == "1"


@pytest.mark.asyncio
async def test_close_refuses_an_unresolved_ticket():
    ctrl, _, _ = controller(FakeTicket(TicketStatus.UNDER_REVIEW))

    with pytest.raises(InvalidTicketTransition):
        await ctrl.close(ticket_id=1, actor="advisor-7")


# ---------------------------------------------------------------------- reopen


@pytest.mark.asyncio
async def test_reopen_un_promotes_and_removes_the_vector():
    """A reopened answer is suspect. Leaving the vector behind is the drift
    Section 1 forbids."""
    ctrl, _, vectordb = controller(
        FakeTicket(TicketStatus.RESOLVED, promoted_to_kb=True, advisor_answer="Six weeks.")
    )

    ticket = await ctrl.reopen(ticket_id=1, actor="student-3", note="that wasn't my question")

    assert ticket.status == TicketStatus.REOPENED.value
    assert ticket.promoted_to_kb is False
    assert vectordb.deleted == [{"ticket_id": "1"}]
    assert vectordb.inserted == []


@pytest.mark.asyncio
async def test_a_closed_ticket_can_be_reopened():
    ctrl, _, _ = controller(
        FakeTicket(TicketStatus.CLOSED, promoted_to_kb=True, advisor_answer="Two weeks.")
    )

    ticket = await ctrl.reopen(ticket_id=1, actor="student-3")
    assert ticket.status == TicketStatus.REOPENED.value


# ------------------------------------------------------------------ not found


@pytest.mark.asyncio
async def test_a_missing_ticket_raises_lookup_error_not_something_else():
    """The router maps LookupError to 404. Every lifecycle method has to agree."""
    ctrl, model, _ = controller(FakeTicket(TicketStatus.PENDING))
    model.ticket = None

    for call in (
        ctrl.release(1, "a"),
        ctrl.reject(1, "a", "reason"),
        ctrl.close(1, "a"),
        ctrl.reopen(1, "a"),
    ):
        with pytest.raises(LookupError):
            await call


# ------------------------------------------------- how failures reach the client


class TestLifecycleErrorMapping:
    """The router maps controller failures onto status codes in one place.

    Worth testing directly: before the shared mapping, each route repeated its own
    try/except, and none of them caught the lost-update case — so two advisors
    double-clicking Claim got a 500 for a race the code already handles correctly.
    """

    @staticmethod
    def _status_for(exc):
        from fastapi import HTTPException

        from customer_support.routers.escalation import _lifecycle_errors

        try:
            with _lifecycle_errors():
                raise exc
        except HTTPException as mapped:
            return mapped.status_code
        raise AssertionError(f"{exc!r} was not mapped to an HTTPException")

    def test_missing_ticket_is_404(self):
        assert self._status_for(LookupError("ticket 9 not found")) == 404

    def test_an_illegal_transition_is_409_not_400(self):
        """The request was well-formed; the ticket's state refused it."""
        exc = InvalidTicketTransition(1, TicketStatus.PENDING, TicketStatus.CLOSED)
        assert self._status_for(exc) == 409

    def test_losing_the_race_is_409_not_500(self):
        """Two advisors clicking the same button is ordinary dashboard traffic."""
        assert self._status_for(ConcurrentTicketUpdate(1, TicketStatus.PENDING)) == 409

    def test_a_broken_invariant_is_400(self):
        assert self._status_for(ValueError("a rejection needs a reason")) == 400

    def test_a_real_fault_is_not_swallowed(self):
        """Anything unrecognised must keep propagating and become a 500 — the mapping
        exists to translate known failures, not to hide bugs."""
        from customer_support.routers.escalation import _lifecycle_errors

        with pytest.raises(RuntimeError):
            with _lifecycle_errors():
                raise RuntimeError("the database is on fire")
