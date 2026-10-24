from datetime import datetime

from pydantic import BaseModel, Field

from customer_support.models.enums.TicketStatusEnum import TicketStatus


class TicketSummary(BaseModel):
    ticket_id: int
    thread_id: str
    student_id: str
    department: str | None
    question: str
    failed_gate: str
    gate_reason: str
    status: TicketStatus
    created_at: datetime

    model_config = {"from_attributes": True}


class TicketDetail(TicketSummary):
    gate_scores: dict | None = None
    retrieved_chunks: list | None = None
    advisor_answer: str | None = None
    resolved_by: str | None = None
    promoted_to_kb: bool = False
    # Set when the contradiction check blocked promotion (Section 5's one veto).
    promotion_held: bool = False
    promotion_hold_reason: str | None = None


#: `advisor_id` is GONE from every request body below, and that is the fix.
#:
#: It used to be a field, so the actor was whatever the caller typed — anyone who
#: could reach the port could resolve a ticket as anybody, and
#: `ticket_status_history` would record the impersonation as fact. The actor now comes
#: from the API key that authenticated the request (`helpers/auth.py`), which is the
#: only way the audit trail means anything.
#:
#: Claim needs no body at all now: the actor was the only thing in it.


class ResolveRequest(BaseModel):
    advisor_answer: str = Field(min_length=10, max_length=5000)
    # Section 5: resolving always delivers; promoting is a separate decision the
    # advisor makes. The generalizability judge only pre-fills this box.
    promote_to_kb: bool = False


class ReleaseRequest(BaseModel):
    """Hand a claimed ticket back to the queue."""

    note: str | None = Field(default=None, max_length=1000)


class RejectRequest(BaseModel):
    """Refuse a question outright."""

    # Required, unlike the other notes. The status history is the only place this
    # decision is recorded, and a rejection with no reason is unreviewable later.
    reason: str = Field(min_length=3, max_length=1000)


class CloseRequest(BaseModel):
    """Finish a resolved ticket. The promoted answer stays in the knowledge base."""

    note: str | None = Field(default=None, max_length=1000)


class ReopenRequest(BaseModel):
    """Send a finished ticket back for another look.

    This used to carry `actor` rather than `advisor_id`, because a reopen is the one
    lifecycle move a student could in principle trigger. That path does not exist:
    there is no student authentication, so a student-initiated reopen would be an
    unauthenticated write to someone else's ticket. The actor comes from the advisor
    key like every other move, and a student-facing reopen needs student auth first.
    """

    note: str | None = Field(default=None, max_length=1000)


class PromotionAssessmentRequest(BaseModel):
    """The draft the advisor is typing, before it is saved.

    Sent as a body rather than read from the ticket because this is called WHILE the
    advisor writes — there is nothing persisted to assess yet.
    """

    advisor_answer: str = Field(min_length=10, max_length=5000)


class PromotionAssessmentResponse(BaseModel):
    """What the dashboard needs to render the checkbox and explain itself."""

    #: What to default the "add to knowledge base" box to. Advice only.
    suggested_promote: bool
    #: True when promotion is blocked regardless of the box. The advisor cannot
    #: tick past a contradiction; the handbook gets reviewed instead.
    held: bool

    generalizable: bool
    generalizability_score: float
    generalizability_reason: str

    contradicts_handbook: bool
    contradiction_score: float
    contradiction_reason: str

    #: The handbook excerpts the contradiction verdict was based on, so an advisor
    #: can judge the judge rather than trusting a bare score.
    compared_against: list[str] = Field(default_factory=list)
