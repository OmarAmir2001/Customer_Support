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


class ClaimRequest(BaseModel):
    advisor_id: str = Field(min_length=1, max_length=64)


class ResolveRequest(BaseModel):
    advisor_id: str = Field(min_length=1, max_length=64)
    advisor_answer: str = Field(min_length=10, max_length=5000)
    # Section 5: resolving always delivers; promoting is a separate decision the
    # advisor makes. The generalizability judge only pre-fills this box.
    promote_to_kb: bool = False

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
