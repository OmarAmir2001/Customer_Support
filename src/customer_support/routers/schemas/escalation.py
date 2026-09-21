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


class ClaimRequest(BaseModel):
    advisor_id: str = Field(min_length=1, max_length=64)


class ResolveRequest(BaseModel):
    advisor_id: str = Field(min_length=1, max_length=64)
    advisor_answer: str = Field(min_length=10, max_length=5000)
    # Section 5: resolving always delivers; promoting is a separate decision the
    # advisor makes. The generalizability judge only pre-fills this box.
    promote_to_kb: bool = False