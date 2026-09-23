"""The API contract for chat. Separate from db_schemas on purpose: what a client
sends and how data is stored change for different reasons."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    student_id: str = Field(min_length=1, max_length=64)
    # Absent on the first message; the client echoes it back afterwards.
    thread_id: str | None = Field(default=None, max_length=64)
    department: str | None = Field(default=None, pattern="^(CS|IS)$")
    # An explicit per-request override — the highest-priority language signal, above
    # the stored profile and above Accept-Language. Not pattern-locked to the
    # supported set: which locales exist is a deployment fact, and the parser falls
    # back (loudly) rather than rejecting a request over it.
    language: str | None = Field(default=None, max_length=16)


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    escalated: bool
    ticket_id: int | None = None
    # Shown to the student as "why", never as an answer.
    escalation_reason: str | None = None