"""Escalation ticket tables (Section 3).

Two tables, not one with a JSONB history column: the stale-ticket scanner asks
"time since the last transition", which is an indexed query on a real table and an
awkward one inside JSONB.
"""

import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from customer_support.models.enums.TicketStatusEnum import TicketStatus

from .customer_support_base import SQLAlchemyBase


class Ticket(SQLAlchemyBase):
    __tablename__ = "tickets"

    ticket_id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)

    # The load-bearing link to the checkpointer: the advisor resolution is a NEW run
    # over this same thread, so without it the conversation cannot be reconnected.
    thread_id = Column(String, nullable=False)

    student_id = Column(String, nullable=False)
    department = Column(String, nullable=True)
    question = Column(Text, nullable=False)

    # Which gate tripped, and why: this is the advisor's escalation summary.
    failed_gate = Column(String, nullable=False)
    gate_reason = Column(Text, nullable=False)
    gate_scores = Column(JSONB, nullable=True)

    # What the bot saw. The advisor needs it, and it is the audit trail.
    retrieved_chunks = Column(JSONB, nullable=True)

    status = Column(String, nullable=False, default=TicketStatus.PENDING.value)

    advisor_answer = Column(Text, nullable=True)
    resolved_by = Column(String, nullable=True)

    # Section 5: resolving and promoting are two separate acts.
    promoted_to_kb = Column(Boolean, nullable=False, default=False)
    promoted_at = Column(DateTime(timezone=True), nullable=True)

    duplicate_of = Column(Integer, ForeignKey("tickets.ticket_id"), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    status_history = relationship(
        "TicketStatusHistory",
        back_populates="ticket",
        order_by="TicketStatusHistory.created_at",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_ticket_status", "status"),
        Index("ix_ticket_thread_id", "thread_id"),
        Index("ix_ticket_student_id", "student_id"),
        # The advisor dashboard's main query: pending tickets for my department,
        # oldest first. One composite index serves it.
        Index("ix_ticket_status_department_created", "status", "department", "created_at"),
    )


class TicketStatusHistory(SQLAlchemyBase):
    __tablename__ = "ticket_status_history"

    history_id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(Integer, ForeignKey("tickets.ticket_id"), nullable=False)

    from_status = Column(String, nullable=True)  # null for the first row
    to_status = Column(String, nullable=False)
    actor = Column(String, nullable=False)  # TicketActor value or an advisor id
    note = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    ticket = relationship("Ticket", back_populates="status_history")

    __table_args__ = (
        # The scanner's query: latest transition per ticket.
        Index("ix_ticket_history_ticket_created", "ticket_id", "created_at"),
    )