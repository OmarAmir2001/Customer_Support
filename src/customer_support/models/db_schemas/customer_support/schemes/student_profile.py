"""The long-term student profile (Section 6).

ONE row per student, patched in place — not a collection of memory objects. A
profile is bounded and structured (a fixed schema), which is exactly what a single
patched document is best at, and it gets read on every turn, so one clean row beats
search-and-assemble. The document-collection shape trades that for recall the
project does not need, and pays for it with over-insertion: "student is confused
about registration" stored again every session until retrieval over memory is noise.

Identity, not history. Individual questions are the checkpointer's job (thread
scoped); promoting them here is how a profile fills with junk.
"""

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, func

from .customer_support_base import SQLAlchemyBase


class StudentProfile(SQLAlchemyBase):
    __tablename__ = "student_profiles"

    profile_id = Column(Integer, primary_key=True, autoincrement=True)

    # The natural key. Unique because a student has exactly one profile — the schema
    # enforces the "single patched document" decision rather than trusting callers.
    student_id = Column(String, nullable=False, unique=True)

    # Section 6's persist-list. Every one is a durable identity fact; all nullable
    # because a profile is built up over many turns and is never complete at first
    # contact.
    name = Column(String, nullable=True)
    department = Column(String, nullable=True)  # "CS" | "IS"
    gpa = Column(Float, nullable=True)
    preferred_language = Column(String, nullable=True)

    # Observability for the frequency gate: how often extraction actually runs, and
    # when it last changed anything. Without these, "we only extract when a fact is
    # plausibly present" is a claim nobody can check.
    last_extracted_at = Column(DateTime(timezone=True), nullable=True)
    extraction_count = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    __table_args__ = (Index("ix_student_profile_student_id", "student_id"),)
