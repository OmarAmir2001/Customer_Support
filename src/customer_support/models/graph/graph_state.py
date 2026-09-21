"""The single typed object that flows through the graph.

Rule (Section 4): if a node needs data, it comes from here. Nodes never invent keys.
``total=False`` lets a node return a partial dict; LangGraph merges it into the state.
"""

from typing import TypedDict

from customer_support.models.db_schemas import RetrievedDocument


class GraphState(TypedDict, total=False):
    # --- input (set by the router before the run) ---
    question: str
    student_id: str
    thread_id: str
    department: str | None  # "CS" | "IS" | None

    # --- retrieval ---
    retrieved_chunks: list[RetrievedDocument]

    # --- gates (Section 2) ---
    gate_scores: dict[str, float]
    failed_gate: str | None  # GateEnum value; None while everything passes
    gate_reason: str | None  # advisor-facing reason; becomes the escalation summary
    escalate: bool

    # --- output ---
    answer: str | None
    ticket_id: int | None
    # Written by the advisor's resolution, which is a SECOND write over this same
    # thread hours later — not part of the original run (Section 3).
    resolved_by: str | None