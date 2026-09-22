"""The single typed object that flows through the graph.

Rule (Section 4): if a node needs data, it comes from here. Nodes never invent keys.
``total=False`` lets a node return a partial dict; LangGraph merges it into the state.

Two kinds of field live here, and the difference matters:

**Run-scoped** — ``question``, ``answer``, ``escalate``, ``ticket_id`` and friends
describe THIS run. The checkpointer merges runs into one thread, so a field written
on only one code path survives into later runs and misreports a previous turn's
outcome as this one's. The router therefore resets them on every invoke.

**Durable** — ``messages`` is the thread's transcript and must accumulate, because a
thread outlives a single run: the student asks, the run ends, and an advisor's
resolution attaches to the same ``thread_id`` hours later. It carries a reducer so
every write appends instead of replacing.
"""

import operator
from typing import Annotated, TypedDict

from customer_support.models.db_schemas import RetrievedDocument
from customer_support.models.graph.conversation import ConversationMessage


class GraphState(TypedDict, total=False):
    # --- input (set by the router before the run) ---
    question: str
    student_id: str
    thread_id: str
    department: str | None  # "CS" | "IS" | None
    # Resolved ONCE by the router from the request, the profile and Accept-Language.
    # Nodes read it; nothing downstream re-negotiates.
    language: str

    # --- the durable transcript ---
    # operator.add, so `{"messages": [msg]}` from any node or from an advisor's
    # out-of-band state update APPENDS. This is the one field a new run must not
    # reset, and the reason an advisor's answer no longer overwrites the student's.
    messages: Annotated[list[ConversationMessage], operator.add]

    # --- retrieval ---
    retrieved_chunks: list[RetrievedDocument]

    # --- gates (Section 2) ---
    gate_scores: dict[str, float]
    failed_gate: str | None  # GateEnum value; None while everything passes
    gate_reason: str | None  # advisor-facing reason; becomes the escalation summary
    escalate: bool

    # --- output (run-scoped: the latest turn, not the history) ---
    answer: str | None
    ticket_id: int | None
    # Written by the advisor's resolution, which is a SECOND write over this same
    # thread hours later — not part of the original run (Section 3).
    resolved_by: str | None
