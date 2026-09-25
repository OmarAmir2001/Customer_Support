"""One turn in a thread's conversation.

A thread spans MORE than one request: the student asks, the run ends, and hours
later an advisor's resolution attaches to the same ``thread_id`` (Section 3). A
single ``answer`` slot in the state cannot represent that — the advisor's answer
overwrites whatever the student last received. A list of turns can, which is why
``GraphState.messages`` accumulates through a reducer instead of being replaced.

Frozen on purpose: a turn is a record of something that was said. Nothing should
edit history in place; corrections are new turns, or a ticket edit plus a re-sync.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from customer_support.models.enums.MessageRoleEnum import MessageRole

#: How many turns a thread keeps in its checkpoint.
#:
#: A module constant rather than a Setting because the reducer is referenced in the
#: state's type annotation, which is evaluated at import time — before any Settings
#: object exists. Prompt-level bounds ARE configurable; this is the storage backstop.
#:
#: The bound matters: every turn is serialised into the checkpoint on every write, so
#: an unbounded list makes each run's write grow with the thread's whole history.
#: 40 turns is roughly 20 exchanges — far more than any real handbook conversation,
#: while keeping the row small.
#:
#: Trimming loses the oldest turns permanently. That is acceptable because the
#: checkpoint is conversation WORKING memory, not the audit trail: an escalated
#: question and the answer it received are durable in the tickets table, and
#: identity facts are durable in the student profile.
CONVERSATION_MAX_STORED_TURNS = 40


class ConversationMessage(BaseModel):
    model_config = {"frozen": True}

    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    #: Set on an advisor turn: which ticket this answer resolved, and who wrote it.
    #: Carrying provenance on the turn itself means the transcript explains where an
    #: answer came from without a second lookup.
    ticket_id: int | None = None
    author: str | None = None

    @classmethod
    def student(cls, content: str) -> "ConversationMessage":
        return cls(role=MessageRole.STUDENT, content=content)

    @classmethod
    def assistant(cls, content: str) -> "ConversationMessage":
        return cls(role=MessageRole.ASSISTANT, content=content)

    @classmethod
    def advisor(
        cls, content: str, ticket_id: int, author: str | None = None
    ) -> "ConversationMessage":
        return cls(role=MessageRole.ADVISOR, content=content, ticket_id=ticket_id, author=author)


def append_and_trim(
    existing: list[ConversationMessage] | None,
    new: list[ConversationMessage] | None,
) -> list[ConversationMessage]:
    """The ``messages`` reducer: append, then keep only the most recent turns.

    Plain ``operator.add`` was correct about appending and silently unbounded — a
    long-lived thread's checkpoint would grow forever, and every run rewrites the
    whole list. Trimming here bounds storage at the one place every write goes
    through, including an advisor's out-of-band ``aupdate_state``.

    Oldest-first eviction, because recent turns are what a follow-up question needs
    ("what about for IS students?" refers to the turn before it, not to turn 3).
    """
    combined = list(existing or []) + list(new or [])

    if len(combined) <= CONVERSATION_MAX_STORED_TURNS:
        return combined
    return combined[-CONVERSATION_MAX_STORED_TURNS:]
