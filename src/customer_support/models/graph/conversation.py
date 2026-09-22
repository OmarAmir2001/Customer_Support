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
        return cls(
            role=MessageRole.ADVISOR, content=content, ticket_id=ticket_id, author=author
        )
