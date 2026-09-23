from enum import Enum


class MessageRole(str, Enum):
    """Who spoke a turn in a conversation.

    Distinct from ``TicketActor``, which records who moved a TICKET through its
    lifecycle. The two overlap in wording and answer different questions: an advisor
    claiming a ticket is a TicketActor event with no conversation turn, and a student
    asking a question is a turn that touches no ticket.
    """

    STUDENT = "student"
    ASSISTANT = "assistant"
    #: A human academic advisor, answering hours or days later in a separate request.
    ADVISOR = "advisor"
