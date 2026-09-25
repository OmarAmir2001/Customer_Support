from enum import StrEnum


class TicketStatus(StrEnum):
    """Section 3 lifecycle.

    ``StrEnum`` rather than ``(str, Enum)``: both compare equal to their raw string,
    but only StrEnum also FORMATS as it — ``f"{TicketStatus.PENDING}"`` is "pending"
    instead of "TicketStatus.PENDING". So a member serialises straight to JSON and to
    a Postgres varchar whether or not the caller remembered ``.value``.
    """

    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    RESOLVED = "resolved"
    REJECTED = "rejected"
    REOPENED = "reopened"
    CLOSED = "closed"
    DUPLICATE = "duplicate"


class TicketActor(StrEnum):
    SYSTEM = "system"
    ADVISOR = "advisor"
    STUDENT = "student"
    SCANNER = "scanner"  # the phase-2 stale-ticket job


# The state machine, written once. EscalationController is the only caller.
ALLOWED_TRANSITIONS: dict[TicketStatus, set[TicketStatus]] = {
    TicketStatus.PENDING: {
        TicketStatus.UNDER_REVIEW,
        TicketStatus.REJECTED,
        TicketStatus.DUPLICATE,
    },
    TicketStatus.UNDER_REVIEW: {TicketStatus.RESOLVED, TicketStatus.REJECTED, TicketStatus.PENDING},
    TicketStatus.RESOLVED: {TicketStatus.CLOSED, TicketStatus.REOPENED},
    # reopened goes back to under_review, never to pending: it already has history.
    TicketStatus.REOPENED: {TicketStatus.UNDER_REVIEW},
    TicketStatus.REJECTED: {TicketStatus.REOPENED},
    TicketStatus.CLOSED: {TicketStatus.REOPENED},
    TicketStatus.DUPLICATE: set(),
}


class InvalidTicketTransition(Exception):
    """Raised when code attempts a transition the lifecycle forbids.

    Fail loudly: a silent no-op would leave a ticket in a state the dashboard shows
    but the scanner never picks up.
    """

    def __init__(self, ticket_id: int, current: TicketStatus, requested: TicketStatus):
        super().__init__(
            f"ticket {ticket_id}: cannot move from {current.value} to {requested.value}"
        )
        self.ticket_id = ticket_id
        self.current = current
        self.requested = requested


class ConcurrentTicketUpdate(Exception):
    """Raised when a ticket moved between the read and the write.

    Distinct from InvalidTicketTransition, and the distinction is the whole point:
    there, the transition was illegal for the state the ticket is in. Here it was
    perfectly legal for the state we READ, and another writer got there first.

    Two advisors clicking Claim on the same ticket is ordinary dashboard traffic, not
    a server fault, so it has to surface as a 409 and a refresh rather than a 500.
    """

    def __init__(self, ticket_id: int, expected: TicketStatus):
        super().__init__(
            f"ticket {ticket_id} was no longer in status {expected.value}; "
            "another process changed it first"
        )
        self.ticket_id = ticket_id
        self.expected = expected


def assert_transition_allowed(
    ticket_id: int, current: TicketStatus, requested: TicketStatus
) -> None:
    if requested not in ALLOWED_TRANSITIONS[current]:
        raise InvalidTicketTransition(ticket_id, current, requested)