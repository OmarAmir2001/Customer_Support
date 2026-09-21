from enum import Enum


class TicketStatus(str, Enum):
    """Section 3 lifecycle. The ``str`` mixin makes it serialise straight to JSON and
    to a Postgres varchar without a converter."""

    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    RESOLVED = "resolved"
    REJECTED = "rejected"
    REOPENED = "reopened"
    CLOSED = "closed"
    DUPLICATE = "duplicate"


class TicketActor(str, Enum):
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


def assert_transition_allowed(
    ticket_id: int, current: TicketStatus, requested: TicketStatus
) -> None:
    if requested not in ALLOWED_TRANSITIONS[current]:
        raise InvalidTicketTransition(ticket_id, current, requested)