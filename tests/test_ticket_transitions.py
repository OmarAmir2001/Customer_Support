"""Pure-logic tests: no database, no LLM, no graph. They run in milliseconds and
fail for exactly one reason."""

import pytest

from customer_support.models.enums.TicketStatusEnum import (
    InvalidTicketTransition,
    TicketStatus,
    assert_transition_allowed,
)


def test_pending_can_be_claimed():
    assert_transition_allowed(1, TicketStatus.PENDING, TicketStatus.UNDER_REVIEW)


def test_pending_cannot_jump_to_resolved():
    """An advisor must claim before resolving, or two advisors answer the same
    ticket without noticing."""
    with pytest.raises(InvalidTicketTransition):
        assert_transition_allowed(1, TicketStatus.PENDING, TicketStatus.RESOLVED)


def test_reopened_goes_to_under_review_not_pending():
    assert_transition_allowed(1, TicketStatus.REOPENED, TicketStatus.UNDER_REVIEW)
    with pytest.raises(InvalidTicketTransition):
        assert_transition_allowed(1, TicketStatus.REOPENED, TicketStatus.PENDING)


def test_duplicate_is_terminal():
    for target in TicketStatus:
        with pytest.raises(InvalidTicketTransition):
            assert_transition_allowed(1, TicketStatus.DUPLICATE, target)


def test_a_claimed_ticket_can_be_released_back_to_pending():
    """The one transition with no endpoint that made tickets unrecoverable: without
    it, claim was a one-way door out of the pending queue."""
    assert_transition_allowed(1, TicketStatus.UNDER_REVIEW, TicketStatus.PENDING)


def test_rejection_is_reachable_before_and_during_review():
    assert_transition_allowed(1, TicketStatus.PENDING, TicketStatus.REJECTED)
    assert_transition_allowed(1, TicketStatus.UNDER_REVIEW, TicketStatus.REJECTED)


def test_only_a_resolved_ticket_can_be_closed():
    assert_transition_allowed(1, TicketStatus.RESOLVED, TicketStatus.CLOSED)
    for origin in (TicketStatus.PENDING, TicketStatus.UNDER_REVIEW, TicketStatus.REJECTED):
        with pytest.raises(InvalidTicketTransition):
            assert_transition_allowed(1, origin, TicketStatus.CLOSED)


def test_every_end_state_can_be_reopened():
    """Nothing except DUPLICATE is a dead end — a wrong answer must always be
    correctable."""
    for terminal in (TicketStatus.RESOLVED, TicketStatus.REJECTED, TicketStatus.CLOSED):
        assert_transition_allowed(1, terminal, TicketStatus.REOPENED)


def test_the_whole_state_machine_is_reachable_over_http():
    """Guards the gap this work closed. Every transition the lifecycle permits needs
    a controller method behind it, or the dashboard grows a button with nothing to
    call and a ticket class that cannot move.

    DUPLICATE is excluded deliberately: duplicate detection is phase 2, and until it
    exists nothing should be able to put a ticket in a terminal state by hand.
    """
    from customer_support.controllers.EscalationController import EscalationController
    from customer_support.models.enums.TicketStatusEnum import ALLOWED_TRANSITIONS

    # target status -> the controller method that reaches it
    served_by = {
        TicketStatus.UNDER_REVIEW: "claim",
        TicketStatus.RESOLVED: "resolve",
        TicketStatus.PENDING: "release",
        TicketStatus.REJECTED: "reject",
        TicketStatus.CLOSED: "close",
        TicketStatus.REOPENED: "reopen",
    }

    reachable = {
        target
        for origin, targets in ALLOWED_TRANSITIONS.items()
        for target in targets
        if origin is not TicketStatus.DUPLICATE
    } - {TicketStatus.DUPLICATE}

    unserved = reachable - set(served_by)
    assert not unserved, f"transitions with no controller method: {unserved}"

    for target, method in served_by.items():
        assert callable(getattr(EscalationController, method, None)), (
            f"{target.value} is reachable but EscalationController.{method} is missing"
        )
