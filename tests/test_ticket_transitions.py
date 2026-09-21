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