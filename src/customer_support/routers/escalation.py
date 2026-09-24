"""Advisor endpoints. Every one delegates to EscalationController — the same methods
the graph calls, so resolve logic exists once."""

from contextlib import contextmanager

from fastapi import APIRouter, HTTPException, Query, Request, status

from customer_support.helpers.logging_config import get_logger
from customer_support.models.enums.TicketStatusEnum import (
    ConcurrentTicketUpdate,
    InvalidTicketTransition,
    TicketStatus,
)
from customer_support.routers.schemas.escalation import (
    ClaimRequest,
    CloseRequest,
    PromotionAssessmentRequest,
    PromotionAssessmentResponse,
    RejectRequest,
    ReleaseRequest,
    ReopenRequest,
    ResolveRequest,
    TicketDetail,
    TicketSummary,
)

logger = get_logger(__name__)

escalation_router = APIRouter(prefix="/api/v1/escalation", tags=["Escalation"])


@contextmanager
def _lifecycle_errors():
    """Map the four ways a lifecycle call fails onto status codes, once.

    Written once rather than inline per route because there are now six of these
    endpoints, and a missing arm in any one of them is a 500 on ordinary traffic.
    """
    try:
        yield
    except LookupError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found") from None
    except InvalidTicketTransition as exc:
        # 409, not 400: the request was well-formed, the ticket's state refused it.
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None
    except ConcurrentTicketUpdate as exc:
        # Also 409, and the same code on purpose. The causes differ — an illegal move
        # versus losing a race — but the client's remedy is identical: your view of
        # this ticket is stale, refresh it. Two advisors double-clicking Claim is
        # ordinary dashboard traffic and must not read as a server fault.
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None
    except ValueError as exc:
        # Controller-level invariants (an empty answer, a reason-less rejection).
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None


@escalation_router.get("/tickets", response_model=list[TicketSummary])
async def list_tickets(
    request: Request,
    ticket_status: TicketStatus | None = None,
    department: str | None = Query(default=None, pattern="^(CS|IS)$"),
    promotion_held: bool | None = Query(
        default=None,
        description=(
            "true returns the handbook review queue: tickets whose answer the "
            "contradiction check blocked from promotion."
        ),
    ),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> list[TicketSummary]:
    tickets = await request.app.state.ticket_model.list_tickets(
        status=ticket_status,
        department=department,
        promotion_held=promotion_held,
        page=page,
        page_size=page_size,
    )
    return [TicketSummary.model_validate(t) for t in tickets]


@escalation_router.get("/tickets/{ticket_id}", response_model=TicketDetail)
async def get_ticket(request: Request, ticket_id: int) -> TicketDetail:
    ticket = await request.app.state.ticket_model.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    return TicketDetail.model_validate(ticket)


@escalation_router.post("/tickets/{ticket_id}/claim", response_model=TicketSummary)
async def claim_ticket(request: Request, ticket_id: int, payload: ClaimRequest) -> TicketSummary:
    with _lifecycle_errors():
        ticket = await request.app.state.escalation_controller.claim(
            ticket_id=ticket_id, advisor_id=payload.advisor_id
        )

    return TicketSummary.model_validate(ticket)


@escalation_router.post("/tickets/{ticket_id}/resolve", response_model=TicketDetail)
async def resolve_ticket(request: Request, ticket_id: int, payload: ResolveRequest) -> TicketDetail:
    with _lifecycle_errors():
        ticket = await request.app.state.escalation_controller.resolve(
            ticket_id=ticket_id,
            advisor_answer=payload.advisor_answer,
            advisor_id=payload.advisor_id,
            promote_to_kb=payload.promote_to_kb,
        )

    return TicketDetail.model_validate(ticket)


@escalation_router.post("/tickets/{ticket_id}/release", response_model=TicketSummary)
async def release_ticket(
    request: Request, ticket_id: int, payload: ReleaseRequest
) -> TicketSummary:
    """Hand a claimed ticket back to the pending queue.

    The counterpart to claim. Without it, claiming was a one-way door: an advisor who
    picked up a ticket and then stopped working left it in under_review forever,
    missing from the pending queue and owned by nobody.
    """
    with _lifecycle_errors():
        ticket = await request.app.state.escalation_controller.release(
            ticket_id=ticket_id, actor=payload.advisor_id, note=payload.note
        )

    return TicketSummary.model_validate(ticket)


@escalation_router.post("/tickets/{ticket_id}/reject", response_model=TicketDetail)
async def reject_ticket(request: Request, ticket_id: int, payload: RejectRequest) -> TicketDetail:
    """Refuse a question outright: out of scope, spam, or not handbook material.

    The reason is required — the status history is the only record of the decision.
    """
    with _lifecycle_errors():
        ticket = await request.app.state.escalation_controller.reject(
            ticket_id=ticket_id, actor=payload.advisor_id, reason=payload.reason
        )

    return TicketDetail.model_validate(ticket)


@escalation_router.post("/tickets/{ticket_id}/close", response_model=TicketDetail)
async def close_ticket(request: Request, ticket_id: int, payload: CloseRequest) -> TicketDetail:
    """Finish a resolved ticket. Terminal unless someone reopens it.

    A closed ticket KEEPS its promoted answer in the knowledge base — closing is the
    happy ending of a resolution, not a retraction of it.
    """
    with _lifecycle_errors():
        ticket = await request.app.state.escalation_controller.close(
            ticket_id=ticket_id, actor=payload.advisor_id, note=payload.note
        )

    return TicketDetail.model_validate(ticket)


@escalation_router.post("/tickets/{ticket_id}/reopen", response_model=TicketDetail)
async def reopen_ticket(request: Request, ticket_id: int, payload: ReopenRequest) -> TicketDetail:
    """Send a finished ticket back for another look.

    Reachable from resolved, rejected and closed — the "this did not actually answer
    my question" path, and the only lifecycle move a student can trigger, which is why
    the body carries `actor` rather than `advisor_id`.

    Reopening un-promotes: the answer in the knowledge base is now suspect, and
    leaving the vector behind is the drift Section 1 forbids.
    """
    with _lifecycle_errors():
        ticket = await request.app.state.escalation_controller.reopen(
            ticket_id=ticket_id, actor=payload.actor, note=payload.note
        )

    return TicketDetail.model_validate(ticket)


@escalation_router.post(
    "/tickets/{ticket_id}/promotion-assessment", response_model=PromotionAssessmentResponse
)
async def assess_promotion(
    request: Request, ticket_id: int, payload: PromotionAssessmentRequest
) -> PromotionAssessmentResponse:
    """What the machine advises about promoting this answer (Section 5).

    Called WHILE the advisor types, so the draft comes in the body — there is
    nothing saved to assess yet. Read-only: it writes nothing and decides nothing.
    The dashboard uses `suggested_promote` to pre-tick the "add to knowledge base"
    box, and `held` to disable it.

    Advisory only, on purpose. `resolve` re-runs the contradiction check itself
    rather than trusting whatever this returned, because a client could skip this
    call entirely and promotion is the path that can poison the knowledge base.
    """
    ticket = await request.app.state.ticket_model.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")

    promotion = getattr(request.app.state, "promotion_controller", None)
    if promotion is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Promotion assessment is not configured.",
        )

    assessment = await promotion.assess(
        question=ticket.question,
        answer=payload.advisor_answer,
        department=ticket.department,
    )

    return PromotionAssessmentResponse(
        suggested_promote=assessment.suggested_promote,
        held=assessment.held,
        generalizable=assessment.generalizable,
        generalizability_score=assessment.generalizability_score,
        generalizability_reason=assessment.generalizability_reason,
        contradicts_handbook=assessment.contradicts_handbook,
        contradiction_score=assessment.contradiction_score,
        contradiction_reason=assessment.contradiction_reason,
        compared_against=assessment.compared_against,
    )
