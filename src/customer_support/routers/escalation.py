"""Advisor endpoints. Every one delegates to EscalationController — the same methods
the graph calls, so resolve logic exists once."""

from fastapi import APIRouter, HTTPException, Query, Request, status

from customer_support.helpers.logging_config import get_logger
from customer_support.models.enums.TicketStatusEnum import (
    InvalidTicketTransition,
    TicketStatus,
)
from customer_support.routers.schemas.escalation import (
    ClaimRequest,
    PromotionAssessmentRequest,
    PromotionAssessmentResponse,
    ResolveRequest,
    TicketDetail,
    TicketSummary,
)

logger = get_logger(__name__)

escalation_router = APIRouter(prefix="/api/v1/escalation", tags=["Escalation"])


@escalation_router.get("/tickets", response_model=list[TicketSummary])
async def list_tickets(
    request: Request,
    ticket_status: TicketStatus | None = None,
    department: str | None = Query(default=None, pattern="^(CS|IS)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> list[TicketSummary]:
    tickets = await request.app.state.ticket_model.list_tickets(
        status=ticket_status, department=department, page=page, page_size=page_size
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
    try:
        ticket = await request.app.state.escalation_controller.claim(
            ticket_id=ticket_id, advisor_id=payload.advisor_id
        )
    except LookupError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found") from None
    except InvalidTicketTransition as exc:
        # 409, not 400: the request was well-formed, the ticket's state refused it.
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None

    return TicketSummary.model_validate(ticket)


@escalation_router.post("/tickets/{ticket_id}/resolve", response_model=TicketDetail)
async def resolve_ticket(request: Request, ticket_id: int, payload: ResolveRequest) -> TicketDetail:
    try:
        ticket = await request.app.state.escalation_controller.resolve(
            ticket_id=ticket_id,
            advisor_answer=payload.advisor_answer,
            advisor_id=payload.advisor_id,
            promote_to_kb=payload.promote_to_kb,
        )
    except LookupError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Ticket not found") from None
    except InvalidTicketTransition as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None

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
