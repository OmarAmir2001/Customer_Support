"""Chat endpoints. HTTP only: no queries, no prompts, no lifecycle rules."""

import uuid

from fastapi import APIRouter, HTTPException, Request, status

from customer_support.helpers.logging_config import get_logger, set_correlation_id
from customer_support.routers.schemas.chat import ChatRequest, ChatResponse

logger = get_logger(__name__)

chat_router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@chat_router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    thread_id = payload.thread_id or str(uuid.uuid4())

    # The thread id IS the correlation id: one grep follows this question through the
    # gates, into the ticket, and on to the advisor's resolution days later.
    set_correlation_id(thread_id)

    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    try:
        final_state = await graph.ainvoke(
            {
                "question": payload.question,
                "student_id": payload.student_id,
                "thread_id": thread_id,
                "department": payload.department,
            },
            config=config,
        )
    except Exception:
        logger.exception("graph_run_failed", student_id=payload.student_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The assistant is temporarily unavailable. Please try again.",
        ) from None

    escalated = bool(final_state.get("ticket_id"))

    return ChatResponse(
        thread_id=thread_id,
        answer=final_state.get("answer") or "",
        escalated=escalated,
        ticket_id=final_state.get("ticket_id"),
        escalation_reason=final_state.get("gate_reason") if escalated else None,
    )


@chat_router.get("/{thread_id}")
async def get_conversation(request: Request, thread_id: str) -> dict:
    """How a student picks up an advisor's answer: delivery is passive.

    Nothing was pushed anywhere when the ticket was resolved; the advisor's message
    was written into the persisted thread state and is read from here on return.
    """

    set_correlation_id(thread_id)

    snapshot = await request.app.state.graph.aget_state({"configurable": {"thread_id": thread_id}})
    if snapshot is None or not snapshot.values:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown conversation")

    return {
        "thread_id": thread_id,
        "question": snapshot.values.get("question"),
        "answer": snapshot.values.get("answer"),
        "escalated": bool(snapshot.values.get("ticket_id")),
        "ticket_id": snapshot.values.get("ticket_id"),
        # Set only once an advisor has resolved the ticket, which is what tells the
        # student the answer they are reading is a human's and not the placeholder.
        "resolved_by": snapshot.values.get("resolved_by"),
    }