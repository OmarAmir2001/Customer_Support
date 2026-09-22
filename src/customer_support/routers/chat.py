"""Chat endpoints. HTTP only: no queries, no prompts, no lifecycle rules."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request, status

from customer_support.helpers.locale import negotiate_language
from customer_support.helpers.logging_config import get_logger, set_correlation_id
from customer_support.models.graph.conversation import ConversationMessage
from customer_support.routers.schemas.chat import ChatRequest, ChatResponse

logger = get_logger(__name__)

chat_router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@chat_router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    request: Request,
    payload: ChatRequest,
    accept_language: Annotated[str | None, Header()] = None,
) -> ChatResponse:
    thread_id = payload.thread_id or str(uuid.uuid4())

    # The thread id IS the correlation id: one grep follows this question through the
    # gates, into the ticket, and on to the advisor's resolution days later.
    set_correlation_id(thread_id)

    # Resolved once, here, because language negotiation is an HTTP concern. The
    # result rides in the graph state so no node or controller re-derives it.
    # profile_language stays None until MemoryController lands (Section 6); the
    # chain just skips that rung until then.
    language = negotiate_language(
        request.app.state.templates,
        requested=payload.language,
        profile_language=None,
        accept_language=accept_language,
    )

    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    try:
        final_state = await graph.ainvoke(
            {
                "question": payload.question,
                "student_id": payload.student_id,
                "thread_id": thread_id,
                "department": payload.department,
                "language": language,
                # Appended, not assigned: messages carries a reducer, so this
                # joins the thread's existing transcript instead of replacing it.
                "messages": [ConversationMessage.student(payload.question)],
                # Run-scoped fields, reset explicitly. The checkpointer MERGES each
                # run into the thread's existing state, so a field only written on
                # one path — ticket_id is written by escalate_node and nowhere else —
                # survives into later runs and reports a previous turn's outcome as
                # this one's. Seeding them here means every run starts from a known
                # state instead of inheriting one.
                "ticket_id": None,
                "escalate": False,
                "failed_gate": None,
                "gate_reason": None,
                "resolved_by": None,
            },
            config=config,
        )
    except Exception:
        logger.exception("graph_run_failed", student_id=payload.student_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The assistant is temporarily unavailable. Please try again.",
        ) from None

    # Read the gates' own verdict, not the presence of a ticket id. grade_node writes
    # `escalate` on EVERY run, so this value is self-healing even if a future field
    # is added without being reset above — whereas an id that is only ever written on
    # the escalate path is not.
    escalated = bool(final_state.get("escalate"))

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

    messages = snapshot.values.get("messages") or []

    return {
        "thread_id": thread_id,
        # The full transcript, oldest first, including an advisor's reply as its
        # own turn. This is the source of truth for what was said; the fields
        # below describe only the most recent turn.
        "messages": [message.model_dump(mode="json") for message in messages],
        "question": snapshot.values.get("question"),
        "answer": snapshot.values.get("answer"),
        "escalated": bool(snapshot.values.get("ticket_id")),
        "ticket_id": snapshot.values.get("ticket_id"),
        # Set only once an advisor has resolved the ticket, which is what tells the
        # student the answer they are reading is a human's and not the placeholder.
        "resolved_by": snapshot.values.get("resolved_by"),
    }