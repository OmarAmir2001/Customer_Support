"""Chat endpoints. HTTP only: no queries, no prompts, no lifecycle rules."""

import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, status

from customer_support.helpers.locale import negotiate_language
from customer_support.helpers.logging_config import get_logger, set_correlation_id
from customer_support.models.graph.conversation import ConversationMessage
from customer_support.routers.schemas.chat import ChatRequest, ChatResponse
from customer_support.utils.metrics import record_question

logger = get_logger(__name__)

chat_router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@chat_router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    request: Request,
    payload: ChatRequest,
    background_tasks: BackgroundTasks,
    accept_language: Annotated[str | None, Header()] = None,
) -> ChatResponse:
    thread_id = payload.thread_id or str(uuid.uuid4())

    # The thread id IS the correlation id: one grep follows this question through the
    # gates, into the ticket, and on to the advisor's resolution days later.
    set_correlation_id(thread_id)

    memory = request.app.state.memory

    # READING the profile is on the critical path — the answer depends on it, and the
    # locale decision below needs it before the graph can start. WRITING it is not,
    # and is scheduled after the response (Section 6).
    profile = await memory.load_profile(payload.student_id)

    # Resolved once, here, because language negotiation is an HTTP concern. The
    # result rides in the graph state so no node or controller re-derives it.
    language = negotiate_language(
        request.app.state.templates,
        requested=payload.language,
        profile_language=profile.preferred_language,
        accept_language=accept_language,
    )

    # The stored profile WINS over the request body. Department drives which handbook
    # is searched, so letting a client assert it means a CS student can read the IS
    # handbook by claiming to be in IS. The body is a fallback for a student we have
    # never seen, whose department this turn's extraction may then learn.
    department = profile.department or payload.department
    if profile.department and payload.department and profile.department != payload.department:
        logger.warning(
            "department_conflict",
            student_id=payload.student_id,
            claimed=payload.department,
            stored=profile.department,
        )

    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    try:
        final_state = await graph.ainvoke(
            {
                "question": payload.question,
                "student_id": payload.student_id,
                "thread_id": thread_id,
                "department": department,
                "profile": profile,
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

    # The one place every question's outcome is known. Recorded here rather than in a
    # node because a run can leave the graph by more than one path, and an escalation
    # rate built from a counter that misses a path is worse than none.
    record_question(
        outcome="escalated" if escalated else "answered",
        failed_gate=final_state.get("failed_gate"),
        department=payload.department,
    )

    # Off the critical path: BackgroundTasks runs after the response is sent, so
    # the student never waits on extraction. MemoryController gates itself — most
    # turns carry no identity fact and cost no model call at all.
    if request.app.state.settings.MEMORY_ENABLED:
        background_tasks.add_task(memory.remember, payload.student_id, payload.question)

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
