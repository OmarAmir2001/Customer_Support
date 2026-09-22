"""Student profile endpoints (Section 6).

HTTP only — MemoryController owns loading, ProfileModel owns the SQL.

These were placeholders returning a hardcoded student. That was harmless while no
memory existed and actively misleading now: a debugging endpoint that always answers
"John Doe, GPA 3.8" hides exactly the bug you would open it to find.
"""

from fastapi import APIRouter, HTTPException, Request, status

from customer_support.helpers.logging_config import get_logger

logger = get_logger(__name__)

profile_router = APIRouter(prefix="/api/v1/profile", tags=["Profile Routes"])


@profile_router.get("/{student_id}")
async def get_user_profile(request: Request, student_id: str) -> dict:
    """What the agent remembers about a student.

    Identity only, never past questions (Section 6): conversation history is the
    checkpointer's job and is readable from GET /api/v1/chat/{thread_id}.

    Returns 200 with empty fields rather than 404 for a student we have not learned
    anything about yet — "we know nothing about them" is a real answer, and a client
    should not have to special-case a first-time student.
    """
    profile = await request.app.state.memory.load_profile(student_id)

    return {
        "student_id": student_id,
        "known": not profile.is_empty,
        "profile": profile.model_dump(),
    }


@profile_router.delete("/{student_id}", status_code=status.HTTP_200_OK)
async def delete_user_profile(request: Request, student_id: str) -> dict:
    """Wipe a student's long-term memory entirely.

    Needed for privacy and reset requests. It deletes the PROFILE only; conversation
    threads live in the checkpointer and are deleted separately, so this is not a
    complete erasure and does not claim to be.
    """
    deleted = await request.app.state.profile_model.delete_profile(student_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No profile for that student"
        )

    logger.info("profile_delete_requested", student_id=student_id)
    return {"student_id": student_id, "deleted": True}
