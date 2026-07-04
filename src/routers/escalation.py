from fastapi import APIRouter

escalation_router = APIRouter(
    prefix="/api/v1/escalation",  # Prefix for all routes in this router
    tags=["Escalation Routes"]  # Tag for documentation purposes
)
# Placeholder implementation for escalation management.
@escalation_router.get("/escalations")
async def get_escalations():
    """
     lists all pending escalations waiting for advisor review. 
     Needed so an advisor has a queue to work from.
    """
    # Placeholder response.
    escalations = [
        {"escalation_id": "1", "user_id": "123", "issue": "grades not appearing", "status": "open"},
        {"escalation_id": "2", "user_id": "456", "issue": "assignment resubmission request", "status": "in_progress"},
        {"escalation_id": "3", "user_id": "789", "issue": "midterm repeat request", "status": "closed"}
    ]
    return {"escalations": escalations}

@escalation_router.get("/escalation/{escalation_id}")
async def get_escalation(escalation_id: str):
    """
     returns the full escalation summary (student context, question, what was found, why it escalated).
     Needed so the advisor has enough context to answer.
    """
    # Placeholder response.
    escalation = {
        "escalation_id": escalation_id,
        "user_id": "123",
        "issue": "grades not appearing",
        "status": "open"
    }
    return {"escalation": escalation}

@escalation_router.post("/escalation/{escalation_id}/resolve")
async def resolve_escalation(escalation_id: str):
    """
     advisor submits their answer. 
     This resumes the interrupted graph AND triggers the learning loop (the Q&A gets embedded into Qdrant). 
     This is the single most important endpoint in the whole project — it's where the "agent gets smarter" 
     feature actually lives.
    """
    # Placeholder response.
    return {"message": f"Escalation {escalation_id} has been resolved."}