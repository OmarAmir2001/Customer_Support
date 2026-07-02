from fastapi import APIRouter,UploadFile, Depends, status
from fastapi.responses import JSONResponse
from helpers import get_settings, Settings
from controllers import DataController



admin_router = APIRouter(
    prefix="/api/v1/admin",  # Prefix for all routes in this router
    tags=["Admin Routes"]  # Tag for documentation purposes
)
# Placeholder implementation for admin management.
@admin_router.post("/ingest/{project_id}")
async def ingest_data(project_id: str, file: UploadFile, app_settings: Settings = Depends(get_settings)):
    """
    Endpoint to ingest data into the system.
     re-runs the handbook ingestion pipeline. 
     Needed for when CS_2023.md or IS_2023.md get updated 
     and you need to refresh Qdrant without redeploying.
    """
    # Validate the uploaded file using DataController
    is_valid,resault_signal = DataController().validate_file(file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": resault_signal}
        )

    # Placeholder response.
    return {'signal': resault_signal}


@admin_router.get("/knowledge_base/stats")
async def get_knowledge_base_stats():
    """
     returns chunk counts, last ingestion date, collection health.
     Useful for debugging and for showing off in a demo ("look, it has 340 chunks indexed").
    """
    # Placeholder response.
    stats = {
        "num_chunks": 340,
        "last_ingestion": "2024-01-01T12:00:00Z",
        "collection_health": "healthy"
    }
    return {"knowledge_base_stats": stats}