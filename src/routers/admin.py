from fastapi import APIRouter,UploadFile, Depends, status
from fastapi.responses import JSONResponse
from helpers import get_settings, Settings
from controllers import DataController
from controllers import ProjectController
import os
import aiofiles
from models import ResponseSignel

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
    data_controller = DataController()
    # Validate the uploaded file using DataController
    is_valid,resault_signal = data_controller.validate_file(file)

    # If the file is not valid, return a 400 Bad Request response with the appropriate signal.
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": resault_signal}
        )
    # Get the project directory path using ProjectController
    project_dir_path=ProjectController().get_project_path(project_id=project_id)
    file_path = data_controller.generate_unique_filename(original_filename=file.filename, project_id=project_id)

    # Save the uploaded file in chunks to the project directory
    async with aiofiles.open(file_path, 'wb') as f:
        while chunk:= await file.read(app_settings.FILE_Default_CHUNK_SIZE):
            await f.write(chunk)

    # Return a success response indicating that the file ingestion was successful.
    return JSONResponse( content={"signal": ResponseSignel.FILE_INGESTION_SUCCESS.value})



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