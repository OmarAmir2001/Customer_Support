from fastapi import APIRouter,UploadFile, Depends, status,Request
from fastapi.responses import JSONResponse
from helpers import get_settings, Settings
from controllers import DataController, ProjectController, ProccessController
import os
import aiofiles
from models import ResponseSignel
import logging
from .schemas import ProcessRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.db_schemas.data_chunk import DataChunk



logger = logging.getLogger('uvicorn.error')

admin_router = APIRouter(
    prefix="/api/v1/admin",  # Prefix for all routes in this router
    tags=["Admin Routes"]  # Tag for documentation purposes
)
# Placeholder implementation for admin management.
@admin_router.post("/ingest/{project_id}")
async def ingest_data(request: Request,project_id: str, file: UploadFile, app_settings: Settings = Depends(get_settings)):
    """
    Endpoint to ingest data into the system.
     re-runs the handbook ingestion pipeline. 
     Needed for when CS_2023.md or IS_2023.md get updated 
     and you need to refresh Qdrant without redeploying.
    """
    project_model = ProjectModel(db_client=request.app.state.db)
    project = await project_model.get_project_or_create_one(project_id=project_id)

                                 

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
    file_path, file_id = data_controller.generate_unique_filepath(original_filename=file.filename, project_id=project_id)

    # Save the uploaded file in chunks to the project directory
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            while chunk:= await file.read(app_settings.FILE_Default_CHUNK_SIZE):
                await f.write(chunk)
    # Handle any exceptions that occur during file ingestion and return a 500 Internal Server Error response with the appropriate signal and error message.
    except Exception as e:
        logger.error(f"Error occurred while ingesting file: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseSignel.FILE_INGESTION_FAILED.value, "error": str(e)}
        )
    # Return a success response indicating that the file ingestion was successful.
    return JSONResponse( content={"signal": ResponseSignel.FILE_INGESTION_SUCCESS.value,
                                   "file_id": file_id,
                                   })



#=========================================================================================================
#======================== Process Endpoint for Re-running Handbook Ingestion Pipeline ====================
#=========================================================================================================
@admin_router.post("/proccess/{project_id}")
async def process_endpoint(request: Request,project_id: str, process_request: ProcessRequest):
    """
    Endpoint to process data in the system.
    re-runs the handbook ingestion pipeline. 
    Needed for when CS_2023.md or IS_2023.md get updated 
    and you need to re chunk and re-ingest the data without redeploying.
    """
    # Placeholder response.
    file_id = process_request.file_id
    chunk_size = process_request.chunk_size
    overlap = process_request.overlap
    do_reset = process_request.do_reset

    project_model = ProjectModel(db_client=request.app.state.db)
    project = await project_model.get_project_or_create_one(project_id=project_id)
    
    process_controller = ProccessController(project_id=project_id)

    file_content = process_controller.get_file_content(file_id=file_id)

    file_chunks = process_controller.proccess_file_content(
        file_content=file_content,
        file_id=file_id, 
        chunk_size=chunk_size,
        overlap=overlap
        )
    if file_chunks is None or len(file_chunks) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignel.FILE_PROCESSING_FAILED.value, "error": "No chunks were created from the file content."}
        )


    file_chunks_records =[
        DataChunk(chunk_text=chunk.page_content,
                  chunk_metadata=chunk.metadata,
                  chunk_order=i+1,
                  chunk_project_id=project.id
                  )
        for i,chunk in enumerate(file_chunks)
                          ]
    
    chunk_model = ChunkModel(db_client=request.app.state.db)

    if do_reset == 1:
        _=await chunk_model.delete_chunk_by_project_id(project_id=project.id)

    
    no_of_records = await chunk_model.insert_many_chunks(chunks=file_chunks_records)
    return JSONResponse(content={"signal": ResponseSignel.FILE_PROCESSING_SUCCESS.value,
                                 "inserted_chunks": no_of_records,
    })

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



