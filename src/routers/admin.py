from fastapi import APIRouter,UploadFile, Depends, status,Request
from fastapi.responses import JSONResponse
from helpers import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController , KBController
import aiofiles
from models import ResponseSignal
import logging
from .schemas import ProcessRequest
from .schemas import PushRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.db_schemas import Asset,DataChunk
from models.AssetModel import AssetModel

from models.enums.AssetTypeEnum import AssetTypeEnum
import os


logger = logging.getLogger(__name__)

admin_router = APIRouter(
    prefix="/api/v1/admin",  # Prefix for all routes in this router
    tags=["Admin Routes"]  # Tag for documentation purposes
)
# Placeholder implementation for admin management.
@admin_router.post("/ingest/{project_id}")
async def ingest_data(request: Request,project_id: int, file: UploadFile, app_settings: Settings = Depends(get_settings)):
    """
    Endpoint to ingest data into the system.
    This endpoint accepts a file upload and associates it with a specific project.
    The file is then chunked and stored in the database.
    """
    project_model = await ProjectModel.create_instance(db_client=request.app.state.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

                                 

    data_controller = DataController()
    # Validate the uploaded file using DataController
    is_valid,result_signal = data_controller.validate_file(file)

    # If the file is not valid, return a 400 Bad Request response with the appropriate signal.
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": result_signal}
        )
    # Get the project directory path using ProjectController
    project_dir_path=ProjectController().get_project_path(project_id=project_id)
    file_path, file_id = data_controller.generate_unique_filepath(original_filename=file.filename, project_id=project_id)

    # Save the uploaded file in chunks to the project directory
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            while chunk:= await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    # Handle any exceptions that occur during file ingestion and return a 500 Internal Server Error response with the appropriate signal and error message.
    except Exception as e:
        logger.error(f"Error occurred while ingesting file: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseSignal.FILE_INGESTION_FAILED.value, "error": str(e)}
        )

    # Store the asset in the database
    asset_model = await AssetModel.create_instance(db_client=request.app.state.db_client)
    asset_resource = Asset(asset_project_id=project.project_id,
                  asset_type=AssetTypeEnum.FILE.value,
                  asset_name=file_id,
                  asset_size=os.path.getsize(file_path)
                  )
    asset_record=await asset_model.create_asset(asset=asset_resource)


    # Return a success response indicating that the file ingestion was successful.
    return JSONResponse( content={"signal": ResponseSignal.FILE_INGESTION_SUCCESS.value,
                                   "file_id": str(asset_record.asset_id),
                                   })



#=========================================================================================================
#======================== Process Endpoint for Re-running Handbook Ingestion Pipeline ====================
#=========================================================================================================
@admin_router.post("/process/{project_id}")
async def process_endpoint(request: Request,project_id: int, process_request: ProcessRequest):
    """
    Endpoint to process data in the system.
    re-runs the handbook ingestion pipeline. 
    Needed for when CS_2023.md or IS_2023.md get updated 
    and you need to re chunk and re-ingest the data without redeploying.
    """
    # Placeholder response.
    chunk_size = process_request.chunk_size
    overlap = process_request.overlap
    do_reset = process_request.do_reset

    project_model = await ProjectModel.create_instance(db_client=request.app.state.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    asset_model = await AssetModel.create_instance(db_client=request.app.state.db_client)

    project_files_ids={}
    if process_request.file_id:
        asset_record = await asset_model.get_asset_by_id(asset_project_id=project.project_id,asset_name=process_request.file_id)
        if asset_record is None:
            return JSONResponse( 
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"signal": ResponseSignal.FILE_ID_ERROR.value}
                
            )
        project_files_ids={
           asset_record.asset_id :asset_record.asset_name
        }
        
    else: 
        project_files = await asset_model.get_all_project_assets(
            asset_project_id=project.project_id,
            asset_type=AssetTypeEnum.FILE.value
              )
        project_files_ids={
            record.asset_id :record.asset_name
              for record in project_files
            }
    if len(project_files_ids) == 0:
        return JSONResponse( 
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.NO_FILES_ERROR.value}
            
        )

    
    process_controller = ProcessController(project_id=project_id)


    no_of_records = 0
    no_of_files=0
    chunk_model = await ChunkModel.create_instance(db_client=request.app.state.db_client)

    if do_reset == 1:
         _= await chunk_model.delete_chunk_by_project_id(project_id=project.project_id)


    for asset_id,file_id in project_files_ids.items():
        file_content = process_controller.get_file_content(file_id=file_id)

        if file_content is None:
            logger.error(f"Error occurred while processing file: {file_id}")
            continue

        file_chunks = process_controller.process_file_content(
            file_content=file_content,
            file_id=file_id, 
            chunk_size=chunk_size,
            overlap=overlap
            )
        if file_chunks is None or len(file_chunks) == 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"signal": ResponseSignal.FILE_PROCESSING_FAILED.value, "error": "No chunks were created from the file content."}
            )


        file_chunks_records =[
            DataChunk(chunk_text=chunk.page_content,
                    chunk_metadata=chunk.metadata,
                    chunk_order=i+1,
                    chunk_project_id=project.project_id,
                    chunk_asset_id=asset_id
                    )
            for i,chunk in enumerate(file_chunks)
                            ]

        
        no_of_records += await chunk_model.insert_many_chunks(chunks=file_chunks_records)
        no_of_files+=1
    return JSONResponse(content={"signal": ResponseSignal.FILE_PROCESSING_SUCCESS.value,
                                "inserted_chunks": no_of_records,
                                "processed_files": no_of_files
    })
    

#=========================================================================================================
#======================== Index Endpoint for getting knowledge base stats ================================
#=========================================================================================================

@admin_router.get("/index_info/stats/{project_id}")
async def get_project_index_stats(request: Request,project_id: int):
    """
    get project index stats
    """
    project_model = await ProjectModel.create_instance(request.app.state.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)
    nlp_controller = KBController(vectordb_client=request.app.state.vectordb_client,
                                      generation_client=request.app.state.generation_client,
                                      embedding_client=request.app.state.embedding_client)
    collection_info = nlp_controller.get_vector_db_collection_info(project=project)
    return JSONResponse(content={
            "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
            "collection_info": collection_info
            })

#=========================================================================================================
#======================== Index Endpoint for pushing knowledge base to vector database ====================
#=========================================================================================================

@admin_router.post("/knowledge_base/push/{project_id}")
async def push_knowledge_base(request: Request,project_id: int,push_request: PushRequest):
    """
     pushes the knowledge base to the vector database.
     Needed for when you want to refresh the vector database.
    """
    project_model = await ProjectModel.create_instance(request.app.state.db_client)

    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.PROJECT_NOT_FOUND.value}
        )
    nlp_controller = KBController(vectordb_client=request.app.state.vectordb_client,
                                  generation_client=request.app.state.generation_client,
                                  embedding_client=request.app.state.embedding_client)

    has_records = True
    page_no = 1
    inserted_items_count = 0
    idx = 0
    chunk_model = await ChunkModel.create_instance(request.app.state.db_client)
    while has_records:
        page_chunks = await chunk_model.get_all_chunks_by_project_id(project_id=project.project_id, page=page_no)

        if not page_chunks or len(page_chunks) == 0:
            has_records = False
            break
        chunks_ids = list(range(idx,idx+len(page_chunks)))
        idx+=len(page_chunks)
        is_inserted = nlp_controller.index_into_vector_db(
                        project=project,
                        chunks=page_chunks,
                        do_reset=push_request.do_reset,
                        chunks_ids=chunks_ids
                    )
        if not is_inserted:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"signal": ResponseSignal.VECTORDB_INSERTION_FAILED.value}
            )
        inserted_items_count+=len(page_chunks)
        page_no+=1

    return JSONResponse(content={
        "signal": ResponseSignal.VECTORDB_INSERTION_SUCCESS.value,
        "inserted_item_count": inserted_items_count
        })
    
#=========================================================================================================
#======================== Index Endpoint for search knowledge base =======================================
#=========================================================================================================

@admin_router.post("/knowledge_base/search")
async def search_knowledge_base(query: str):
    """
     searches the knowledge base for a specific query.
     Needed for when you want to search the knowledge base.
    """
    # Placeholder response.
    return {"message": "Search results for the query: " + query}