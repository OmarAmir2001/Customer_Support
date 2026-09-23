from fastapi import APIRouter,UploadFile, Depends, status,Request
from fastapi.responses import JSONResponse
from customer_support.helpers import get_settings, Settings
from customer_support.controllers import DataController, ProjectController, ProcessController , KBController,RetrievalController
import aiofiles
from customer_support.models import ResponseSignal
from customer_support.helpers.logging_config import get_logger
from .schemas import ProcessRequest,SearchRequest
from .schemas import PushRequest
from customer_support.models.ProjectModel import ProjectModel
from customer_support.models.ChunkModel import ChunkModel
from customer_support.models.db_schemas import Asset,DataChunk
from customer_support.models.AssetModel import AssetModel

from customer_support.models.enums.AssetTypeEnum import AssetTypeEnum
from customer_support.models.enums.ProcessingEnum import ProcessingEnum
import os


logger = get_logger(__name__)

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
        logger.error(
            "file_ingestion_failed",
            project_id=project_id,
            file_id=file_id,
            error=str(e),
            exc_info=True,
        )
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
        if process_controller.get_file_extension(file_id=file_id) == ProcessingEnum.JSON.value:
            file_chunks = process_controller.process_json_content(file_id=file_id)
        else:
            file_content = process_controller.get_file_content(file_id=file_id)
            if file_content is None:
                logger.error(
                    "file_content_unreadable",
                    project_id=project_id,
                    asset_id=asset_id,
                    file_id=file_id,
                )
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

@admin_router.get("/index_info/info/{project_id}")
async def get_project_index_info(request: Request,project_id: int):
    """
    get project index stats
    """
    project_model = await ProjectModel.create_instance(request.app.state.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)
    nlp_controller = KBController(vectordb_client=request.app.state.vectordb_client,
                                      generation_client=request.app.state.generation_client,
                                      embedding_client=request.app.state.embedding_client)
    collection_info = await nlp_controller.get_vector_db_collection_info(project=project)
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

    # Collect every chunk BEFORE syncing, rather than syncing page by page.
    #
    # sync_sections replaces one (source, section) group at a time. A section's
    # chunks can straddle a page boundary, so syncing per page would have page 2
    # delete the rows page 1 had just inserted for the same section — silently
    # losing them. get_all_chunks_by_project_id has no ORDER BY either, so which
    # chunks land together is not even deterministic.
    #
    # Reading a project's chunks into memory is fine at handbook scale (hundreds).
    # A corpus large enough to matter would need paging BY SECTION, not by row.
    all_chunks = []
    page_no = 1
    chunk_model = await ChunkModel.create_instance(request.app.state.db_client)
    while True:
        page_chunks = await chunk_model.get_all_chunks_by_project_id(
            project_id=project.project_id, page=page_no
        )
        if not page_chunks:
            break
        all_chunks.extend(page_chunks)
        page_no += 1

    if not all_chunks:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.NO_FILES_ERROR.value},
        )

    # The chunks' REAL ids, not a counter. These land in the vector table's chunk_id
    # column, which is the stable handle back to the source row; a per-push counter
    # renumbers every chunk on every push and points the FK at arbitrary rows.
    result = await nlp_controller.sync_sections(
        project=project,
        chunks=all_chunks,
        chunks_ids=[chunk.chunk_id for chunk in all_chunks],
        do_reset=push_request.do_reset,
    )
    if not result:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.VECTORDB_INSERTION_FAILED.value},
        )

    inserted_items_count = result["inserted"]

    return JSONResponse(content={
        "signal": ResponseSignal.VECTORDB_INSERTION_SUCCESS.value,
        "inserted_item_count": inserted_items_count
        })
    
#=========================================================================================================
#======================== Index Endpoint for search knowledge base =======================================
#=========================================================================================================

@admin_router.post("/knowledge_base/search/{project_id}")
async def search_knowledge_base(request: Request,project_id: int, search_request: SearchRequest):
    """
     searches the knowledge base for a specific query.
     Needed for when you want to search the knowledge base.
    """
    project_model = await ProjectModel.create_instance(request.app.state.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    # KBController owns the collection-naming rule, so the name is taken from there
    # rather than rebuilt by hand in a second place.
    kb_controller = KBController(vectordb_client=request.app.state.vectordb_client,
                                 generation_client=request.app.state.generation_client,
                                 embedding_client=request.app.state.embedding_client)
    collection_name = kb_controller.create_collection_name(project_id=project.project_id)

    # The same controller the graph uses, so this endpoint shows exactly what the agent
    # would see — department filter and handbook precedence included.
    nlp_controller = RetrievalController(vectordb_client=request.app.state.vectordb_client,
                                         embedding_client=request.app.state.embedding_client,
                                         collection_name=collection_name)

    results = await nlp_controller.retrieve(question=search_request.query,
                                            department=search_request.department,
                                            limit=search_request.limit)
    if not results:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.VECTORDB_SEARCH_FAILED.value}
        )

    return JSONResponse(content={
            "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
            "results": [doc.model_dump() for doc in results]
            })