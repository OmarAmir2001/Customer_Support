from .BaseController import BaseController
from customer_support.models.db_schemas import Project, DataChunk
from customer_support.stores.llm.LLMEnum import DocumentTypeEnum
from customer_support.helpers.logging_config import get_logger
from typing import List
import json

class KBController(BaseController):
    def __init__(self,vectordb_client,generation_client,embedding_client):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.logger = get_logger(__name__)


    def create_collection_name(self,project_id:str):
        return f"collection_{project_id}".strip()

    async def reset_vectordb_collection(self,project:Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return await self.vectordb_client.delete_collection(collection_name=collection_name)

    async def get_vector_db_collection_info(self,project:Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = await self.vectordb_client.get_collection_info(
            collection_name=collection_name)

        # The await belongs on the provider call, not on json.loads: awaiting a dict
        # raises TypeError, which is what made this endpoint a guaranteed 500.
        return json.loads(json.dumps(collection_info, default=lambda o: o.__dict__))

    async def index_into_vector_db(self,project:Project,chunks:List[DataChunk],chunks_ids:List[int],do_reset:bool=False):

        # step 1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step 2 : get the data from the chunks
        texts = [chunk.chunk_text for chunk in chunks]
        metadatas = [chunk.chunk_metadata for chunk in chunks]
        vectors = self.embedding_client.embed_text(
                                    text=texts,
                                    document_type=DocumentTypeEnum.DOCUMENT.value)
        self.logger.info(
            "kb_embedding_complete",
            collection=collection_name,
            text_count=len(texts),
            vector_type=type(vectors).__name__,
        )
        if not vectors or len(vectors) != len(texts):
            self.logger.error(
                "kb_embedding_failed",
                collection=collection_name,
                text_count=len(texts),
                vector_count=len(vectors) if vectors else 0,
            )
            return False

        # step 3: create the collection if it doesn't exist
        _ = await self.vectordb_client.create_collection(collection_name=collection_name,
                                                   embedding_size=self.embedding_client.embedding_size,
                                                   do_reset=do_reset)

        # step 4: insert the data into the collection
        _ = await self.vectordb_client.insert_many(collection_name=collection_name,
                                             texts=texts,
                                             vectors=vectors,
                                             metadata=metadatas,
                                             record_ids=chunks_ids)
        return True

    # def search_vector_db_collection(self,project:Project,query:str,limit:int=10):

    #     collection_name = self.create_collection_name(project_id=project.project_id)

    #     vector = self.embedding_client.embed_text(text=query,
    #                                               document_type=DocumentTypeEnum.QUERY.value)[0]

    #     if not vector or len(vector) == 0:
    #         self.logger.error("Error while embedding query")
    #         return False

    #     results = self.vectordb_client.search_by_vector(collection_name=collection_name,
    #                                                     vector=vector,
    #                                                     limit=limit)
    #     if not results or len(results) == 0:
    #         return False

        
    #     return json.loads(json.dumps(results, default=lambda o: o.__dict__))