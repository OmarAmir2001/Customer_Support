from .BaseController import BaseController
from models.db_schemas import Project, DataChunk
from stores.llm.LLMEnum import DocumentTypeEnum
import logging
from typing import List
import json

class KBController(BaseController):
    def __init__(self,vectordb_client,generation_client,embedding_client):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.logger = logging.getLogger(__name__)


    def create_collection_name(self,project_id:str):
        return f"collection_{project_id}".strip()

    def reset_vectordb_collection(self,project:Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)

    def get_vector_db_collection_info(self,project:Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vectordb_client.get_collection_info(collection_name=collection_name)
        
        return json.loads(json.dumps(collection_info, default=lambda o: o.__dict__))

    def index_into_vector_db(self,project:Project,chunks:List[DataChunk],chunks_ids:List[int],do_reset:bool=False):

        # step 1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step 2 : get the data from the chunks
        texts = [chunk.chunk_text for chunk in chunks]
        metadatas = [chunk.chunk_metadata for chunk in chunks]
        vectors = self.embedding_client.embed_text(
                                    text=texts,
                                    document_type=DocumentTypeEnum.DOCUMENT.value)
        self.logger.info("texts=%s vectors=%s", len(texts), type(vectors))
        if not vectors or len(vectors) != len(texts):
            self.logger.error("Embedding failed for collection %s", collection_name)
            return False

        # step 3: create the collection if it doesn't exist
        _ = self.vectordb_client.create_collection(collection_name=collection_name,
                                                   embedding_size=self.embedding_client.embedding_size,
                                                   do_reset=do_reset)

        # step 4: insert the data into the collection
        _ = self.vectordb_client.insert_many(collection_name=collection_name,
                                             texts=texts,
                                             vectors=vectors,
                                             metadata=metadatas,
                                             record_ids=chunks_ids)
        return True