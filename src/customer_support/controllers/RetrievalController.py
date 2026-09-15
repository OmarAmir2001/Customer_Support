from .BaseController import BaseController
from models.db_schemas import Project, DataChunk
from stores.llm.LLMEnum import DocumentTypeEnum
import logging
from typing import List
import json

class RetrievalController(BaseController):
    def __init__(self,vectordb_client,generation_client,embedding_client):
        super().__init__()
    
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.logger = logging.getLogger(__name__)

    def create_collection_name(self,project_id:str):
            return f"collection_{project_id}".strip()

    def search_vector_db_collection(self,project:Project,query:str,limit:int=10):
    
        collection_name = self.create_collection_name(project_id=project.project_id)
    
        vector = self.embedding_client.embed_text(text=query,
                                                  document_type=DocumentTypeEnum.QUERY.value)[0]
    
        if not vector or len(vector) == 0:
            self.logger.error("Error while embedding query")
            return False
    
        results = self.vectordb_client.search_by_vector(collection_name=collection_name,
                                                        vector=vector,
                                                        limit=limit)
        if not results or len(results) == 0:
            return False
    
    
        return json.loads(json.dumps(results, default=lambda o: o.__dict__))