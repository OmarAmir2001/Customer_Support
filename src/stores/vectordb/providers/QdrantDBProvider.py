from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnum import DistanceMethodEnums
from qdrant_client import model,QdrantClient
from typing import List
import logging


class QdrantDBProvider(VectorDBInterface):
    def __init__(self, db_path:str , distance_method:str= None):
        self.client=None
        self.db_path = db_path
        self.distance_method = distance_method

        if self.distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = model.Distance.COSINE

        elif self.distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = model.Distance.DOT

        self.logger = logging.getLogger(__name__)

    def connect(self):
        self.client = QdrantClient(path=self.db_path)

    def disconnect(self):
        self.client.close()

    def is_collection_exists(self, collection_name:str)-> bool:
        return self.client.collection_exists(collection_name=collection_name)

    def list_all_collections(self)-> List:
        return self.client.get_collections()

    def get_collection_info(self,collection_name:str) -> dict:
        return self.client.get_collection(collection_name=collection_name)

    def delete_collection(self,collection_name:str):

        if self.is_collection_exists(collection_name=collection_name):
            return self.client.delete_collection(collection_name=collection_name)
        else:
            self.logger.error(f"Collection {collection_name} does not exist")
            return False

    def create_collection(self,collection_name:str,
                              embedding_size:int,
                              do_reset:bool=False):

        if do_reset:
           _= self.delete_collection(collection_name=collection_name)

        if not self.is_collection_exists(collection_name=collection_name):
            _= self.client.create_collection(
                collection_name=collection_name,
                vectors_config=model.VectorParams(size=embedding_size,
                                                distance=self.distance_method))
            return True
        return False

    def insert_one(self, collection_name:str, text:str ,vector: list,
                        metadata:dict=None, record_id:str=None):
        
        if not self.is_collection_exists(collection_name=collection_name):
            self.logger.error(f"Collection {collection_name} does not exist")
            return False
        try:
            _= self.client.upload_record(
                collection_name=collection_name,
                records=[model.Record(vector=vector,payload={"text":text,"metadata":metadata})])
        except Exception as e:
            self.logger.error(f"Error while inserting batch {e}")
            return False

    def insert_many( self, collection_name:str , texts:list[str],vectors:list[str]
                    , metadata: list[dict]=None, record_ids:list[str]=None , batch_size:int=50):
        if metadata is None:
            metadata = [None]*len(texts)
        if record_ids is None:
            record_ids = [None]*len(texts)

        if not self.is_collection_exists(collection_name=collection_name):
            self.logger.error(f"Collection {collection_name} does not exist")
            return False

        for i in range(0,len(texts),batch_size):
            batch_end = i+batch_size
            batch_texts= texts[i:batch_end]
            batch_vectors= vectors[i:batch_end]
            batch_metadata= metadata[i:batch_end]
            batch_record_ids= record_ids[i:batch_end]
            batch_records=[
                model.Record(vector=batch_vectors[x],
                            payload={"text":batch_texts[x],"metadata":batch_metadata[x]})
                for x in range(len(batch_texts))

            ]
            try:
                
                _= self.client.upload_records(
                    collection_name=collection_name,
                    records=batch_records)
            except Exception as e:
                self.logger.error(f"Error while inserting batch {e}")
            
                return False

    def search_by_vector(self, collection_name:str, vector: list,limit:int = 5):

        result = self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit
        )
        return result
    
            

        


        


