from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnum import DistanceMethodEnums
from qdrant_client import models,QdrantClient
from typing import List
from customer_support.helpers.logging_config import get_logger
import uuid
from customer_support.models.db_schemas import RetrievedDocument


class QdrantDBProvider(VectorDBInterface):
    def __init__(self , db_client,default_vector_size:int=786,
                    distance_method:str= None, index_threshold:int=10000):
        self.client=None
        self.db_client = db_client
        self.distance_method = distance_method

        if self.distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE

        elif self.distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT

        self.logger = get_logger(__name__)

    async def connect(self):
        self.client = QdrantClient(path=self.db_client)

    async def disconnect(self):
        self.client.close()

    async def is_collection_exists(self, collection_name:str)-> bool:
        return self.client.collection_exists(collection_name=collection_name)

    async def list_all_collections(self)-> List:
        return self.client.get_collections()

    async def get_collection_info(self,collection_name:str) -> dict:
        return self.client.get_collection(collection_name=collection_name)

    async def delete_collection(self,collection_name:str):

        if self.is_collection_exists(collection_name=collection_name):
            return self.client.delete_collection(collection_name=collection_name)
        else:
            self.logger.error(
                "collection_missing", collection=collection_name, operation="delete_collection"
            )
            return False

    async def create_collection(self,collection_name:str,
                              embedding_size:int,
                              do_reset:bool=False):

        if do_reset:
           _= self.delete_collection(collection_name=collection_name)

        if not self.is_collection_exists(collection_name=collection_name):
            _= self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=embedding_size,
                                                distance=self.distance_method))
            return True
        return False

    async def insert_one(self, collection_name:str, text:str ,vector: list,
                        metadata:dict=None, record_id:str=None):
        
        if not self.is_collection_exists(collection_name=collection_name):
            self.logger.error(
                "collection_missing", collection=collection_name, operation="insert_one"
            )
            return False
        if record_id is None:
                record_id = uuid.uuid4().hex
        try:
            _= self.client.upload_record(
                collection_name=collection_name,
                records=[models.Record(id=[record_id],vector=vector,payload={"text":text,"metadata":metadata})])
        except Exception as e:
            self.logger.error(
                "qdrant_insert_failed",
                collection=collection_name,
                operation="insert_one",
                error=str(e),
                exc_info=True,
            )
            return False

    async def insert_many( self, collection_name:str , texts:list[str],vectors:list[str]
                    , metadata: list[dict]=None, record_ids:list[str]=None , batch_size:int=50):
        if metadata is None:
            metadata = [None]*len(texts)
        if record_ids is None:
            record_ids = list(range(0,len(texts)))

        if not self.is_collection_exists(collection_name=collection_name):
            self.logger.error(
                "collection_missing", collection=collection_name, operation="insert_many"
            )
            return False

        for i in range(0,len(texts),batch_size):
            batch_end = i+batch_size
            batch_texts= texts[i:batch_end]
            batch_vectors= vectors[i:batch_end]
            batch_metadata= metadata[i:batch_end]
            batch_record_ids= record_ids[i:batch_end]
            batch_points = [
                models.PointStruct(id=batch_record_ids[x],
                                   vector=batch_vectors[x],
                                   payload={"text": batch_texts[x], "metadata": batch_metadata[x]})
                for x in range(len(batch_texts))
            ]
            try:
                _ = self.client.upsert(
                    collection_name=collection_name,
                    points=batch_points,
                    wait=True)
            except Exception as e:
                self.logger.error(
                    "qdrant_insert_failed",
                    collection=collection_name,
                    operation="insert_many",
                    batch_start=i,
                    error=str(e),
                    exc_info=True,
                )
                return False

        return True

    async def delete_by_metadata(self, collection_name: str, criteria: dict) -> int:
        """Delete every point matching ALL of `criteria`.

        Qdrant reports no deleted count, so this returns the number of matching points
        counted before the delete — the caller only logs it.
        """
        if not criteria:
            self.logger.error("delete_by_metadata_refused_empty", collection=collection_name)
            return 0

        if not self.is_collection_exists(collection_name=collection_name):
            self.logger.error(
                "collection_missing", collection=collection_name, operation="delete_by_metadata"
            )
            return 0

        # `must` is AND, matching the interface's contract.
        selector = models.Filter(
            must=[
                models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchValue(value=str(value)),
                )
                for key, value in criteria.items()
            ]
        )
        matched = self.client.count(
            collection_name=collection_name, count_filter=selector, exact=True
        ).count
        self.client.delete(collection_name=collection_name, points_selector=selector)
        return matched

    async def search_by_vector(self, collection_name:str, vector: list,limit:int = 10):

        result = self.client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=limit,
    )
        if not result or len(result.points) == 0:
            return False

        return [RetrievedDocument(**{
                "text":record.payload["text"],
                "score":record.score,
                "metadata":record.payload["metadata"]
                                  })
                for record in result.points
                ]

    
            

        


        


