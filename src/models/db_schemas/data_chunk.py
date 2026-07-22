from pydantic import BaseModel,Field,validators
from typing import Optional
from bson.objectid import ObjectId


class DataChunk(BaseModel):
    id: Optional[ObjectId]= Field(default=None, alias="_id")
    chunk_text: str = Field(..., description="Text content of the data chunk", min_length=1)
    chunk_metadata: dict = Field(..., description="Metadata associated with the data chunk")
    chunk_order: int = Field(..., description="Order of the chunk in the original document", ge=0)
    chunk_project_id: ObjectId

    class Config:
        arbitrary_types_allowed = True
    

    @classmethod
    def get_indexes(cls):
        
        return[
            {
                "key":[
                    ("chunk_project_id",1)
                       ],
                "name":"chunk_project_id_index_1",
                "unique":False
            }
        ]