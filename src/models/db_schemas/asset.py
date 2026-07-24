from pydantic import BaseModel,Field,field_validator
from typing import Optional
from bson.objectid import ObjectId
from datetime import datetime
class Asset(BaseModel):
     id: Optional[ObjectId]= Field(default=None, alias="_id")
     asset_project_id: ObjectId
     asset_type: str = Field(..., description="Type of asset", min_length=1)
     asset_name: str = Field(..., description="Name of the asset", min_length=1)
     asset_size: int = Field(description="Size of the asset in bytes",ge=0,default=0)
     asset_config: dict = Field(description="Configuration of the asset",default={})
     asset_pushed_at: datetime = Field(description="Date and time when the asset was pushed to the database",default_factory=datetime.now)

     class Config:
        arbitrary_types_allowed = True
    
     @classmethod
     def get_indexes(cls):
        
        return[
            {
                "key":[("asset_project_id",1)],
                "name":"asset_project_id_index_1",
                "unique":False
            },
            {
                 "key":[("asset_project_id",1),("asset_name",1)],
                 "name":"asset_project_id_name_index_1",
                 "unique":True
            }
        ]


