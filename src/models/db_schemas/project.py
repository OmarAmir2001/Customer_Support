from pydantic import BaseModel,Field,validators
from typing import Optional
from bson.objectid import ObjectId

class Project(BaseModel):
    _id: Optional[ObjectId]
    project_id: str = Field(..., description="Unique identifier for the project", min_length=1)

    @validators('project_id')
    def validate_project_id(cls, value):
        if not value.isalnum() or not value.strip():
            raise ValueError("Project ID cannot be empty and must be alphanumeric.")
        return value
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str
        }


