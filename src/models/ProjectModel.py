from .BaseDataModel import BaseDataModel
from .db_schemas import Project
from .enums import DatabaseEnum

class ProjectModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client)
        self.collection= self.client[DatabaseEnum.COLLECTION_PROJECT_NAME.value]

    async def create_project(self, project: Project):
        result = await self.collection.insert_one(project.model_dump())
        project._id=result.inserted_id
        return project