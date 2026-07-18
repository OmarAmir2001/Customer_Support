from .BaseDataModel import BaseDataModel
from .db_schemas import Project
from .enums.DatabaseEnum import DatabaseEnum

class ProjectModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client)
        self.collection= self.db_client[DatabaseEnum.COLLECTION_PROJECT_NAME.value]

    async def create_project(self, project: Project):
        result = await self.collection.insert_one(project.model_dump(by_alias=True, exclude_unset=True))
        project._id=result.inserted_id
        return project
    
    async def get_project_or_create_one(self,project_id: str):

        record= await self.collection.find_one({
            "project_id":project_id
        })

        if record is None:
            # create new project
            project = Project(project_id=project_id)
            await self.create_project(project=project)
            return project
        return Project(**record)
    
    async def get_all_projects(self, page: int=1 ,page_size: int=10):

        # count total number of documents
        total_count = await self.collection.count_documents({})
        
        # calculate total number of pages
        total_pages = total_count // page_size
        if total_count % page_size != 0:
            total_pages += 1
        
        cursor = self.collection.find().skip((page-1)).limit(page_size)
        projects = []
        async for documents in cursor:
            projects.append((
                Project(**documents)
            ))
        return projects, total_pages