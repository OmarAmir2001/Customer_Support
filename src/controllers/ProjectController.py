from .BaseController import BaseController
from fastapi import UploadFile
from models import ResponseSignel
import os

class ProjectController(BaseController):
    def __init__(self):
        super().__init__()

    def get_project_path(self, project_id: str):
        """
        Get the path of the project based on the project ID.
        """
        project_dir = os.path.join(self.files_dir, project_id)

        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
      
        return project_dir