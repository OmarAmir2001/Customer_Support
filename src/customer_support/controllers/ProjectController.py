import os

from .BaseController import BaseController


class ProjectController(BaseController):
    def __init__(self, settings=None):
        # Injectable, like the other controllers. It was not, so anything that
        # constructed one — including ProcessController's own __init__ — forced a
        # read of the local .env even when its caller had settings to hand.
        super().__init__(settings)

    def get_project_path(self, project_id: str):
        """
        Get the path of the project based on the project ID.
        """
        project_dir = os.path.join(self.files_dir, str(project_id))

        if not os.path.exists(project_dir):
            os.makedirs(project_dir)

        return project_dir
