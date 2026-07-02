from helpers.config import get_settings, Settings
import os
class BaseController:
    def __init__(self, config: Settings = get_settings()):

        self.app_settings = config
        self.base_dir= os.path.dirname(os.path.dirname(__file__))

        self.files_dir =os.path.join(self.base_dir, "assets", "files")
        #self.file_dir =os.base_dir + "/" + "assets/files" another way to get the file path but the above way is more robust and platform-independent
