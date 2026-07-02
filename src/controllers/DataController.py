from .BaseController import BaseController
from fastapi import UploadFile
from models.enums import ResponseSignel

class DataController(BaseController):
    def __init__(self):
        super().__init__()

    def validate_file(self, file: UploadFile):
        if file.content_type not in self.app_settings.File_Allowed_Types:
            return False, ResponseSignel.FILE_TYPE_NOT_ALLOWED.value
        
        if file.size > self.app_settings.File_Max_Size:
            return False,ResponseSignel.FILE_SIZE_EXCEEDS_LIMIT.value
        
        return True, ResponseSignel.FILE_VALIDATED_SUCCESS.value
