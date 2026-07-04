from .BaseController import BaseController
from .ProjectController import ProjectController
from fastapi import UploadFile
from models.enums import ResponseSignel
import re
import os

class DataController(BaseController):
    def __init__(self):
        super().__init__()

    def validate_file(self, file: UploadFile):
        if file.content_type not in self.app_settings.File_Allowed_Types:
            return False, ResponseSignel.FILE_TYPE_NOT_ALLOWED.value
        
        if file.size > self.app_settings.File_Max_Size:
            return False,ResponseSignel.FILE_SIZE_EXCEEDS_LIMIT.value
        
        return True, ResponseSignel.FILE_VALIDATED_SUCCESS.value
    
    def generate_unique_filepath(self ,original_filename:str,project_id:str):
        """
        Generate a unique filepath by appending a timestamp to the original filename.
        """
        # Generate a random key to ensure uniqueness
        random_key = self.generate_random_string()
        project_path=ProjectController().get_project_path(project_id=project_id)
        cleaned_filename = self.get_clean_filename(original_filename)

        # Generate a new file path with the random key and cleaned filename
        new_file_path = os.path.join(project_path,
                                      f"{random_key}_{cleaned_filename}")
        # Ensure the filename is unique within the project directory
        while os.path.exists(new_file_path):
            random_key = self.generate_random_string()
            new_file_path = os.path.join(project_path, f"{random_key}_{cleaned_filename}")
        
        return new_file_path, f"{random_key}_{cleaned_filename}"
    

    def get_clean_filename(self, original_filename: str):
        """
        Generate a clean filename by removing special characters and spaces.
        """
        # Use regex to replace any character that is not a word character, dot, or hyphen with an underscore
        cleaned_filename = re.sub(r'[^\w.-]', '_', original_filename.strip())
        cleaned_filename = cleaned_filename.replace(' ', '_')  # Replace spaces with underscores
        return cleaned_filename


