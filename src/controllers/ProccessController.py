from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from models.enums import ProcessingEnum
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ProccessController(BaseController):
    def __init__(self, project_id: str):
        super().__init__()

        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)

    def get_file_extension(self, file_id: str):
        """
        Get the file extension of the specified file in the project directory.
        """
        return os.path.splitext(file_id)[-1] # Returns the file extension without the dot
    def get_file_loader(self, file_id: str):
        """
        Get the appropriate file loader based on the file extension.
        """
        file_path = os.path.join(self.project_path, file_id)
        file_ext = self.get_file_extension(file_id)

        if file_ext == ProcessingEnum.TXT.value:
            return TextLoader(file_path, encoding='utf-8')
        if file_ext == ProcessingEnum.PDF.value:
            return PyMuPDFLoader(file_path)
        if file_ext == ProcessingEnum.MD.value:
            return TextLoader(file_path,encoding='utf-8')
        return None
    
    def get_file_content(self, file_id: str):
        """
        Get the content of the specified file in the project directory.
        """
        loader = self.get_file_loader(file_id=file_id)
        return loader.load() if loader else None

    def proccess_file_content(self,file_content:list,
                               file_id:str, chunk_size:int=100, overlap:int=20):
        """
        Process the content of the specified file in the project directory.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
        )
        # Split the file content into chunks
        file_content_text = [doc.page_content for doc in file_content]

        # Split the file content metadata into chunks
        file_content_metadata = [doc.metadata for doc in file_content]

        # Create documents from the file content and metadata
        chunks = text_splitter.create_documents(file_content_text, metadatas=file_content_metadata)

        return chunks

        
        
        
