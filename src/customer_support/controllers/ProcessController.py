from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from customer_support.models.enums import ProcessingEnum
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from langchain_core.documents import Document
from customer_support.helpers.logging_config import get_logger


# Handbook name -> department. Both halves of this mapping are load-bearing at
# retrieval time: RetrievalController ranks by `source` (handbook chunks outrank
# instructor-resolved ones) and filters by `department` (a CS student is never
# answered out of the IS handbook). Neither works unless ingestion writes them, so
# every chunk gets tagged here, at the one place chunks are created.
HANDBOOK_DEPARTMENTS = {
    "CS_2023": "CS",
    "IS_2023": "IS",
}


class ProcessController(BaseController):
    def __init__(self, project_id: str):
        super().__init__()

        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)
        self.logger = get_logger(__name__)

    def identify_handbook(self, file_id: str):
        """Map a stored file id back to (source, department).

        Stored ids carry a random prefix — `3EsFAHwA7Z8L_CS_2023.md` — and a loader's
        own `source` is the full path on disk. Either would make
        `source.startswith("CS_2023")` false, which silently drops handbook chunks
        BELOW instructor-resolved ones in the ranking. So the handbook name is
        recovered here and written as a clean `source`.
        """
        for handbook, department in HANDBOOK_DEPARTMENTS.items():
            if handbook.lower() in (file_id or "").lower():
                return handbook, department
        return None, None

    def _chunk_metadata(self, file_id: str, base: dict = None) -> dict:
        """Normalise one chunk's metadata: clean `source`, plus `department`."""
        metadata = dict(base or {})
        source, department = self.identify_handbook(file_id)

        if source is not None:
            metadata["source"] = source
            metadata["department"] = department
        else:
            # Not a recognised handbook: keep whatever source the loader gave, and
            # leave department unset so the chunk is treated as shared content.
            metadata.setdefault("source", file_id)
            metadata.setdefault("department", None)

        return metadata

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

        if not os.path.exists(file_path):
            return None

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

    def process_file_content(self,file_content:list,
                               file_id:str, chunk_size:int=1000, overlap:int=50):
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

        # Split the file content metadata into chunks, tagged so retrieval can rank and
        # filter them. Without this the loader's `source` is a file path.
        file_content_metadata = [
            self._chunk_metadata(file_id, doc.metadata) for doc in file_content
        ]

        # Create documents from the file content and metadata
        chunks = text_splitter.create_documents(file_content_text, metadatas=file_content_metadata)

        return chunks


    def process_json_content(self,file_id:str):
        """
        Load a pre-chunked JSON handbook (list of {text, source, section}).
        Skips extraction + splitting — the chunks are already made.
        Returns LangChain Documents so the rest of the pipeline is unchanged.
        """
        file_path = os.path.join(self.project_path, file_id)
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_chunks = json.load(f)

        documents = []
        for i,chunk in enumerate(raw_chunks):
            if "text" not in chunk:
                self.logger.error(
                    "json_chunk_missing_text", file_id=file_id, chunk_index=i
                )
                return None
            documents.append(Document(page_content=chunk["text"],
                                      metadata=self._chunk_metadata(
                                          chunk.get("source") or file_id,
                                          {
                                              "source":chunk.get("source"),
                                              "section":chunk.get("section")
                                          },
                                      )))
        return documents
            


        
        
        
