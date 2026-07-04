from enum import Enum

class DatabaseEnum(str, Enum):
    COLLECTION_PROJECT_NAME = "projects"
    COLLECTION_CHUNK_NAME = "chunks"