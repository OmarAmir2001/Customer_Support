from enum import Enum

class ResponseSignal(Enum):
    FILE_VALIDATED_SUCCESS = "File validated successfully."
    FILE_VALIDATED_FAILURE = "File validation failed."
    FILE_TYPE_NOT_ALLOWED = "File type is not allowed."
    FILE_SIZE_EXCEEDS_LIMIT = "File size exceeds the maximum limit."
    FILE_INGESTION_SUCCESS = "File ingested successfully."
    FILE_INGESTION_FAILED = "File ingestion failed."
    FILE_PROCESSING_SUCCESS = "File processed successfully."
    FILE_PROCESSING_FAILED= "File processing failed."
    NO_FILES_ERROR = "Not found files."
    FILE_ID_ERROR = "No file found with the specified id."
    PROJECT_NOT_FOUND = "Project not found."
    VECTORDB_INSERTION_FAILED = "VectorDB insertion failed."
    VECTORDB_INSERTION_SUCCESS = "VectorDB insertion success."
    VECTORDB_SEARCH_SUCCESS = "VectorDB search success."