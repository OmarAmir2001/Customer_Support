from enum import Enum

class ResponseSignel(Enum):
    FILE_VALIDATED_SUCCESS = "File validated successfully."
    FILE_VALIDATED_FAILURE = "File validation failed."
    FILE_TYPE_NOT_ALLOWED = "File type is not allowed."
    FILE_SIZE_EXCEEDS_LIMIT = "File size exceeds the maximum limit."
    FILE_INGESTION_SUCCESS = "File ingested successfully."
    FILE_INGESTION_FAILURE = "File ingestion failed."