from pathlib import Path

from sqlalchemy.orm import sessionmaker

from .providers import PgVectorProvider, QdrantDBProvider
from .VectorDBEnum import VectorDBEnums


class VectorDBProviderFactory:
    def __init__(self, config, db_client: sessionmaker = None):
        self.config = config
        self.db_client = db_client

    def create(self, provider: str):
        if provider == VectorDBEnums.QDRANT.value:
            quadrant_db_client = (
                Path(self.config.ASSETS_DIR) / "database" / self.config.VECTOR_DB_PATH
            )
            quadrant_db_client.mkdir(parents=True, exist_ok=True)
            return QdrantDBProvider(
                db_client=str(quadrant_db_client),
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD,
                default_vector_size=self.config.VECTOR_DB_DEFAULT_VECTOR_SIZE,
                index_threshold=self.config.VECTOR_DB_PGVEC_INDEX_THRESHOLD,
            )
        if provider == VectorDBEnums.PGVECTOR.value:
            return PgVectorProvider(
                db_client=self.db_client,
                default_vector_size=self.config.VECTOR_DB_DEFAULT_VECTOR_SIZE,
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD,
                index_threshold=self.config.VECTOR_DB_PGVEC_INDEX_THRESHOLD,
            )
        return None
