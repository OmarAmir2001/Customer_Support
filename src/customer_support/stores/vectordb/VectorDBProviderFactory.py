from pathlib import Path

from .providers import QdrantDBProvider
from .VectorDBEnum import VectorDBEnums


class VectorDBProviderFactory:
    def __init__(self, config):
        self.config = config

    def create(self, provider: str):
        if provider == VectorDBEnums.QDRANT.value:
            db_path = Path(self.config.ASSETS_DIR) / "database" / self.config.VECTOR_DB_PATH
            db_path.mkdir(parents=True, exist_ok=True)
            return QdrantDBProvider(
                db_path=str(db_path),
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD,
            )
        return None