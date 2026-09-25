from abc import ABC, abstractmethod

from customer_support.models.db_schemas import RetrievedDocument


class VectorDBInterface(ABC):
    @abstractmethod
    async def connect(self):
        pass

    @abstractmethod
    async def disconnect(self):
        pass

    @abstractmethod
    async def is_collection_exists(self, collection_name: str) -> bool:
        pass

    @abstractmethod
    async def list_all_collections(self) -> list:
        pass

    @abstractmethod
    async def get_collection_info(self, collection_name: str) -> dict:
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str):
        pass

    @abstractmethod
    async def create_collection(
        self, collection_name: str, embedding_size: int, do_reset: bool = False
    ):
        pass

    @abstractmethod
    async def insert_one(
        self,
        collection_name: str,
        text: str,
        vector: list,
        metadata: dict = None,
        record_id: str = None,
    ):
        pass

    @abstractmethod
    async def insert_many(
        self,
        collection_name: str,
        texts: list[str],
        vectors: list[str],
        metadata: list[dict] = None,
        record_ids: list[str] = None,
        batch_size: int = 50,
    ):
        pass

    @abstractmethod
    async def delete_by_metadata(self, collection_name: str, criteria: dict) -> int:
        """Delete every row matching ALL of `criteria`. Returns the row count.

        This is the primitive the Section 1 sync is built on: without delete-by-key,
        re-syncing leaves the stale vector behind and the agent can retrieve either
        the old or the corrected text.

        Criteria is a dict, not a single key/value pair, because the stable id is not
        always one field. A promoted ticket answer is keyed on `ticket_id` alone; a
        handbook chunk is keyed on `source` + `section` together. AND semantics —
        deleting on either half of a composite key alone would take out far more
        rows than intended.
        """
        pass

    @abstractmethod
    async def search_by_vector(
        self, collection_name: str, vector: list, limit: int
    ) -> list[RetrievedDocument]:
        pass
