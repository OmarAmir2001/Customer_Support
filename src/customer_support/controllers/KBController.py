import asyncio
import json

from customer_support.helpers.logging_config import get_logger
from customer_support.models.db_schemas import DataChunk, Project
from customer_support.stores.llm.LLMEnum import DocumentTypeEnum

from .BaseController import BaseController


class KBController(BaseController):
    def __init__(self, vectordb_client, generation_client, embedding_client):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.logger = get_logger(__name__)

    def create_collection_name(self, project_id: str):
        return f"collection_{project_id}".strip()

    async def reset_vectordb_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return await self.vectordb_client.delete_collection(collection_name=collection_name)

    async def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = await self.vectordb_client.get_collection_info(
            collection_name=collection_name
        )

        # The await belongs on the provider call, not on json.loads: awaiting a dict
        # raises TypeError, which is what made this endpoint a guaranteed 500.
        return json.loads(json.dumps(collection_info, default=lambda o: o.__dict__))

    async def index_into_vector_db(
        self,
        project: Project,
        chunks: list[DataChunk],
        chunks_ids: list[int],
        do_reset: bool = False,
    ):
        # step 1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step 2 : get the data from the chunks
        texts = [chunk.chunk_text for chunk in chunks]
        metadatas = [chunk.chunk_metadata for chunk in chunks]
        vectors = self.embedding_client.embed_text(
            text=texts, document_type=DocumentTypeEnum.DOCUMENT.value
        )
        self.logger.info(
            "kb_embedding_complete",
            collection=collection_name,
            text_count=len(texts),
            vector_type=type(vectors).__name__,
        )
        if not vectors or len(vectors) != len(texts):
            self.logger.error(
                "kb_embedding_failed",
                collection=collection_name,
                text_count=len(texts),
                vector_count=len(vectors) if vectors else 0,
            )
            return False

        # step 3: create the collection if it doesn't exist
        _ = await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        # step 4: insert the data into the collection
        _ = await self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            vectors=vectors,
            metadata=metadatas,
            record_ids=chunks_ids,
        )
        return True

    async def _embed_in_batches(self, texts: list[str], batch_size: int = 96):
        """Embed every text, in batches the provider will accept.

        96 is Cohere's per-call ceiling. Sending more fails outright; sending one at
        a time is what tripped the rate limiter. Returns None on any failure — a
        partial embedding would index some sections and silently skip others, which
        is worse than indexing none.
        """
        if not texts:
            return []

        vectors: list = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            # to_thread: the provider SDK is synchronous, and embedding inline would
            # block the event loop for the whole round trip.
            batch_vectors = await asyncio.to_thread(
                self.embedding_client.embed_text,
                batch,
                DocumentTypeEnum.DOCUMENT.value,
            )
            if not batch_vectors or len(batch_vectors) != len(batch):
                self.logger.error(
                    "embedding_batch_failed",
                    batch_start=start,
                    expected=len(batch),
                    received=len(batch_vectors) if batch_vectors else 0,
                )
                return None
            vectors.extend(batch_vectors)

        return vectors

    async def sync_sections(
        self,
        project: Project,
        chunks: list[DataChunk],
        chunks_ids: list[int],
        do_reset: bool = False,
    ):
        """Section 1's handbook sync: delete-then-insert, keyed on source + section.

        ``index_into_vector_db`` only ever INSERTS, so re-running it after a handbook
        edit leaves the old vectors in place and the agent can retrieve either the
        stale rule or the corrected one. The only escape was ``do_reset``, which
        rebuilds the entire collection — correct but heavy-handed, and it also wipes
        every promoted ticket answer as collateral.

        This replaces one section at a time, which makes re-ingestion idempotent
        WITHOUT a reset: running it once or ten times leaves the same rows. That is
        the same primitive as ``sync_ticket_to_vectors``, with a composite key
        instead of a single id — "delete by stable id, then insert the current
        state", so the caller never has to know which case it is in.

        Promoted ticket answers are untouched: they carry
        ``source: instructor_resolved`` and no ``section``, so no handbook group's
        criteria can match them.
        """
        collection_name = self.create_collection_name(project_id=project.project_id)

        await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        # Embed EVERYTHING first, in provider-sized batches, before touching the
        # collection. Embedding and syncing want opposite batch sizes: a section is
        # the right unit to delete-and-replace, and the worst possible unit to embed
        # — one API round trip per section means ~90 calls for two handbooks, which
        # is slow and trips the provider's rate limit outright. Separating the two
        # turns that into three calls.
        all_vectors = await self._embed_in_batches([chunk.chunk_text for chunk in chunks])
        if all_vectors is None:
            self.logger.error(
                "section_sync_embedding_failed",
                collection=collection_name,
                text_count=len(chunks),
            )
            return False

        groups: dict[tuple, dict] = {}
        # strict=True: a length mismatch between chunks and ids is a caller bug,
        # and plain zip() would silently drop the tail rather than say so.
        for chunk, chunk_id, vector in zip(chunks, chunks_ids, all_vectors, strict=True):
            metadata = chunk.chunk_metadata or {}
            key = (metadata.get("source"), metadata.get("section"))
            group = groups.setdefault(key, {"texts": [], "metadatas": [], "ids": [], "vectors": []})
            group["texts"].append(chunk.chunk_text)
            group["metadatas"].append(metadata)
            group["ids"].append(chunk_id)
            group["vectors"].append(vector)

        deleted = inserted = 0
        for (source, section), group in groups.items():
            criteria = {
                key: value
                for key, value in (("source", source), ("section", section))
                if value is not None
            }
            if not criteria:
                # Nothing stable to key on, so a delete would be unbounded. Insert
                # only, and say so — this is what an untagged chunk looks like.
                self.logger.warning(
                    "section_sync_unkeyed",
                    collection=collection_name,
                    chunk_count=len(group["texts"]),
                )
            else:
                deleted += await self.vectordb_client.delete_by_metadata(
                    collection_name=collection_name, criteria=criteria
                )

            await self.vectordb_client.insert_many(
                collection_name=collection_name,
                texts=group["texts"],
                vectors=group["vectors"],
                metadata=group["metadatas"],
                record_ids=group["ids"],
            )
            inserted += len(group["texts"])

        self.logger.info(
            "sections_synced",
            collection=collection_name,
            sections=len(groups),
            deleted=deleted,
            inserted=inserted,
        )
        return {"sections": len(groups), "deleted": deleted, "inserted": inserted}

    # def search_vector_db_collection(self,project:Project,query:str,limit:int=10):

    #     collection_name = self.create_collection_name(project_id=project.project_id)

    #     vector = self.embedding_client.embed_text(text=query,
    #                                               document_type=DocumentTypeEnum.QUERY.value)[0]

    #     if not vector or len(vector) == 0:
    #         self.logger.error("Error while embedding query")
    #         return False

    #     results = self.vectordb_client.search_by_vector(collection_name=collection_name,
    #                                                     vector=vector,
    #                                                     limit=limit)
    #     if not results or len(results) == 0:
    #         return False

    #     return json.loads(json.dumps(results, default=lambda o: o.__dict__))
