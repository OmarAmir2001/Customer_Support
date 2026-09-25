"""Retrieval + Section 5 precedence.

Precedence is NOT automatic. pgvector returns the semantically closest rows, and
instructor-resolved chunks are phrased in student language, so they out-retrieve
formal handbook text for question-shaped queries. Handbook chunks are ranked first
here, deliberately: ticket answers fill gaps, they never override the handbook.
"""

import asyncio

from customer_support.helpers.logging_config import get_logger
from customer_support.models.db_schemas import RetrievedDocument
from customer_support.stores.llm.LLMEnum import DocumentTypeEnum

from .BaseController import BaseController

HANDBOOK_SOURCES = ("CS_2023", "IS_2023")
INSTRUCTOR_RESOLVED_SOURCE = "instructor_resolved"


class RetrievalController(BaseController):
    def __init__(self, vectordb_client, embedding_client, collection_name: str, settings=None):
        super().__init__(settings)
        self.vectordb_client = vectordb_client
        self.embedding_client = embedding_client
        self.collection_name = collection_name
        self.logger = get_logger(__name__)

    async def retrieve(
        self, question: str, department: str | None = None, limit: int = 5
    ) -> list[RetrievedDocument]:
        """Return the chunks the answer may be built from.

        Returns an empty list when nothing is found — never False. A falsy-but-typed
        return keeps the caller honest: gate 1 turns "empty" into an escalation.
        """

        if not question or not question.strip():
            raise ValueError("retrieve() requires a non-empty question")

        vectors = await asyncio.to_thread(
            self.embedding_client.embed_text,
            question,
            DocumentTypeEnum.QUERY.value,
        )
        if not vectors:
            raise RuntimeError("query embedding failed")

        # Over-fetch, because department filtering and precedence re-ranking both
        # discard rows. Fetching exactly `limit` would leave gaps after filtering.
        raw = await self.vectordb_client.search_by_vector(
            collection_name=self.collection_name,
            vector=vectors[0],
            limit=limit * self.app_settings.RETRIEVAL_OVERFETCH_FACTOR,
        )
        if not raw:
            self.logger.info("retrieval_empty", department=department)
            return []

        filtered = self._filter_by_department(raw, department)

        # Relevance decides WHICH chunks are used; precedence decides the ORDER they are
        # presented in. That is why the cut to `limit` happens FIRST, on similarity.
        #
        # Ranking before the cut looks equivalent but silently breaks the learning loop:
        # handbook chunks all sort ahead of instructor-resolved ones, so as soon as
        # `limit` handbook chunks survive the filter, a promoted ticket answer can never
        # reach the generator no matter how well it matches. Section 5 wants ticket
        # answers to FILL GAPS the handbook does not cover — unreachable is not
        # "additive-only", it is dead. Ordering after the cut keeps handbook text first
        # whenever both are present, which is the precedence that was actually asked for.
        selected = filtered[:limit]  # already in similarity order from the vector store
        ranked = self._apply_source_precedence(selected)

        self.logger.info(
            "retrieval_complete",
            fetched=len(raw),
            after_filter=len(filtered),
            returned=len(ranked),
            sources=[(doc.metadata or {}).get("source") for doc in ranked],
            department=department,
        )
        return ranked

    @staticmethod
    def _filter_by_department(
        documents: list[RetrievedDocument], department: str | None
    ) -> list[RetrievedDocument]:
        """Keep chunks for this department plus chunks that belong to no department.

        A CS student must not be answered from the IS handbook; shared content has no
        department and stays.
        """
        if not department:
            return documents

        return [
            doc for doc in documents if (doc.metadata or {}).get("department") in (department, None)
        ]

    @staticmethod
    def _apply_source_precedence(
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Handbook chunks first, then instructor-resolved ones, each group keeping
        its similarity order. Stable sort, so scores still decide within a group."""

        def rank(doc: RetrievedDocument) -> int:
            source = (doc.metadata or {}).get("source", "")
            if any(source.startswith(h) for h in HANDBOOK_SOURCES):
                return 0
            if source == INSTRUCTOR_RESOLVED_SOURCE:
                return 1
            return 2

        return sorted(documents, key=rank)
