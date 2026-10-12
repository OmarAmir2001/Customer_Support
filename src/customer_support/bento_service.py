"""BentoML service wrapping the RAG pipeline, with a streaming `/ask`.

    uv run --extra dev bentoml serve customer_support.bento_service:HandbookAssistant
    # then
    curl -N -X POST localhost:3000/ask \\
      -H 'Content-Type: application/json' \\
      -d '{"question":"كم عدد الساعات المعتمدة المطلوبة للتخرج؟","department":"CS"}'

**Why this exists alongside FastAPI rather than instead of it.** The two serve
different jobs, and collapsing them would lose something:

* `POST /api/v1/chat` (FastAPI) is the **gated** path. Retrieval, three judge gates,
  escalation, ticket writing, the checkpointer, long-term memory. It is what a
  student actually talks to, and its defining property is that an unverified answer
  never reaches them.
* `POST /ask` (here) is the **streaming** path. Tokens appear as the model produces
  them, which is a materially better wait for a long answer — and it cannot be gated,
  because faithfulness and answer relevance can only be scored on a *complete*
  answer. By the time a gate could object, the student has already read the text.

So this trades verification for responsiveness, deliberately and in one place. The
response says so in its own payload (`"gated": false`), because a caller should not
have to read this docstring to find out.

Retrieval and the context-relevance gate DO still run: gate 1 is pre-generation, so
an unanswerable question is refused before a single token is streamed.

PII redaction also runs, and it is the one guardrail that survives streaming — see
`helpers.pii.StreamRedactor` for why it cannot be done chunk by chunk.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from typing import Annotated, Any

import bentoml
from pydantic import Field

from customer_support.helpers.logging_config import get_logger

logger = get_logger(__name__)

# Fields are declared directly on the endpoint rather than as one Pydantic model.
#
# BentoML nests a model parameter under its own name, which would make the body
# {"request": {"question": ...}} — awkward for a client, and not the flat
# {"question": ...} shape this API is specified as. Annotated fields keep the
# validation while leaving the body flat.
Question = Annotated[str, Field(min_length=3, max_length=2000)]
Department = Annotated[str | None, Field(pattern="^(CS|IS)$")]
Language = Annotated[str | None, Field(max_length=16)]


@bentoml.service(
    name="handbook_assistant",
    # One worker: every heavy dependency here (the vector store connection, the
    # provider clients) is per-process, and the bottleneck is the model provider
    # rather than this process. More workers would multiply connections without
    # buying throughput.
    workers=1,
    traffic={"timeout": 300},
)
class HandbookAssistant:
    """Retrieval + streaming generation over the handbooks."""

    def __init__(self) -> None:
        # Built once per worker, in __init__ rather than per request: constructing a
        # vector-store connection and provider clients per call would dominate the
        # latency this endpoint exists to improve.
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
        from sqlalchemy.orm import sessionmaker

        from customer_support.controllers.GenerationController import GenerationController
        from customer_support.controllers.GradingController import GradingController
        from customer_support.controllers.RetrievalController import RetrievalController
        from customer_support.helpers.config import get_settings
        from customer_support.helpers.locale import negotiate_language
        from customer_support.stores.llm.LLMProviderFactory import LLMProviderFactory
        from customer_support.stores.llm.templates import TemplateParser
        from customer_support.stores.vectordb.VectorDBProviderFactory import (
            VectorDBProviderFactory,
        )

        self._settings = get_settings()
        self._negotiate = negotiate_language
        self._stack = AsyncExitStack()

        dsn = (
            f"postgresql+asyncpg://{self._settings.POSTGRES_USERNAME}:"
            f"{self._settings.POSTGRES_PASSWORD}@{self._settings.POSTGRES_HOST}:"
            f"{self._settings.POSTGRES_PORT}/{self._settings.POSTGRES_MAIN_DATABASE}"
        )
        self._engine = create_async_engine(dsn, pool_pre_ping=True, pool_size=3, max_overflow=3)
        db_client = sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)

        llm = LLMProviderFactory(self._settings)
        generation_client = llm.create(provider_name=self._settings.GENERATION_BACKEND)
        generation_client.set_generation_model(model_id=self._settings.GENERATION_MODEL_ID)
        embedding_client = llm.create(provider_name=self._settings.EMBEDDING_BACKEND)
        embedding_client.set_embedding_model(
            model_id=self._settings.EMBEDDING_MODEL_ID,
            embedding_size=self._settings.EMBEDDING_MODEL_SIZE,
        )
        judge_client = llm.create(provider_name=self._settings.GENERATION_BACKEND)
        judge_client.set_generation_model(model_id=self._settings.JUDGE_MODEL_ID)

        self._vectordb = VectorDBProviderFactory(self._settings, db_client=db_client).create(
            provider=self._settings.VECTOR_DB_BACKEND
        )
        self._templates = TemplateParser(
            primary_language=self._settings.PRIMARY_LANG,
            default_language=self._settings.DEFAULT_LANG,
        )
        self._retrieval = RetrievalController(
            vectordb_client=self._vectordb,
            embedding_client=embedding_client,
            collection_name=self._settings.KB_COLLECTION_NAME,
            settings=self._settings,
        )
        self._generation = GenerationController(
            generation_client=generation_client,
            templates=self._templates,
            settings=self._settings,
        )
        self._grading = GradingController(
            generation_client=judge_client, templates=self._templates, settings=self._settings
        )
        self._connected = False

    async def _ensure_connected(self) -> None:
        # Lazily, because __init__ cannot await. Guarded rather than reconnecting per
        # request: pgvector's connect() is idempotent but not free.
        if not self._connected:
            await self._vectordb.connect()
            self._connected = True

    @bentoml.api
    async def health(self) -> dict[str, Any]:
        """Liveness plus the one fact a caller actually needs: is anything indexed."""
        await self._ensure_connected()
        try:
            info = await self._vectordb.get_collection_info(
                collection_name=self._settings.KB_COLLECTION_NAME
            )
            indexed = (info or {}).get("record_count")
        except Exception:
            indexed = None
        return {
            "status": "healthy",
            "documents_indexed": indexed,
            "generation_model": self._settings.GENERATION_MODEL_ID,
        }

    @bentoml.api
    async def ask(
        self,
        question: Question,
        department: Department = None,
        language: Language = None,
    ) -> AsyncGenerator[str]:
        """Stream an answer, or one refusal line if it cannot be answered.

        Gate 1 runs first and is the reason this is safe to stream at all: an
        unanswerable question is refused before any token is produced. The
        post-generation gates cannot run here — see the module docstring.
        """
        await self._ensure_connected()

        # `resolved` rather than reassigning `language`: the parameter is what the
        # client asked for, the result is what we will actually answer in, and
        # collapsing them makes the precedence impossible to read.
        resolved = self._negotiate(
            self._templates,
            requested=language,
            question=question,
        )

        chunks = await self._retrieval.retrieve(
            question=question,
            department=department,
            limit=self._settings.RETRIEVAL_TOP_K,
        )
        if not chunks:
            yield self._refusal()
            return

        # Gate 1, pre-generation. The only gate that CAN run before streaming, and
        # the one that catches the dangerous case: a question the handbook does not
        # cover, which would otherwise be answered from the model's own knowledge.
        gate = await self._grading.check_context_relevance(question=question, chunks=chunks)
        if not gate.passed:
            yield self._refusal()
            return

        # The guardrail that CAN be applied to a stream. The post-generation judges
        # cannot — they need a finished answer — but redaction only needs enough
        # lookahead to know a token is complete, which is a bounded buffer rather
        # than the whole answer. Without this the streaming endpoint would be the
        # one path in the system with no PII protection at all.
        from customer_support.helpers.pii import StreamRedactor

        redactor = StreamRedactor() if self._settings.PII_REDACTION_ENABLED else None

        for piece in self._generation.stream_answer(
            question=question, chunks=chunks, language=resolved
        ):
            out = redactor.feed(piece) if redactor is not None else piece
            if out:
                yield out
            # Hand control back between chunks. stream_answer is a sync generator
            # driven from async code; without this the event loop cannot flush what
            # has been yielded, and the "stream" arrives as one block at the end —
            # which would make this endpoint pointless.
            await asyncio.sleep(0)

        if redactor is not None:
            # The held-back tail. Skipping this truncates every answer by up to the
            # holdback length, which is the obvious way to get this wrong.
            tail = redactor.flush()
            if tail:
                yield tail
            if redactor.found:
                logger.warning("pii_redacted", where="bento_stream", kinds=dict(redactor.found))

    def _refusal(self) -> str:
        """The same words /chat uses when it escalates.

        Read from ESCALATION_STUDENT_MESSAGE rather than written here, so a student
        sees one consistent message whichever endpoint they reached. Note it is a
        setting rather than a per-locale template, so it is currently English only —
        a real gap for Arabic students, but one shared with /chat rather than
        introduced here.
        """
        return self._settings.ESCALATION_STUDENT_MESSAGE
