"""Entry point: build expensive things once, wire the layers, hand them to the app.

Startup order matters — each line below depends on the one above it. Anything that
fails here fails at boot with a clear traceback, not in a request three weeks later.
"""

from contextlib import AsyncExitStack, asynccontextmanager

from customer_support.helpers.config import get_settings
from customer_support.routers.admin import admin_router
from customer_support.routers.health import base_router
from customer_support.routers.history import history_router
from customer_support.routers.profile import profile_router
from customer_support.stores.llm.LLMProviderFactory import LLMProviderFactory
from customer_support.stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from customer_support.controllers.EscalationController import EscalationController
from customer_support.controllers.GenerationController import GenerationController
from customer_support.controllers.GradingController import GradingController
from customer_support.controllers.RetrievalController import RetrievalController
from customer_support.graph.builder import build_graph
from customer_support.graph.dependencies import GraphDeps
from customer_support.helpers.logging_config import configure_logging, get_logger
from customer_support.models.TicketModel import TicketModel
from customer_support.routers.chat import chat_router
from customer_support.routers.escalation import escalation_router
from customer_support.stores.checkpointer import checkpointer_context

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(level=settings.LOG_LEVEL, json_logs=settings.LOG_JSON)

    # AsyncExitStack closes everything opened here, in reverse order, even if a later
    # startup step raises. Without it a failed boot leaks connections on every reload.
    async with AsyncExitStack() as stack:
        # --- database ---
        dsn = (
            f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}"
            f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
        )
        engine = create_async_engine(dsn, pool_pre_ping=True)  # pre_ping: survive DB restarts
        stack.push_async_callback(engine.dispose)

        db_client = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        app.state.db_engine = engine
        app.state.db_client = db_client

        # --- LLM clients ---
        llm_factory = LLMProviderFactory(settings)

        generation_client = llm_factory.create(provider_name=settings.GENERATION_BACKEND)
        if generation_client is None:
            raise RuntimeError(f"unknown GENERATION_BACKEND: {settings.GENERATION_BACKEND}")
        generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

        embedding_client = llm_factory.create(provider_name=settings.EMBEDDING_BACKEND)
        if embedding_client is None:
            raise RuntimeError(f"unknown EMBEDDING_BACKEND: {settings.EMBEDDING_BACKEND}")
        embedding_client.set_embedding_model(
            model_id=settings.EMBEDDING_MODEL_ID, embedding_size=settings.EMBEDDING_MODEL_SIZE
        )

        # A separate, cheaper model for the three judges. Judges run up to three times
        # per question, so this is the difference between a viable and a costly system.
        judge_client = llm_factory.create(provider_name=settings.GENERATION_BACKEND)
        judge_client.set_generation_model(
            model_id=settings.JUDGE_MODEL_ID or settings.GENERATION_MODEL_ID
        )

        # --- vector store ---
        vectordb_client = VectorDBProviderFactory(settings, db_client=db_client).create(
            provider=settings.VECTOR_DB_BACKEND
        )
        if vectordb_client is None:
            raise RuntimeError(f"unknown VECTOR_DB_BACKEND: {settings.VECTOR_DB_BACKEND}")
        await vectordb_client.connect()
        stack.push_async_callback(vectordb_client.disconnect)

        # The admin router reaches for these three off app.state (ingest, push, search),
        # so they are published here rather than only handed to the controllers.
        app.state.vectordb_client = vectordb_client
        app.state.generation_client = generation_client
        app.state.embedding_client = embedding_client

        # --- data models ---
        ticket_model = await TicketModel.create_instance(db_client=db_client)
        app.state.ticket_model = ticket_model

        # --- controllers (all logic lives here) ---
        collection_name = settings.KB_COLLECTION_NAME

        retrieval = RetrievalController(
            vectordb_client=vectordb_client,
            embedding_client=embedding_client,
            collection_name=collection_name,
            settings=settings,
        )
        grading = GradingController(generation_client=judge_client, settings=settings)
        generation = GenerationController(generation_client=generation_client, settings=settings)
        escalation = EscalationController(
            ticket_model=ticket_model,
            vectordb_client=vectordb_client,
            embedding_client=embedding_client,
            collection_name=collection_name,
            settings=settings,
        )
        app.state.escalation_controller = escalation

        # --- graph ---
        checkpointer = await stack.enter_async_context(checkpointer_context(settings))
        graph = build_graph(
            deps=GraphDeps(
                retrieval=retrieval,
                grading=grading,
                generation=generation,
                escalation=escalation,
                settings=settings,
            ),
            checkpointer=checkpointer,
        )
        app.state.graph = graph

        # Closing the cycle: the graph needs the escalation controller to build its
        # escalate node, and the controller needs the graph to write an advisor's answer
        # back into the student's thread (Section 3). Assigned here rather than passed in
        # because the graph does not exist yet when the controller is constructed.
        escalation.thread_writer = graph

        logger.info(
            "application_ready",
            vector_db=settings.VECTOR_DB_BACKEND,
            generation_model=settings.GENERATION_MODEL_ID,
        )
        yield

    logger.info("application_stopped")


settings = get_settings()
app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)

app.include_router(base_router)
app.include_router(profile_router)
app.include_router(admin_router)
app.include_router(chat_router)
app.include_router(escalation_router)
app.include_router(history_router)