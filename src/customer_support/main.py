"""Entry point: build expensive things once, wire the layers, hand them to the app.

Startup order matters — each line below depends on the one above it. Anything that
fails here fails at boot with a clear traceback, not in a request three weeks later.
"""

from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from customer_support.controllers.EscalationController import EscalationController
from customer_support.controllers.GenerationController import GenerationController
from customer_support.controllers.GradingController import GradingController
from customer_support.controllers.MemoryController import MemoryController
from customer_support.controllers.PromotionController import PromotionController
from customer_support.controllers.RetrievalController import RetrievalController
from customer_support.graph.builder import build_graph
from customer_support.graph.dependencies import GraphDeps
from customer_support.helpers.config import get_settings
from customer_support.helpers.logging_config import configure_logging, get_logger
from customer_support.models.ProfileModel import ProfileModel
from customer_support.models.TicketModel import TicketModel
from customer_support.routers.admin import admin_router
from customer_support.routers.chat import chat_router
from customer_support.routers.escalation import escalation_router
from customer_support.routers.health import base_router
from customer_support.routers.profile import profile_router
from customer_support.stores.checkpointer import checkpointer_context
from customer_support.stores.llm.LLMProviderFactory import LLMProviderFactory
from customer_support.stores.llm.templates import TemplateParser
from customer_support.stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory

# import metrics setup
from .utils.metrics import setup_metrics

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
        # The pool is sized per WORKER, and uvicorn runs several of them, so the real
        # budget is (pool_size + max_overflow) x workers + one checkpointer connection
        # each, against Postgres' default max_connections of 100. SQLAlchemy's
        # defaults (5 + 10) would spend 15 per worker and leave almost nothing for
        # psql, the exporter or a migration — failing as "too many clients" only
        # under the load that needs it most. 5 + 5 here is 10 per worker, so the
        # --workers count in the Dockerfile can change without re-doing this sum:
        # at the current 4 workers that is 44 of 100.
        engine = create_async_engine(
            dsn,
            pool_pre_ping=True,  # pre_ping: survive DB restarts
            pool_size=5,
            max_overflow=5,
        )
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

        # Memory extraction is an EASY task (pull 'department: CS' out of a
        # sentence) while answering handbook questions is a hard one, so it gets
        # its own right-sized client rather than borrowing the 120b generator.
        memory_client = llm_factory.create(provider_name=settings.GENERATION_BACKEND)
        memory_client.set_generation_model(
            model_id=settings.MEMORY_MODEL_ID
            or settings.JUDGE_MODEL_ID
            or settings.GENERATION_MODEL_ID
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

        profile_model = await ProfileModel.create_instance(db_client=db_client)
        app.state.profile_model = profile_model

        # --- prompt templates ---
        # One shared instance is safe because TemplateParser is stateless: the
        # language is an argument to every call, never instance state. A parser
        # with a set_language() mutated per request would be a data race here.
        templates = TemplateParser(
            primary_language=settings.PRIMARY_LANG,
            default_language=settings.DEFAULT_LANG,
        )
        # The chat router negotiates the locale per request, so it needs this.
        app.state.templates = templates

        # --- controllers (all logic lives here) ---
        collection_name = settings.KB_COLLECTION_NAME

        retrieval = RetrievalController(
            vectordb_client=vectordb_client,
            embedding_client=embedding_client,
            collection_name=collection_name,
            settings=settings,
        )
        grading = GradingController(
            generation_client=judge_client, templates=templates, settings=settings
        )
        generation = GenerationController(
            generation_client=generation_client, templates=templates, settings=settings
        )
        # Section 5's quality gate, in front of the automatic ingestion path.
        # It reuses the judge client (same cheap model, same JSON contract) and the
        # retrieval controller, because "does this contradict the handbook" is a
        # retrieval question before it is a judging one.
        promotion = PromotionController(
            judge_client=judge_client,
            retrieval=retrieval,
            templates=templates,
            settings=settings,
        )
        app.state.promotion_controller = promotion

        escalation = EscalationController(
            ticket_model=ticket_model,
            vectordb_client=vectordb_client,
            embedding_client=embedding_client,
            collection_name=collection_name,
            settings=settings,
            promotion=promotion,
        )
        app.state.escalation_controller = escalation

        # Memory sits outside GraphDeps on purpose. Loading happens in the router
        # (the locale decision needs the profile BEFORE the graph starts) and
        # saving happens after the response is sent, so neither is a graph node.
        app.state.memory = MemoryController(
            extraction_client=memory_client,
            profile_model=profile_model,
            settings=settings,
        )
        # The chat router reads MEMORY_ENABLED to decide whether to schedule
        # extraction at all.
        app.state.settings = settings

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
            locales=list(templates.supported_languages),
            primary_language=templates.primary_language,
            memory_enabled=settings.MEMORY_ENABLED,
            memory_model=settings.MEMORY_MODEL_ID
            or settings.JUDGE_MODEL_ID
            or settings.GENERATION_MODEL_ID,
        )
        yield

    logger.info("application_stopped")


settings = get_settings()
app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)
setup_metrics(app)

app.include_router(base_router)
app.include_router(profile_router)
app.include_router(admin_router)
app.include_router(chat_router)
app.include_router(escalation_router)
