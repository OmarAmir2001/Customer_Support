from fastapi import FastAPI
from routers.health import base_router
from routers.history import history_router
from routers.profile import profile_router
from routers.admin import admin_router
from routers.chat import chat_router
from routers.escalation import escalation_router
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from helpers.config import get_settings
from contextlib import asynccontextmanager
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings = get_settings()

    postgress_conn= f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"  # the connection/client

    app.state.db_engine = create_async_engine(postgress_conn)  # the engine
    app.state.db_client = sessionmaker(
       app.state.db_engine,class_=AsyncSession,expire_on_commit=False)  # the actual client

    app.state.llm_provider_factory = LLMProviderFactory(settings)
    app.state.vectordb_provider_factory = VectorDBProviderFactory(settings)

    # Generation Client
    app.state.generation_client = app.state.llm_provider_factory.create(provider_name=settings.GENERATION_BACKEND)
    app.state.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    # Embedding Client
    app.state.embedding_client = app.state.llm_provider_factory.create(provider_name=settings.EMBEDDING_BACKEND)
    app.state.embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID, embedding_size=settings.EMBEDDING_MODEL_SIZE)

    # Vector DB Client
    app.state.vectordb_client = app.state.vectordb_provider_factory.create(provider=settings.VECTOR_DB_BACKEND)

    app.state.vectordb_client.connect()


    yield

    # Shutdown
    await app.state.db_engine.dispose()
    app.state.vectordb_client.disconnect()


app = FastAPI(lifespan=lifespan)

app.include_router(base_router)
app.include_router(profile_router)
app.include_router(admin_router)
app.include_router(chat_router)
app.include_router(escalation_router)
app.include_router(history_router)
