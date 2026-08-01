from fastapi import FastAPI
from routers.health import base_router
from routers.history import history_router
from routers.profile import profile_router
from routers.admin import admin_router
from routers.chat import chat_router
from routers.escalation import escalation_router
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from contextlib import asynccontextmanager
from stores.llm.LLMProviderFactory import LLMProviderFactory

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings = get_settings()
    app.state.db_client = AsyncIOMotorClient(settings.MONGODB_URL)   # the connection/client
    app.state.db = app.state.db_client[settings.MONGODB_DATABASE]     # the actual database

    app.state.llm_provider_factory = LLMProviderFactory(settings)

    # Generation Client
    app.state.generation_client = app.state.llm_provider_factory.create(provider_name=settings.GENERATION_BACKEND)
    app.state.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    # Embedding Client
    app.state.embedding_client = app.state.llm_provider_factory.create(provider_name=settings.EMBEDDING_BACKEND)
    app.state.embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID, embedding_size=settings.EMBEDDING_MODEL_SIZE)

    yield

    # Shutdown
    app.state.db_client.close()   # now correctly closes the real client
app = FastAPI(lifespan=lifespan)

app.include_router(base_router)
app.include_router(profile_router)
app.include_router(admin_router)
app.include_router(chat_router)
app.include_router(escalation_router)
app.include_router(history_router)
