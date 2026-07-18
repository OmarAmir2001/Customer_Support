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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings = get_settings()
    app.state.db_client = AsyncIOMotorClient(settings.MONGODB_URL)   # the connection/client
    app.state.db = app.state.db_client[settings.MONGODB_DATABASE]     # the actual database

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
