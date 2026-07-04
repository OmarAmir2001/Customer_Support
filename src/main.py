from fastapi import FastAPI
from routers.health import base_router, history_router, profile_router, admin_router ,chat_router, escalation_router
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings

app = FastAPI()

@app.lifespan("startup")
async def startup_db_client():
    settings = get_settings()
    app.state.mongodb_client = AsyncIOMotorClient(settings.MONGODB_URL)
    app.state.mongodb_database = app.state.mongodb_client[settings.MONGODB_DATABASE]

@app.lifespan("shutdown")
async def shutdown_db_client():
    app.state.mongodb_client.close()

app.include_router(base_router)
app.include_router(history_router)
app.include_router(profile_router)
app.include_router(admin_router)
app.include_router(chat_router)
app.include_router(escalation_router)
