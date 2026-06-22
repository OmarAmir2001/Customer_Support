from fastapi import FastAPI
from dotenv import load_dotenv
from src.routers.health import base_router
from src.routers.history import history_router
from src.routers.profile import profile_router
from src.routers.admin import admin_router
from src.routers.chat import chat_router
from src.routers.escalation import escalation_router

load_dotenv(".env")
app = FastAPI()
app.include_router(base_router)
app.include_router(history_router)
app.include_router(profile_router)
app.include_router(admin_router)
app.include_router(chat_router)
app.include_router(escalation_router)
