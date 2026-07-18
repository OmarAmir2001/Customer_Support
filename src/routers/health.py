from fastapi import APIRouter, Depends
from helpers.config import get_settings, Settings
# Create a router for the base routes
base_router = APIRouter(
    prefix="",  # Prefix for all routes in this router
    tags=["Base Routes"]  # Tag for documentation purposes
)
@base_router.get("/")
async def health_check(app_settings:Settings = Depends(get_settings)):
    """
    Health check endpoint for the Customer Support Agent With Escalation application.
    Returns App name and Version as well as a welcome message.
    """
    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION

    return {'APP_NAME': app_name,
            'APP_VERSION': app_version,
            "message": "Welcome to the Customer Support Agent With Escalation !"}