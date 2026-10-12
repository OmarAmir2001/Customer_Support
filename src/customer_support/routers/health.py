"""Liveness and readiness.

Two endpoints, because they answer two different questions and the container
orchestrator needs the cheap one:

* ``GET /`` is **liveness** — is the process up. It touches nothing, so a degraded
  dependency can never restart a container that is otherwise serving fine. This is
  what the compose healthcheck probes.
* ``GET /health`` is **readiness** — is this instance able to answer a question. It
  reports ``documents_indexed``, which is the one dependency the whole system is
  useless without: an empty vector store makes every question escalate, and that
  failure is otherwise invisible because /chat keeps returning 200.
"""

from fastapi import APIRouter, Depends, Request

from customer_support.helpers.config import Settings, get_settings
from customer_support.helpers.logging_config import get_logger

logger = get_logger(__name__)

# Create a router for the base routes
base_router = APIRouter(
    prefix="",  # Prefix for all routes in this router
    tags=["Base Routes"],  # Tag for documentation purposes
)


@base_router.get("/")
async def health_check(app_settings: Settings = Depends(get_settings)):
    """
    Health check endpoint for the Customer Support Agent With Escalation application.
    Returns App name and Version as well as a welcome message.
    """
    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION

    return {
        "APP_NAME": app_name,
        "APP_VERSION": app_version,
        "message": "Welcome to the Customer Support Agent With Escalation !",
    }


@base_router.get("/health")
async def readiness(request: Request, app_settings: Settings = Depends(get_settings)) -> dict:
    """``{status, documents_indexed, ...}``.

    ``status`` is ``"degraded"`` rather than ``"healthy"`` when the count cannot be
    read, and ``documents_indexed`` is then null. Reporting healthy with a null
    count would be the kind of green dashboard that hides the outage — but note the
    compose healthcheck deliberately probes ``/`` and not this, so a vector store
    blip does not get the app container restarted while it is still serving.
    """
    collection = app_settings.KB_COLLECTION_NAME
    # getattr, because this endpoint must answer even before the lifespan has
    # finished wiring state — and during tests, which run without a lifespan at all.
    vectordb_client = getattr(request.app.state, "vectordb_client", None)

    indexed: int | None = None
    if vectordb_client is not None:
        try:
            info = await vectordb_client.get_collection_info(collection_name=collection)
            indexed = (info or {}).get("record_count")
        except Exception as exc:
            # Logged, not raised: a readiness probe that 500s tells a load balancer
            # less than one that answers "degraded".
            logger.warning("readiness_count_failed", collection=collection, error=str(exc))

    return {
        "status": "healthy" if indexed is not None else "degraded",
        "documents_indexed": indexed,
        "collection": collection,
        "version": app_settings.APP_VERSION,
    }
