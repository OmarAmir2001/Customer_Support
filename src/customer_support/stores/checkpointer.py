"""LangGraph checkpointer setup.

The checkpointer is not "a paused function": it stores the conversation state that
two separate runs attach to — the student's question now, the advisor's resolution
hours later — keyed by ``thread_id``.

Driver note: AsyncPostgresSaver is built on psycopg 3, while the app uses asyncpg and
Alembic uses psycopg2. Same database, three drivers; all three URLs are built from
the same Settings so they cannot drift.
"""

from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from customer_support.helpers.config import Settings
from customer_support.helpers.logging_config import get_logger

logger = get_logger(__name__)


def build_checkpointer_dsn(settings: Settings) -> str:
    return (
        f"postgresql://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    )


@asynccontextmanager
async def checkpointer_context(settings: Settings):
    """Open the saver for the lifetime of the app.

    ``setup()`` creates the checkpoint tables if they are missing — idempotent, so it
    is safe on every boot. Those tables are managed by LangGraph, not Alembic; see the
    ``include_object`` filter in migrations/env.py.
    """

    async with AsyncPostgresSaver.from_conn_string(build_checkpointer_dsn(settings)) as saver:
        await saver.setup()
        logger.info("checkpointer_ready")
        yield saver