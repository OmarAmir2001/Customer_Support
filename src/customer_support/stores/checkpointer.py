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


#: Arbitrary but fixed key for the advisory lock around ``setup()``. Any process
#: using the same number contends on the same lock; nothing else in this app uses one.
SETUP_ADVISORY_LOCK = 8_531_2024


@asynccontextmanager
async def checkpointer_context(settings: Settings):
    """Open the saver for the lifetime of the app.

    ``setup()`` creates the checkpoint tables if they are missing — idempotent, so it
    is safe on every boot. Those tables are managed by LangGraph, not Alembic; see the
    ``include_object`` filter in migrations/env.py.

    Idempotent is not the same as concurrency-safe, though, and uvicorn runs with
    ``--workers``: every worker boots at once and issues the same CREATE TABLE against
    the same database. Concurrent DDL on one table either deadlocks or raises
    DuplicateTable, and it would do it at start-up on a fresh volume — the one moment
    nobody is watching the logs. A session-level advisory lock serialises them, so the
    first worker creates the tables and the rest wait, then find nothing to do.
    """

    async with AsyncPostgresSaver.from_conn_string(build_checkpointer_dsn(settings)) as saver:
        async with saver.conn.cursor() as cur:
            await cur.execute("SELECT pg_advisory_lock(%s)", (SETUP_ADVISORY_LOCK,))
        try:
            await saver.setup()
        finally:
            # Released explicitly rather than left to connection close: this
            # connection lives for the whole process, so the lock would be held for
            # the lifetime of the app and the next deploy's workers would all block.
            async with saver.conn.cursor() as cur:
                await cur.execute("SELECT pg_advisory_unlock(%s)", (SETUP_ADVISORY_LOCK,))

        logger.info("checkpointer_ready")
        yield saver
