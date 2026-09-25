"""structlog configuration.

Two things make this production-shaped rather than "structlog instead of logging":

1. **One renderer for every log line.** Uvicorn, SQLAlchemy and httpx log through the
   standard library, not structlog. ``ProcessorFormatter`` runs those records through
   the same processor chain, so a request log and a gate verdict come out in the same
   JSON shape — otherwise half your output is unparseable.
2. **Context variables instead of passing a logger around.** ``bind_contextvars``
   attaches the correlation id (the graph's ``thread_id``) to every event emitted in
   the same async task, including inside controllers that never heard of a request.

Call ``configure_logging()`` exactly once, at application startup.
"""

import logging
import sys

import structlog

# A structlog logger. Modules do: log = get_logger(__name__)
get_logger = structlog.get_logger


def set_correlation_id(value: str) -> None:
    """Bind the correlation id for the current async task.

    Everything logged from here on in this request carries it — no plumbing through
    function signatures, and no leakage between concurrent requests.
    """
    structlog.contextvars.bind_contextvars(correlation_id=value)


def clear_correlation_id() -> None:
    structlog.contextvars.clear_contextvars()


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Configure structlog and route stdlib logging through the same renderer.

    Idempotent: uvicorn --reload calls it on every restart.
    """

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    # Applied to structlog events AND (via foreign_pre_chain) to stdlib records.
    shared_processors = [
        structlog.contextvars.merge_contextvars,  # pulls in correlation_id
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            # Hand the event dict to ProcessorFormatter rather than rendering here,
            # so structlog and stdlib share one final renderer.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=[
            *shared_processors,
            structlog.stdlib.ExtraAdder(),  # keeps extra={...} from third-party libs
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()  # replace uvicorn's handler rather than logging twice
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Uvicorn installs its own handlers; clearing them stops duplicate lines.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    # Noisy at DEBUG, and they say nothing the app logs don't.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
