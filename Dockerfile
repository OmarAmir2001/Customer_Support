# syntax=docker/dockerfile:1

# uv's own image: it ships uv plus the matching CPython, so there is no
# pip-bootstrap step and no chance of the runtime drifting from .python-version.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies in their own layer, before the source. This is the slow step, and
# keeping it ahead of `COPY . .` means editing a controller does not re-resolve
# and re-download the whole dependency tree.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY . .

# Install the project itself. Fast — its dependencies are already in the layer above.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# alembic.ini is gitignored (it can hold a real database URL), so a clean clone has
# only the template. That is all we need: migrations/env.py overwrites
# sqlalchemy.url from Settings, so the .ini only supplies script_location and
# logging config. Baking it here keeps the image reproducible from a fresh clone.
RUN cp alembic.ini.example alembic.ini

# Run as a non-root user. The assets directory is created here, before the named
# volume is attached, so Docker seeds the volume with this ownership instead of
# root's — otherwise uploads fail with EACCES on first boot.
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /app/assets/files /app/assets/database \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "customer_support.main:app", "--host", "0.0.0.0", "--port", "8000"]
