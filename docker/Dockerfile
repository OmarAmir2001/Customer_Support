# syntax=docker/dockerfile:1
#
# Built with the REPO ROOT as context, not this directory — the source, uv.lock
# and handbooks all live up there. compose sets `context: ..` for that reason, and
# .dockerignore stays at the root because Docker resolves it against the context.

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
 && mkdir -p /app/assets/files /app/assets/database /tmp/prometheus \
 && chown -R appuser:appuser /app /tmp/prometheus
USER appuser

# --workers runs N separate processes, and prometheus_client keeps its counters in
# process memory. A scrape reaches whichever worker the kernel hands the socket to,
# so without this every number on the dashboard would be roughly 1/N of the truth,
# and a different Nth each scrape. In multiprocess mode the workers write to mmap
# files here and the scrape endpoint sums them.
#
# Deliberately a tmpfs-ish path and not a volume: these files describe the CURRENT
# set of processes, so they must not survive a restart. The entrypoint clears them.
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus

EXPOSE 8000

ENTRYPOINT ["/app/docker/docker-entrypoint.sh"]
CMD ["uvicorn", "customer_support.main:app", "--host", "0.0.0.0", "--port", "8000","--workers","4"]
