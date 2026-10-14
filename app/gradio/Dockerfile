# The operator console, as its own image.
#
# Built from the REPO ROOT as context, like the API image:
#   docker build -f app/gradio/Dockerfile -t handbook-ui .
#
# It installs ONLY the `ui` extra — not the API's dependencies. This process needs
# gradio and an HTTP client and nothing else: no asyncpg, no provider SDKs, no
# langgraph. That is the whole point of the UI importing nothing from
# customer_support, and it is visible here as a much smaller image.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Only what pip needs to resolve the extra. Copying the whole repo first would
# invalidate this layer on every source change.
COPY pyproject.toml README.md ./
COPY src/customer_support/__init__.py src/customer_support/__init__.py

RUN pip install --no-cache-dir ".[ui]"

COPY app/gradio ./app/gradio

# 0.0.0.0 INSIDE the container only. Publish it to 127.0.0.1 on the host, the way
# every other service in docker-compose.yml is published — nginx is the only thing
# that should ever be reachable from outside.
ENV UI_HOST=0.0.0.0 \
    UI_PORT=7860

EXPOSE 7860

# A UI whose API is down must still start and say so, so there is no healthcheck
# dependency on the API here.
CMD ["python", "app/gradio/main.py"]
