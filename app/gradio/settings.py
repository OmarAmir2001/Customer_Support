"""Configuration for the UI, from the environment.

Deliberately NOT `customer_support.helpers.config.Settings`. This app is a separate
deployable that talks to the API over HTTP — importing the backend's settings would
drag in Postgres credentials, provider keys and the whole dependency tree into a
process that needs none of them, and would stop the UI being runnable against a
remote deployment.

The only thing it needs to know is where the API is.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    #: The gated FastAPI app. Everything except streaming goes here.
    api_base_url: str = _env("UI_API_BASE_URL", "http://127.0.0.1:8000")

    #: The BentoML service, which is a SEPARATE process on a separate port and is
    #: optional — the UI degrades to "streaming unavailable" rather than breaking.
    stream_base_url: str = _env("UI_STREAM_BASE_URL", "http://127.0.0.1:3001")

    #: Where this UI listens. 127.0.0.1 by default, matching every other service in
    #: this project: nginx is the only thing that should ever bind 0.0.0.0.
    host: str = _env("UI_HOST", "127.0.0.1")
    port: int = int(_env("UI_PORT", "7860"))

    #: Generous, because a cold question runs retrieval plus up to three judge calls
    #: against a rate-limited provider. A 10s default would time out a working system.
    request_timeout: float = float(_env("UI_REQUEST_TIMEOUT", "180"))

    #: Dashboards the System tab links to. Read from the environment so a deployed
    #: instance can point at real hosts instead of localhost.
    grafana_url: str = _env("UI_GRAFANA_URL", "http://127.0.0.1:3000")
    prometheus_url: str = _env("UI_PROMETHEUS_URL", "http://127.0.0.1:9090")
    langfuse_url: str = _env("UI_LANGFUSE_URL", "http://127.0.0.1:3002")
    mlflow_url: str = _env("UI_MLFLOW_URL", "http://127.0.0.1:5000")

    #: Shown as the default advisor in the queue tab. A real deployment would take
    #: this from an authenticated session; there is no auth here, and the README
    #: says so rather than implying otherwise.
    default_advisor_id: str = _env("UI_ADVISOR_ID", "advisor_demo")
    default_student_id: str = _env("UI_STUDENT_ID", "student_demo")


settings = Settings()
