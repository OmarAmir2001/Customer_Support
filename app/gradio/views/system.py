"""Health, memory, and the design decisions worth pointing at.

This tab exists because the most interesting properties of this system are invisible
from the chat window. An empty vector collection makes every question escalate while
`/chat` keeps returning 200 with normal latency — so `documents_indexed` belongs on
screen, not buried in a readiness probe nobody calls.
"""

from __future__ import annotations

import html

import gradio as gr

from api_client import ApiClient, ApiError
from settings import settings
from theme import banner


def build(client: ApiClient) -> None:
    gr.Markdown("### System")

    with gr.Row():
        check = gr.Button("Check health", variant="primary", scale=1)
        gr.Markdown(
            f"<small>API: `{settings.api_base_url}` · "
            f"streaming: `{settings.stream_base_url}`</small>"
        )

    health_status = gr.HTML()
    health_table = gr.Dataframe(
        headers=["Field", "Value"], column_count=(2, "fixed"), interactive=False, wrap=True
    )

    def run_health():
        try:
            health = client.health()
        except ApiError as exc:
            return banner("danger", "API unreachable", html.escape(str(exc))), []

        indexed = health.get("documents_indexed")
        if health.get("status") == "healthy" and indexed:
            head = banner(
                "ok",
                f"Ready — {indexed} chunks indexed",
                "Readiness means the collection can be counted and is not empty.",
            )
        elif indexed == 0:
            head = banner(
                "danger",
                "Up, but NOTHING is indexed",
                "Every question will escalate, and /chat will keep returning 200 with "
                "normal latency the whole time. Run scripts/index_corpus.py.",
            )
        else:
            head = banner(
                "warn",
                f"Degraded — status {health.get('status')}",
                "The collection could not be counted. The app reports degraded rather "
                "than claiming healthy with a null count.",
            )

        rows = [[key, str(value)] for key, value in health.items()]
        return head, rows

    check.click(run_health, outputs=[health_status, health_table])

    gr.Markdown("---\n#### Long-term memory")
    gr.Markdown(
        "<small>One patched profile per student — identity, not history. Extraction is "
        "frequency-gated (most turns cost no model call), runs on a smaller model, and "
        "happens after the response is sent. A patch can never delete a known fact, and "
        "any field carrying PII is dropped rather than stored.</small>"
    )

    with gr.Row():
        student_id = gr.Textbox(label="Student id", value=settings.default_student_id, scale=3)
        load_profile = gr.Button("Load profile", scale=1)
        wipe_profile = gr.Button("Delete profile", variant="stop", scale=1)

    profile_status = gr.HTML()
    profile_json = gr.JSON(label="Profile")

    def run_profile(student):
        student = (student or "").strip()
        if not student:
            return banner("warn", "Type a student id", ""), None
        try:
            return banner("ok", f"Profile for {html.escape(student)}", ""), client.profile(student)
        except ApiError as exc:
            return banner("danger", "Could not load the profile", html.escape(str(exc))), None

    def run_wipe(student):
        student = (student or "").strip()
        if not student:
            return banner("warn", "Type a student id", ""), None
        try:
            result = client.delete_profile(student)
        except ApiError as exc:
            return banner("danger", "Could not delete the profile", html.escape(str(exc))), None
        return (
            banner(
                "ok",
                "Profile deleted",
                "This is the privacy/reset path. The conversation transcript is separate "
                "and lives in the checkpointer.",
            ),
            result,
        )

    load_profile.click(run_profile, inputs=[student_id], outputs=[profile_status, profile_json])
    wipe_profile.click(run_wipe, inputs=[student_id], outputs=[profile_status, profile_json])

    # Built from data rather than one long f-string: the row text is long, and a
    # markdown table cannot be wrapped without breaking the table.
    dashboards = [
        (
            "Grafana",
            settings.grafana_url,
            "escalation rate, judge failures, query drift, token cost",
        ),
        ("Prometheus", settings.prometheus_url, "raw metrics and the seven alert rules"),
        ("Langfuse", settings.langfuse_url, "one trace per question, searchable by `thread_id`"),
        ("MLflow", settings.mlflow_url, "the chunking sweep and the registered retrieval config"),
        ("API docs", f"{settings.api_base_url}/docs", "the generated OpenAPI schema"),
    ]
    rows = "\n".join(f"| {name} | [{url}]({url}) | {what} |" for name, url, what in dashboards)
    gr.Markdown(
        "---\n#### Dashboards\n\n"
        "| | | |\n|---|---|---|\n"
        f"{rows}\n\n"
        "All of these bind to loopback by design; nginx is the only service "
        "on `0.0.0.0`."
    )

    gr.Markdown(
        """---
#### What this UI is deliberately not

**There is no authentication.** "Acting as" in the advisor tab is a text box, not a
session. A real deployment puts an identity provider in front and takes `advisor_id`
from the token — which is why the API takes it as a parameter rather than inferring
it, and why the audit trail records whoever the caller claimed to be.

**It holds no business logic.** Every decision on screen — which gate tripped, whether
promotion is held, whether a transition is legal — is made by the API and rendered
here. The only thing this process knows how to do is draw and make HTTP calls. That is
why it can point at a remote deployment by changing one environment variable, and why
breaking it cannot break the assistant.

**It is not the product UI.** This is a showcase and an operator console. The
student-facing interface is a separate piece of design work.
"""
    )
