"""Entry point for the Gradio showcase UI.

    uv run --extra ui python app/gradio/main.py

**Why this directory has no `__init__.py`, and why that is deliberate.** The folder is
called `gradio`. If it were an importable package and its parent were on `sys.path`,
`import gradio` inside these modules would resolve to *this directory* instead of the
library, and the failure would be a baffling `AttributeError` on `gradio.Blocks`. So
the modules here are flat and the directory itself is put on `sys.path` — never its
parent. Running the file directly does that automatically; the guard below makes it
true however the file is invoked.

**Why this is a separate process from the API.** It imports nothing from
`customer_support`. It holds no business logic, no database handle and no provider
key — every decision it renders was made by the API. That means it can point at a
remote deployment by changing one environment variable, it can be deployed and scaled
on its own, and breaking it cannot break the assistant.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Must happen before the local imports below, and before `gradio` is imported, so the
# flat-module layout resolves whether this is run as a file, from another directory,
# or by a process manager.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import gradio as gr  # noqa: E402

from api_client import ApiClient  # noqa: E402
from settings import settings  # noqa: E402
from theme import CSS, THEME  # noqa: E402
from views import advisor, ask, conversation, knowledge, system  # noqa: E402

HEADER = """
# 🎧 Handbook Assistant

Corrective RAG over the CS and IS department handbooks, with **confidence-gated
escalation to a human advisor** and a human-gated learning loop. Bilingual
Arabic/English.

Three orthogonal judges decide whether an answer reaches the student. If any of them
refuses, the run ends, a ticket is written, and the question goes to a person — the
drafted answer is **discarded**, never shown as a best guess.
"""


def build_app() -> gr.Blocks:
    """Build the interface without starting a server.

    Separate from `main()` so it can be built in a test: a smoke test that constructs
    every tab catches a bad event wiring or a renamed component, which is most of what
    breaks in a UI, without needing a browser or a running API.
    """
    client = ApiClient()

    # Gradio 6 moved `theme` and `css` from the Blocks constructor to launch(), so
    # they are applied in main() rather than here. Keeping build_app() free of them
    # is also what lets a test construct the whole interface without a server.
    with gr.Blocks(title="Handbook Assistant", fill_height=True) as demo:
        gr.Markdown(HEADER)

        with gr.Tabs():
            with gr.Tab("Ask"):
                ask.build(client)
            with gr.Tab("Conversation"):
                conversation.build(client)
            with gr.Tab("Advisor queue"):
                advisor.build(client)
            with gr.Tab("Knowledge base"):
                knowledge.build(client)
            with gr.Tab("System"):
                system.build(client)

        gr.Markdown(
            "<small>This console makes HTTP calls to the API and holds no logic of its "
            "own. There is no authentication — see the System tab.</small>"
        )

    return demo


def main() -> None:
    demo = build_app()
    # queue() is what makes the streaming tab work: without it Gradio collects a
    # generator's output and sends it once, which would silently turn the streaming
    # path back into the blocking one.
    demo.queue(default_concurrency_limit=8).launch(
        server_name=settings.host,
        server_port=settings.port,
        theme=THEME,
        css=CSS,
        # Nothing here should reach the internet. An operator console with a public
        # tunnel is a different risk conversation entirely.
        share=False,
        # Show the real error text rather than a generic "Error" toast. This is an
        # operator console: the person reading it is the person who can fix the
        # thing that broke, and api_client.py has already phrased every failure for
        # them. On a public, user-facing app this would be the wrong default.
        show_error=True,
        mcp_server=False,
    )


if __name__ == "__main__":
    main()
