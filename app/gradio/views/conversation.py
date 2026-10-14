"""Reading a thread — which is how an advisor's answer reaches the student.

Delivery in this system is **passive**. Nothing is pushed anywhere when a ticket is
resolved: the advisor's message is written into the persisted LangGraph thread state
under the original `thread_id`, and the student reads it the next time they return.
This tab is that return visit.

It is the cheapest way to see the whole escalation loop close, so it is worth having
its own tab rather than being folded into Ask.
"""

from __future__ import annotations

import html

import gradio as gr

from api_client import ApiClient, ApiError
from theme import banner

ROLE_STYLES = {
    "student": ("You", "#475569"),
    "assistant": ("Assistant", "#0d9488"),
    "advisor": ("Advisor", "#7c3aed"),
    "system": ("System", "#64748b"),
}


def _transcript_html(messages: list[dict]) -> str:
    if not messages:
        return "<div style='opacity:0.65'>No turns in this thread yet.</div>"

    blocks = []
    for message in messages:
        role = str(message.get("role", "")).lower()
        label, colour = ROLE_STYLES.get(role, (role or "—", "#64748b"))
        content = html.escape(str(message.get("content", "")))
        at = message.get("created_at") or message.get("timestamp") or ""
        blocks.append(
            f"<div style='margin:0 0 0.9rem 0;padding:0.7rem 0.9rem;border-left:3px solid {colour};"
            "background:rgba(100,116,139,0.06);border-radius:6px'>"
            f"<div style='font-size:0.75rem;text-transform:uppercase;letter-spacing:0.04em;"
            f"color:{colour};margin-bottom:0.3rem'>{html.escape(label)}"
            f"{f' · {html.escape(str(at)[:19])}' if at else ''}</div>"
            f"<div class='answer-panel' style='white-space:pre-wrap'>{content}</div>"
            "</div>"
        )
    return "".join(blocks)


def build(client: ApiClient) -> None:
    gr.Markdown(
        "### Read a conversation\n"
        "Delivery is **passive** — resolving a ticket writes the advisor's answer into "
        "this thread's persisted state. Nothing is emailed or pushed; the student finds "
        "it here on their next visit. Paste a thread id from the Ask tab."
    )

    with gr.Row():
        thread_id = gr.Textbox(label="Thread id", placeholder="uuid from the Ask tab", scale=4)
        load = gr.Button("Load", variant="primary", scale=1)

    status = gr.HTML()
    transcript = gr.HTML()

    with gr.Accordion("Latest-turn fields", open=False):
        fields = gr.Dataframe(
            headers=["Field", "Value"],
            column_count=(2, "fixed"),
            interactive=False,
            wrap=True,
        )
        raw = gr.JSON(label="Raw response")

    def run(thread):
        thread = (thread or "").strip()
        if not thread:
            return (
                banner(
                    "warn",
                    "Paste a thread id",
                    "Ask a question first; the id is in the Ask tab.",
                ),
                "",
                [],
                None,
            )
        try:
            payload = client.conversation(thread)
        except ApiError as exc:
            failed = banner("danger", "Could not load the thread", html.escape(str(exc)))
            return failed, "", [], None

        resolved_by = payload.get("resolved_by")
        if resolved_by:
            head = banner(
                "ok",
                f"An advisor has answered — resolved by {html.escape(str(resolved_by))}",
                "`resolved_by` being set is what tells the student the answer they are "
                "reading is a human's and not the holding placeholder.",
            )
        elif payload.get("escalated"):
            head = banner(
                "warn",
                f"Escalated, still open — ticket #{payload.get('ticket_id')}",
                "The holding message is the last assistant turn. Resolve the ticket in "
                "the Advisor queue tab and reload this thread.",
            )
        else:
            head = banner(
                "info", "Answered by the assistant", "No human was involved in this thread."
            )

        rows = [
            ["Thread", payload.get("thread_id") or "—"],
            ["Escalated", "yes" if payload.get("escalated") else "no"],
            ["Ticket", f"#{payload['ticket_id']}" if payload.get("ticket_id") else "—"],
            ["Resolved by", resolved_by or "—"],
            ["Turns", str(len(payload.get("messages") or []))],
        ]
        return head, _transcript_html(payload.get("messages") or []), rows, payload

    load.click(run, inputs=[thread_id], outputs=[status, transcript, fields, raw])
    thread_id.submit(run, inputs=[thread_id], outputs=[status, transcript, fields, raw])
