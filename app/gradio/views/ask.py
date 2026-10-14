"""The student's view — and the one screen that shows this project's central trade-off.

Two modes sit side by side on purpose:

* **Verified** hits `POST /api/v1/chat`. Three judge gates, escalation to a human,
  a ticket, citations. The answer arrives all at once and it has been checked.
* **Streaming** hits the BentoML `/ask`. Tokens appear as the model produces them,
  and the post-generation gates **cannot** run — faithfulness and answer relevance
  need a complete answer, and by the time they could object the student has read it.

Putting them one radio button apart is the clearest way to show that streaming buys
responsiveness with verification, which is the decision a reviewer should see rather
than read about.
"""

from __future__ import annotations

import html

import gradio as gr

from api_client import ApiClient, ApiError
from settings import settings
from theme import banner

GATE_LABELS = {
    "context_relevance": "Gate 1 — context relevance",
    "faithfulness": "Gate 2a — faithfulness",
    "answer_relevance": "Gate 2b — answer relevance",
    "generation": "Generation",
}


def _citations_html(sources: list[str] | None) -> str:
    if not sources:
        return (
            "<div style='opacity:0.65;font-size:0.9rem'>No citations. Either the "
            "question escalated — in which case the answer field holds a holding "
            "message, not an answer — or no retrieved chunk could name its "
            "origin.</div>"
        )
    chips = "".join(f"<span class='citation'>{html.escape(s)}</span>" for s in sources)
    return (
        "<div style='font-size:0.85rem;opacity:0.7;margin-bottom:0.35rem'>"
        "Sections this answer was built from — openable and checkable, not chunk ids:"
        f"</div>{chips}"
    )


def _gate_rows(payload: dict) -> list[list]:
    """Gate scores are not on ChatResponse, so show what the response does carry.

    The full per-gate scores live on the ticket (the advisor sees them), which is the
    right place: a student has no use for "faithfulness 0.62", and exposing it would
    invite arguing with the number rather than reading the answer.
    """
    rows = []
    if payload.get("escalated"):
        rows.append(["Outcome", "escalated to a human advisor"])
        failed = payload.get("failed_gate")
        if failed:
            rows.append(["Gate that tripped", GATE_LABELS.get(failed, failed)])
        if payload.get("escalation_reason"):
            rows.append(["Advisor-facing reason", payload["escalation_reason"]])
        if payload.get("ticket_id"):
            rows.append(["Ticket", f"#{payload['ticket_id']} — see the Advisor queue tab"])
    else:
        rows.append(["Outcome", "answered — every gate passed"])
        rows.append(["Citations", str(len(payload.get("sources") or []))])
    rows.append(["Thread", payload.get("thread_id") or "—"])
    return rows


def build(client: ApiClient) -> None:
    gr.Markdown(
        "### Ask the handbook assistant\n"
        "The **verified** path is what a student actually talks to. The **streaming** "
        "path is faster to first token and cannot be gated — it is here so the trade "
        "is visible rather than described."
    )

    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="Question",
                placeholder="كم عدد الساعات المعتمدة المطلوبة للتخرج؟   /   "
                "How many credit hours do I need to graduate?",
                lines=3,
                elem_classes=["answer-panel"],
            )
        with gr.Column(scale=1):
            mode = gr.Radio(
                choices=[
                    ("Verified — gated, cited, can escalate", "verified"),
                    ("Streaming — fast, ungated", "streaming"),
                ],
                value="verified",
                label="Path",
            )

    with gr.Row():
        student_id = gr.Textbox(label="Student id", value=settings.default_student_id, scale=2)
        department = gr.Dropdown(
            choices=[("Let memory decide", ""), ("CS", "CS"), ("IS", "IS")],
            value="",
            label="Department",
            info="The stored profile wins over this — a student cannot read the other "
            "handbook by claiming to be in it.",
            scale=2,
        )
        language = gr.Dropdown(
            choices=[("Detect from the question", ""), ("English", "en"), ("العربية", "ar")],
            value="",
            label="Answer language",
            scale=2,
        )

    with gr.Row():
        ask_button = gr.Button("Ask", variant="primary", scale=3)
        continue_thread = gr.Checkbox(
            label="Continue the same conversation",
            value=False,
            info="Reuses the thread id below, so the transcript accumulates.",
            scale=2,
        )

    status = gr.HTML()
    answer = gr.Markdown(elem_classes=["answer-panel"])
    sources = gr.HTML()

    with gr.Accordion("What happened", open=False):
        detail = gr.Dataframe(
            headers=["", ""],
            column_count=(2, "fixed"),
            interactive=False,
            elem_classes=["gate-table"],
            wrap=True,
        )
        raw = gr.JSON(label="Raw response")

    thread_state = gr.State("")

    # --------------------------------------------------------------- handlers

    def run(question_text, mode_value, student, dept, lang, keep_thread, thread_id):
        """One generator for both paths, so the outputs stay in one shape.

        Gradio needs every yield to match the outputs list. Returning early from a
        generator on an error would silently leave components stale, so each branch
        yields a complete tuple.
        """
        blank = (gr.skip(), "", "", [], None, thread_id)

        question_text = (question_text or "").strip()
        if len(question_text) < 3:
            yield (
                banner(
                    "warn",
                    "Type a question first",
                    "Three characters minimum — the API rejects shorter with a 422.",
                ),
                *blank[1:],
            )
            return

        if mode_value == "streaming":
            yield (
                banner(
                    "warn",
                    "Streaming — this answer is NOT verified",
                    "The faithfulness and answer-relevance judges need a complete "
                    "answer, so they cannot run here. Gate 1 (context relevance) did "
                    "run, so an unanswerable question is still refused before any "
                    "token is produced. PII redaction also runs.",
                ),
                "",
                "",
                [["Path", "BentoML /ask — streaming, ungated"]],
                None,
                thread_id,
            )
            accumulated = ""
            try:
                for piece in client.stream(question_text, dept or None, lang or None):
                    accumulated += piece
                    yield (
                        gr.skip(),
                        accumulated,
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                    )
            except ApiError as exc:
                yield (
                    banner("danger", "Streaming unavailable", html.escape(str(exc))),
                    accumulated,
                    "",
                    [["Path", "BentoML /ask — not reachable"]],
                    None,
                    thread_id,
                )
                return

            yield (
                banner(
                    "warn",
                    "Streamed, ungated",
                    "Switch to the verified path to get citations, gate verdicts and escalation.",
                ),
                accumulated,
                _citations_html(None),
                [
                    ["Path", "BentoML /ask — streaming, ungated"],
                    ["Characters streamed", str(len(accumulated))],
                ],
                None,
                thread_id,
            )
            return

        # ---- verified path
        yield (
            banner("info", "Working…", "Retrieval, then up to three judge calls."),
            "",
            "",
            [],
            None,
            thread_id,
        )
        try:
            payload = client.ask(
                question=question_text,
                student_id=(student or settings.default_student_id).strip(),
                department=dept or None,
                language=lang or None,
                thread_id=thread_id if keep_thread and thread_id else None,
            )
        except ApiError as exc:
            yield (
                banner("danger", "The request failed", html.escape(str(exc))),
                "",
                "",
                [],
                None,
                thread_id,
            )
            return

        escalated = bool(payload.get("escalated"))
        if escalated:
            head = banner(
                "warn",
                f"Escalated to a human advisor — ticket #{payload.get('ticket_id')}",
                "A gate refused the drafted answer, so it was withheld rather than "
                "shown. The text below is the holding message the student sees. The "
                "advisor's reply will appear in this same thread.",
            )
        else:
            head = banner(
                "ok",
                "Answered — every gate passed",
                f"{len(payload.get('sources') or [])} citation(s). The answer was "
                "built only from the retrieved excerpts.",
            )

        yield (
            head,
            payload.get("answer") or "",
            _citations_html(payload.get("sources")),
            _gate_rows(payload),
            payload,
            payload.get("thread_id") or thread_id,
        )

    outputs = [status, answer, sources, detail, raw, thread_state]
    inputs = [question, mode, student_id, department, language, continue_thread, thread_state]

    ask_button.click(run, inputs=inputs, outputs=outputs)
    question.submit(run, inputs=inputs, outputs=outputs)

    gr.Markdown(
        "<small>The thread id is kept in session state. Tick *continue* to add turns "
        "to the same conversation, or paste it into the **Conversation** tab to read "
        "the transcript — including an advisor's reply once a ticket is resolved."
        "</small>"
    )
