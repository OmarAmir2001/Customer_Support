"""The advisor queue — the tab that shows what makes this project more than a RAG demo.

An escalation here is a **lifecycle, not an event**. The graph run ended the moment a
gate refused the answer; the ticket is a database row, and everything on this screen
is a separate request made hours or days later. That is why there is a state machine,
an audit trail, and optimistic concurrency rather than a callback waiting in memory.

Three things on this screen are deliberate and worth noticing:

* the queue hands the advisor **which gate tripped, why, and the excerpts the bot
  actually saw**, so they are not re-deriving what went wrong;
* **resolve** and **promote** are separate decisions. Resolving always delivers the
  answer to the student; promoting it into the knowledge base is a second, explicit
  act — never a side effect;
* the promotion judges give **advice**, and only one of them can block. The
  generalizability check pre-fills the checkbox; the contradiction check can *hold*
  promotion outright, because an answer contradicting the handbook means the handbook
  needs review by a person.
"""

from __future__ import annotations

import html

import gradio as gr

from api_client import ApiClient, ApiError
from settings import settings
from theme import banner

STATUSES = [
    "",
    "pending",
    "under_review",
    "resolved",
    "rejected",
    "reopened",
    "closed",
    "duplicate",
]

QUEUE_HEADERS = ["id", "status", "dept", "gate", "question", "created"]


def _queue_rows(tickets: list[dict]) -> list[list]:
    return [
        [
            ticket.get("ticket_id"),
            ticket.get("status"),
            ticket.get("department") or "—",
            ticket.get("failed_gate") or "—",
            (ticket.get("question") or "")[:110],
            str(ticket.get("created_at") or "")[:19],
        ]
        for ticket in tickets
    ]


def _chunks_html(chunks: list | None) -> str:
    """The excerpts the bot saw, with their provenance.

    Shown because an advisor judging an escalation needs to know whether retrieval
    found the wrong thing or the handbook simply does not say — those have different
    fixes, and the gate reason alone does not distinguish them.
    """
    if not chunks:
        return "<div style='opacity:0.65'>No excerpts recorded on this ticket.</div>"

    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("chunk_text") or ""
            meta = chunk.get("metadata") or {}
            score = chunk.get("score")
        else:
            text, meta, score = str(chunk), {}, None

        citation = meta.get("citation") or meta.get("source") or "unattributed"
        score_text = f" · score {score:.3f}" if isinstance(score, int | float) else ""
        blocks.append(
            "<div style='margin-bottom:0.8rem;padding:0.7rem 0.9rem;border-radius:6px;"
            "background:rgba(100,116,139,0.06);border-left:3px solid rgba(13,148,136,0.5)'>"
            f"<div style='font-size:0.75rem;opacity:0.7;margin-bottom:0.3rem'>"
            f"Excerpt {index} · {html.escape(str(citation))}{score_text}</div>"
            f"<div class='answer-panel' style='white-space:pre-wrap;font-size:0.95rem'>"
            f"{html.escape(str(text))[:1200]}</div></div>"
        )
    return "".join(blocks)


def _gate_rows(ticket: dict) -> list[list]:
    rows = [
        ["Status", ticket.get("status") or "—"],
        ["Gate that tripped", ticket.get("failed_gate") or "—"],
        ["Reason given to the advisor", ticket.get("gate_reason") or "—"],
    ]
    for gate, score in (ticket.get("gate_scores") or {}).items():
        try:
            rows.append([f"score · {gate}", f"{float(score):.3f}"])
        except (TypeError, ValueError):
            rows.append([f"score · {gate}", str(score)])
    rows += [
        ["Promoted to KB", "yes" if ticket.get("promoted_to_kb") else "no"],
        ["Promotion held", "yes" if ticket.get("promotion_held") else "no"],
    ]
    if ticket.get("promotion_hold_reason"):
        rows.append(["Hold reason", ticket["promotion_hold_reason"]])
    if ticket.get("resolved_by"):
        rows.append(["Resolved by", ticket["resolved_by"]])
    rows.append(["Thread", ticket.get("thread_id") or "—"])
    return rows


def _assessment_html(assessment: dict) -> str:
    held = assessment.get("held")
    suggested = assessment.get("suggested_promote")

    if held:
        head = banner(
            "danger",
            "Promotion HELD — this answer contradicts the handbook",
            "You cannot tick past this. The handbook goes for review instead; the "
            "answer is still delivered to the student.",
        )
    elif suggested:
        head = banner(
            "ok",
            "Judges suggest promoting this",
            "Advice only — the checkbox is yours.",
        )
    else:
        head = banner(
            "info",
            "Judges suggest NOT promoting",
            "Usually means the answer is specific to one student rather than generally "
            "true. A wrong call here only moves a checkbox.",
        )

    def line(label: str, value: bool, score, reason: str) -> str:
        try:
            score_text = f"{float(score):.2f}"
        except (TypeError, ValueError):
            score_text = str(score)
        return (
            f"<div style='margin:0.5rem 0'><strong>{html.escape(label)}:</strong> "
            f"{'yes' if value else 'no'} (score {score_text})"
            f"<div style='opacity:0.85;font-size:0.9rem;margin-top:0.2rem'>"
            f"{html.escape(str(reason or ''))}</div></div>"
        )

    body = line(
        "Generally applicable",
        bool(assessment.get("generalizable")),
        assessment.get("generalizability_score"),
        assessment.get("generalizability_reason", ""),
    ) + line(
        "Contradicts the handbook",
        bool(assessment.get("contradicts_handbook")),
        assessment.get("contradiction_score"),
        assessment.get("contradiction_reason", ""),
    )

    compared = assessment.get("compared_against") or []
    if compared:
        items = "".join(
            f"<li style='margin-bottom:0.4rem'>{html.escape(str(text))[:400]}</li>"
            for text in compared
        )
        body += (
            "<details style='margin-top:0.6rem'><summary>Handbook excerpts the "
            "contradiction verdict was based on</summary>"
            f"<ul style='margin-top:0.5rem'>{items}</ul></details>"
        )

    return head + body


def build(client: ApiClient) -> None:
    gr.Markdown(
        "### Advisor queue\n"
        "Every move here is a separate request over the same `thread_id` the student "
        "asked on. **Resolving delivers; promoting is a second decision.**"
    )

    with gr.Row():
        advisor_id = gr.Textbox(label="Acting as", value=settings.default_advisor_id, scale=2)
        status_filter = gr.Dropdown(
            choices=[("any status", "")] + [(s, s) for s in STATUSES if s],
            value="pending",
            label="Status",
            scale=2,
        )
        dept_filter = gr.Dropdown(
            choices=[("any", ""), ("CS", "CS"), ("IS", "IS")], value="", label="Department", scale=1
        )
        held_filter = gr.Checkbox(
            label="Handbook review queue only",
            value=False,
            info="Tickets whose answer the contradiction check blocked.",
            scale=2,
        )
        refresh = gr.Button("Refresh", variant="primary", scale=1)

    queue_status = gr.HTML()
    queue = gr.Dataframe(
        headers=QUEUE_HEADERS,
        column_count=(len(QUEUE_HEADERS), "fixed"),
        interactive=False,
        wrap=True,
        label="Click a row to open a ticket",
    )

    gr.Markdown("---")

    with gr.Row():
        ticket_id = gr.Number(label="Ticket id", precision=0, value=None, scale=1)
        open_button = gr.Button("Open", scale=1)

    ticket_status = gr.HTML()

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Markdown(elem_classes=["answer-panel"])
            with gr.Accordion("The excerpts the bot saw", open=False):
                chunks_box = gr.HTML()
        with gr.Column(scale=2):
            gate_box = gr.Dataframe(
                headers=["Field", "Value"],
                column_count=(2, "fixed"),
                interactive=False,
                wrap=True,
                elem_classes=["gate-table"],
            )

    gr.Markdown("#### Act on it")

    with gr.Row():
        claim_button = gr.Button("Claim")
        release_button = gr.Button("Release")
        close_button = gr.Button("Close")
        reopen_button = gr.Button("Reopen")

    with gr.Group():
        draft = gr.Textbox(
            label="Your answer to the student",
            lines=6,
            placeholder="At least 10 characters. This is delivered into the student's thread.",
            elem_classes=["answer-panel"],
        )
        with gr.Row():
            assess_button = gr.Button("Ask the promotion judges about this draft")
            promote = gr.Checkbox(
                label="Also add this to the knowledge base",
                value=False,
                info="A separate decision from resolving. The judges only advise.",
            )
            resolve_button = gr.Button("Resolve", variant="primary")
        assessment_box = gr.HTML()

    with gr.Accordion("Reject this question instead", open=False):
        reason = gr.Textbox(
            label="Reason (required)",
            lines=2,
            info="Required, unlike the other notes: the status history is the only place "
            "this decision is recorded, and a reason-less rejection is unreviewable later.",
        )
        reject_button = gr.Button("Reject", variant="stop")

    raw = gr.JSON(label="Raw ticket", visible=True)

    # --------------------------------------------------------------- handlers

    def load_queue(status_value, dept_value, held):
        try:
            tickets = client.list_tickets(
                ticket_status=status_value or None,
                department=dept_value or None,
                promotion_held=True if held else None,
            )
        except ApiError as exc:
            return banner("danger", "Could not load the queue", html.escape(str(exc))), []

        if not tickets:
            return (
                banner(
                    "info",
                    "Nothing in this queue",
                    "Escalate a question from the Ask tab to put something here.",
                ),
                [],
            )
        return (
            banner("ok", f"{len(tickets)} ticket(s)", "Click a row, or type an id below."),
            _queue_rows(tickets),
        )

    def _render(ticket: dict):
        status_line = str(ticket.get("status"))
        kind = {
            "pending": "warn",
            "under_review": "info",
            "resolved": "ok",
            "closed": "ok",
            "rejected": "danger",
        }.get(status_line, "info")
        head = banner(
            kind,
            f"Ticket #{ticket.get('ticket_id')} — {status_line}",
            f"Student {html.escape(str(ticket.get('student_id')))} · "
            f"{html.escape(str(ticket.get('department') or 'no department'))}",
        )
        question_md = f"**The student asked:**\n\n{ticket.get('question') or ''}" + (
            f"\n\n**Answer on file:**\n\n{ticket.get('advisor_answer')}"
            if ticket.get("advisor_answer")
            else ""
        )
        return (
            head,
            question_md,
            _chunks_html(ticket.get("retrieved_chunks")),
            _gate_rows(ticket),
            ticket,
            ticket.get("advisor_answer") or "",
        )

    def open_ticket(ticket_number):
        if not ticket_number:
            return (
                banner("warn", "No ticket id", "Click a row above or type an id."),
                "",
                "",
                [],
                None,
                "",
            )
        try:
            ticket = client.ticket(int(ticket_number))
        except ApiError as exc:
            failed = banner("danger", "Could not open the ticket", html.escape(str(exc)))
            return failed, "", "", [], None, ""
        return _render(ticket)

    def on_row_select(event: gr.SelectData, rows):
        """Gradio hands back the clicked cell's index; the id is column 0 of that row."""
        try:
            row_index = event.index[0]
            value = rows.iloc[row_index, 0] if hasattr(rows, "iloc") else rows[row_index][0]
            return int(value)
        except Exception:
            return gr.skip()

    def _action(fn, ticket_number, *args):
        """Run a lifecycle move, then re-render from the API's own response.

        Re-rendering from the response rather than from local state is the point: the
        API is the only thing that knows the resulting status, and a 409 here means
        either an illegal move or another advisor got there first.
        """
        if not ticket_number:
            return (
                banner("warn", "No ticket open", "Open a ticket first."),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )
        try:
            fn(int(ticket_number), *args)
            ticket = client.ticket(int(ticket_number))
        except ApiError as exc:
            return (
                banner("danger", "Refused", html.escape(str(exc))),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )
        return _render(ticket)

    detail_outputs = [ticket_status, question_box, chunks_box, gate_box, raw, draft]

    refresh.click(
        load_queue, inputs=[status_filter, dept_filter, held_filter], outputs=[queue_status, queue]
    )
    queue.select(on_row_select, inputs=[queue], outputs=[ticket_id]).then(
        open_ticket, inputs=[ticket_id], outputs=detail_outputs
    )
    open_button.click(open_ticket, inputs=[ticket_id], outputs=detail_outputs)

    claim_button.click(
        lambda tid, who: _action(client.claim, tid, who),
        inputs=[ticket_id, advisor_id],
        outputs=detail_outputs,
    )
    release_button.click(
        lambda tid, who: _action(client.release, tid, who, "released from the UI"),
        inputs=[ticket_id, advisor_id],
        outputs=detail_outputs,
    )
    close_button.click(
        lambda tid, who: _action(client.close, tid, who, "closed from the UI"),
        inputs=[ticket_id, advisor_id],
        outputs=detail_outputs,
    )
    reopen_button.click(
        lambda tid, who: _action(client.reopen, tid, who, "reopened from the UI"),
        inputs=[ticket_id, advisor_id],
        outputs=detail_outputs,
    )
    resolve_button.click(
        lambda tid, who, text, flag: _action(client.resolve, tid, who, text, bool(flag)),
        inputs=[ticket_id, advisor_id, draft, promote],
        outputs=detail_outputs,
    )
    reject_button.click(
        lambda tid, who, why: _action(client.reject, tid, who, why),
        inputs=[ticket_id, advisor_id, reason],
        outputs=detail_outputs,
    )

    def assess(ticket_number, text):
        if not ticket_number:
            return banner("warn", "No ticket open", "Open a ticket first.")
        if len((text or "").strip()) < 10:
            return banner(
                "warn", "Write a draft first", "The judges need at least 10 characters to assess."
            )
        try:
            return _assessment_html(client.assess_promotion(int(ticket_number), text.strip()))
        except ApiError as exc:
            return banner("danger", "Assessment failed", html.escape(str(exc)))

    assess_button.click(assess, inputs=[ticket_id, draft], outputs=[assessment_box])
