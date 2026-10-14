"""Loading the knowledge base — upload, chunk, embed, index, and debug retrieval.

**There are two ingestion paths in this project, and this tab drives the ad-hoc one.**
That distinction matters enough to put on screen:

* *Offline* — `scripts/build_corpus.py` then `scripts/index_corpus.py`, driven by DVC.
  Deterministic, content-hashed, eight validation rules that refuse to write a bad
  corpus. This is what the handbooks and every chunking experiment use, because a
  metric is meaningless without a data version.
* *HTTP* — the three buttons below. For an ad-hoc document someone hands you. It
  writes rows to the `chunks` table and passes their ids to the vector store.

Use the offline path for anything you will measure. Use this for anything you will not.

Two destructive things live here, so both are gated rather than being a plain
checkbox: a push with reset rebuilds the whole collection — **including deleting
every promoted ticket answer** — and every push spends embedding calls.
"""

from __future__ import annotations

import html

import gradio as gr

from api_client import ApiClient, ApiError
from theme import banner

#: Must match the number in KB_COLLECTION_NAME (`collection_3`), or you will index
#: into a collection the agent never reads. The commonest way to spend money and
#: change nothing, so it is a visible field with a warning rather than a constant.
DEFAULT_PROJECT_ID = 3


def _results_html(results: list[dict]) -> str:
    if not results:
        return "<div style='opacity:0.65'>No matches.</div>"

    blocks = []
    for index, doc in enumerate(results, start=1):
        meta = doc.get("metadata") or {}
        citation = meta.get("citation") or meta.get("source") or "unattributed"
        section = meta.get("section") or ""
        dept = meta.get("department") or "—"
        score = doc.get("score")
        score_text = f"{score:.4f}" if isinstance(score, int | float) else str(score)
        promoted = meta.get("source") == "instructor_resolved"
        border = "#7c3aed" if promoted else "rgba(13,148,136,0.5)"
        tag = (
            "<span style='font-size:0.7rem;padding:0.1rem 0.4rem;border-radius:4px;"
            "background:rgba(124,58,237,0.15);color:#7c3aed;margin-left:0.4rem'>"
            "promoted ticket answer</span>"
            if promoted
            else ""
        )
        blocks.append(
            f"<div style='margin-bottom:0.8rem;padding:0.7rem 0.9rem;border-radius:6px;"
            f"background:rgba(100,116,139,0.06);border-left:3px solid {border}'>"
            f"<div style='font-size:0.75rem;opacity:0.75;margin-bottom:0.3rem'>"
            f"#{index} · score {score_text} · {html.escape(str(citation))} · "
            f"dept {html.escape(str(dept))}{tag}</div>"
            + (
                f"<div style='font-size:0.7rem;opacity:0.55;margin-bottom:0.3rem'>"
                f"{html.escape(str(section))}</div>"
                if section
                else ""
            )
            + "<div class='answer-panel' style='white-space:pre-wrap;font-size:0.93rem'>"
            f"{html.escape(str(doc.get('text') or ''))[:1000]}</div></div>"
        )
    return "".join(blocks)


def build(client: ApiClient) -> None:
    gr.Markdown(
        "### Knowledge base\n"
        "Upload → chunk → embed and index. **For anything you will measure, use the "
        "offline path instead** (`dvc repro` then `scripts/index_corpus.py`): it is "
        "deterministic, content-hashed and validated, and the chunking experiments "
        "depend on it. This tab is for an ad-hoc document."
    )

    with gr.Row():
        project_id = gr.Number(
            label="Project id",
            value=DEFAULT_PROJECT_ID,
            precision=0,
            info="Becomes collection_<id>. Must match KB_COLLECTION_NAME or you will "
            "index into a collection the agent never reads.",
            scale=1,
        )
        info_button = gr.Button("Collection status", variant="secondary", scale=1)

    info_status = gr.HTML()
    info_table = gr.Dataframe(
        headers=["Field", "Value"], column_count=(2, "fixed"), interactive=False, wrap=True
    )

    def show_info(pid):
        try:
            payload = client.index_info(int(pid))
        except ApiError as exc:
            return banner("danger", "Could not read the collection", html.escape(str(exc))), []

        collection = payload.get("collection_info") or payload
        count = None
        if isinstance(collection, dict):
            count = collection.get("record_count")

        if count:
            head = banner(
                "ok",
                f"collection_{int(pid)} holds {count} vectors",
                "This is what the agent searches.",
            )
        elif count == 0:
            head = banner(
                "danger",
                f"collection_{int(pid)} is EMPTY",
                "Every question will escalate, and /chat will keep returning 200 with "
                "normal latency the whole time.",
            )
        else:
            head = banner("warn", "No record count reported", "See the raw fields below.")

        rows = (
            [[k, str(v)] for k, v in collection.items()]
            if isinstance(collection, dict)
            else [["response", str(collection)]]
        )
        return head, rows

    info_button.click(show_info, inputs=[project_id], outputs=[info_status, info_table])

    gr.Markdown("---\n#### 1 · Upload")

    with gr.Row():
        upload = gr.File(
            label="Handbook file",
            file_types=[".md", ".pdf", ".txt", ".json"],
            type="filepath",
            scale=3,
        )
        ingest_button = gr.Button("Upload", variant="primary", scale=1)

    ingest_status = gr.HTML()
    file_id = gr.Textbox(
        label="file_id",
        info="Returned by the upload and consumed by step 2. Leave blank there to "
        "process every file this project has.",
        interactive=True,
    )

    def do_ingest(pid, path):
        if not path:
            return banner("warn", "Pick a file first", ""), gr.skip()
        try:
            payload = client.ingest(int(pid), path)
        except ApiError as exc:
            return banner("danger", "Upload failed", html.escape(str(exc))), gr.skip()

        return (
            banner(
                "ok",
                f"Uploaded as {html.escape(str(payload.get('asset_name')))}",
                f"file_id <code>{html.escape(str(payload.get('file_id')))}</code> — "
                "carried into step 2 below. Nothing has been chunked or embedded yet.",
            ),
            str(payload.get("file_id") or ""),
        )

    ingest_button.click(do_ingest, inputs=[project_id, upload], outputs=[ingest_status, file_id])

    gr.Markdown(
        "---\n#### 2 · Chunk into Postgres\n"
        "<small>Free — no model calls. Markdown splits on its own headings, so a "
        "chunk's `section` is a real citable path rather than a character offset.</small>"
    )

    with gr.Row():
        chunk_size = gr.Slider(200, 4000, value=1000, step=100, label="Chunk size")
        overlap = gr.Slider(0, 400, value=20, step=10, label="Overlap")
        process_reset = gr.Checkbox(
            label="Delete this project's existing chunk rows first",
            value=False,
            info="Postgres rows only — it does not touch the vector collection.",
        )
    process_button = gr.Button("Chunk", variant="primary")
    process_status = gr.HTML()

    def do_process(pid, fid, size, over, reset):
        try:
            payload = client.process(int(pid), fid or None, size, over, bool(reset))
        except ApiError as exc:
            return banner("danger", "Chunking failed", html.escape(str(exc)))
        return banner(
            "ok",
            f"{payload.get('inserted_chunks')} chunks from "
            f"{payload.get('processed_files')} file(s)",
            "In Postgres only. Nothing is searchable until step 3 embeds it.",
        )

    process_button.click(
        do_process,
        inputs=[project_id, file_id, chunk_size, overlap, process_reset],
        outputs=[process_status],
    )

    gr.Markdown("---\n#### 3 · Embed and index")
    gr.HTML(
        banner(
            "warn",
            "This step spends money",
            "One embedding call per batch of chunks, on your Cohere key. The sync is "
            "idempotent per (source, section) — pushing twice replaces rather than "
            "duplicates, so a re-push after a re-chunk is safe and does not need a reset.",
        )
    )

    with gr.Row():
        push_reset = gr.Checkbox(
            label="Rebuild the whole collection (destructive)",
            value=False,
            info="Deletes the collection and every promoted ticket answer in it. The "
            "per-section sync is already idempotent, so you almost never want this.",
        )
        confirm_reset = gr.Textbox(
            label="To rebuild, type: RESET",
            placeholder="required only when the box above is ticked",
        )
    push_button = gr.Button("Embed and index", variant="primary")
    push_status = gr.HTML()

    def do_push(pid, reset, confirmation):
        # Two gates on one destructive action: the checkbox states the intent, the
        # typed word proves it was not a misclick. A reset silently deleting the
        # learning loop's accumulated answers is not recoverable from this screen.
        if reset and (confirmation or "").strip() != "RESET":
            return banner(
                "danger",
                "Rebuild not confirmed",
                "Type RESET in the box to confirm, or untick the box. A rebuild "
                "deletes every promoted ticket answer and cannot be undone here.",
            )
        try:
            payload = client.push(int(pid), bool(reset))
        except ApiError as exc:
            return banner("danger", "Indexing failed", html.escape(str(exc)))
        return banner(
            "ok",
            f"{payload.get('inserted_item_count')} vectors indexed",
            "Check the collection status above, then try the search below — it uses "
            "the same retrieval the agent uses.",
        )

    push_button.click(
        do_push, inputs=[project_id, push_reset, confirm_reset], outputs=[push_status]
    )

    gr.Markdown(
        "---\n#### 4 · Debug retrieval\n"
        "<small>The same `RetrievalController` the graph uses, so this shows exactly "
        "what the agent would see — department filter and handbook precedence "
        "included. Promoted ticket answers are tagged, and they always rank below "
        "handbook text when both are present.</small>"
    )

    with gr.Row():
        query = gr.Textbox(label="Query", placeholder="كم عدد الساعات المعتمدة؟", scale=3)
        search_dept = gr.Dropdown(
            choices=[("no filter", ""), ("CS", "CS"), ("IS", "IS")],
            value="",
            label="Department",
            scale=1,
        )
        limit = gr.Slider(1, 20, value=5, step=1, label="Top k", scale=1)
    search_button = gr.Button("Search", variant="primary")
    search_status = gr.HTML()
    search_results = gr.HTML()

    def do_search(pid, text, dept, k):
        text = (text or "").strip()
        if not text:
            return banner("warn", "Type a query", ""), ""
        try:
            payload = client.kb_search(int(pid), text, int(k), dept or None)
        except ApiError as exc:
            # The endpoint answers 400 for "no matches", not 200 with an empty list,
            # so an empty collection and a bad query look alike from here.
            return (
                banner(
                    "warn",
                    "No results",
                    html.escape(str(exc))
                    + " — note this endpoint reports no-matches as a 400, so check the "
                    "collection status above if you expected hits.",
                ),
                "",
            )
        results = payload.get("results") or []
        return (
            banner("ok", f"{len(results)} match(es)", "Ranked as the agent would see them."),
            _results_html(results),
        )

    search_button.click(
        do_search,
        inputs=[project_id, query, search_dept, limit],
        outputs=[search_status, search_results],
    )
    query.submit(
        do_search,
        inputs=[project_id, query, search_dept, limit],
        outputs=[search_status, search_results],
    )
