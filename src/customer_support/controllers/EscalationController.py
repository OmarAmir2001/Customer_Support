"""Section 3 (lifecycle) + Section 1 (source-of-truth sync).

The ONLY place a ticket changes state, and the only place resolved answers are
pushed into pgvector. Both the graph's escalate node and the advisor's HTTP endpoint
call these same methods, so the logic exists once.
"""

import asyncio
import json
from datetime import UTC, datetime

from customer_support.helpers.logging_config import get_logger, set_correlation_id
from customer_support.helpers.pii import redact
from customer_support.models.db_schemas import RetrievedDocument
from customer_support.models.db_schemas.customer_support.schemes.ticket import Ticket
from customer_support.models.enums.TicketStatusEnum import (
    TicketActor,
    TicketStatus,
    assert_transition_allowed,
)
from customer_support.models.graph.conversation import ConversationMessage
from customer_support.models.graph.graph_state import GraphState
from customer_support.models.TicketModel import TicketModel
from customer_support.stores.llm.LLMEnum import DocumentTypeEnum

from .BaseController import BaseController

INSTRUCTOR_RESOLVED_SOURCE = "instructor_resolved"

# The node the advisor's answer is attributed to when it is written into the thread.
# escalate_node is where the original run ended, and its only edge goes to END, so
# attributing the write there leaves the thread finished rather than mid-graph.
RESOLUTION_AS_NODE = "escalate_node"


class EscalationController(BaseController):
    def __init__(
        self,
        ticket_model: TicketModel,
        vectordb_client,
        embedding_client,
        collection_name: str,
        settings=None,
        thread_writer=None,
        promotion=None,
    ):
        super().__init__(settings)
        self.ticket_model = ticket_model
        # Section 5's quality gate. Optional so the lifecycle can still be exercised
        # without it; when absent, promotion falls back to trusting the advisor's
        # checkbox exactly as it did before the gate existed.
        self.promotion = promotion
        self.vectordb_client = vectordb_client
        self.embedding_client = embedding_client
        self.collection_name = collection_name
        # Anything with ``aupdate_state(config, values, as_node=...)`` — in practice the
        # compiled graph. Injected rather than imported: a controller must not depend on
        # the graph layer (Section 4's one-directional rule). It is assigned after
        # construction because the graph needs this controller to be built first.
        self.thread_writer = thread_writer
        self.logger = get_logger(__name__)

    # ------------------------------------------------------------- lifecycle

    async def create_ticket(self, state: GraphState) -> Ticket:
        """Called by the escalate node. The graph run ends right after this — nothing
        stays parked in memory, an open ticket costs a row."""

        for field in ("question", "student_id", "thread_id", "failed_gate", "gate_reason"):
            if not state.get(field):
                raise ValueError(f"cannot create a ticket without '{field}'")

        ticket = Ticket(
            thread_id=state["thread_id"],
            student_id=state["student_id"],
            department=state.get("department"),
            question=state["question"],
            failed_gate=state["failed_gate"],
            gate_reason=state["gate_reason"],
            gate_scores=state.get("gate_scores") or {},
            retrieved_chunks=self._serialise_chunks(state.get("retrieved_chunks") or []),
            status=TicketStatus.PENDING.value,
        )

        return await self.ticket_model.create_ticket(ticket, actor=TicketActor.SYSTEM.value)

    async def claim(self, ticket_id: int, advisor_id: str) -> Ticket:
        """pending -> under_review. Separate from resolve so the dashboard can show
        who is already working on a ticket."""

        ticket = await self._require_ticket(ticket_id)
        current = TicketStatus(ticket.status)
        assert_transition_allowed(ticket_id, current, TicketStatus.UNDER_REVIEW)

        return await self.ticket_model.apply_transition(
            ticket_id=ticket_id,
            from_status=current,
            to_status=TicketStatus.UNDER_REVIEW,
            actor=advisor_id,
        )

    async def resolve(
        self, ticket_id: int, advisor_answer: str, advisor_id: str, promote_to_kb: bool
    ) -> Ticket:
        """Deliver the advisor's answer, and (separately) decide whether it enters
        the knowledge base.

        Order matters: the ticket is written first, because the ticket is the source
        of truth. The vector index is derived, so if the sync fails the next run of
        ``sync_ticket_to_vectors`` repairs it — no data is lost either way.
        """

        if not advisor_answer or not advisor_answer.strip():
            raise ValueError("advisor answer cannot be empty")

        ticket = await self._require_ticket(ticket_id)
        current = TicketStatus(ticket.status)
        assert_transition_allowed(ticket_id, current, TicketStatus.RESOLVED)

        # Section 5's one veto. Re-checked HERE rather than trusted from whatever the
        # dashboard was shown while the advisor typed: the assessment endpoint is
        # advisory and a client could skip it, and promotion is the path that can
        # poison the knowledge base.
        #
        # Only runs when the advisor actually ticked the box — an answer that is not
        # being promoted cannot contradict anything into the KB, so there is nothing
        # to check and no reason to pay for a judge call.
        hold_reason = None
        if promote_to_kb and self.promotion is not None:
            # check_contradiction, NOT assess.
            #
            # Only the veto is read here. assess() also runs the generalizability
            # judge, whose answer feeds `suggested_promote` — the checkbox DEFAULT,
            # which is meaningless once the advisor has already ticked the box and hit
            # resolve. Every promoted resolve was paying for a judge call it threw
            # away, on the path the rate limits exist for.
            verdict = await self.promotion.check_contradiction(
                question=ticket.question,
                answer=advisor_answer,
                department=ticket.department,
            )
            if verdict.held:
                hold_reason = verdict.reason
                promote_to_kb = False
                self.logger.warning(
                    "promotion_held_on_contradiction",
                    ticket_id=ticket_id,
                    reason=hold_reason,
                    contradiction_score=round(verdict.score, 3),
                )

        ticket = await self.ticket_model.apply_transition(
            ticket_id=ticket_id,
            from_status=current,
            to_status=TicketStatus.RESOLVED,
            actor=advisor_id,
            fields={
                "advisor_answer": advisor_answer.strip(),
                "resolved_by": advisor_id,
                "promoted_to_kb": promote_to_kb,
                "promoted_at": datetime.now(UTC) if promote_to_kb else None,
                "promotion_held": hold_reason is not None,
                "promotion_hold_reason": hold_reason,
            },
        )

        # Everything past this point is DERIVED from the ticket that just committed, so
        # a failure here must not fail the advisor's request — they did their part, and
        # both writes are idempotent and re-runnable from the ticket alone. They are
        # logged at error level with the ticket id so a repair is one call.
        await self._deliver_to_thread(ticket)
        await self._sync_quietly(ticket_id)
        return ticket

    async def reopen(self, ticket_id: int, actor: str, note: str | None = None) -> Ticket:
        """A reopen must re-sync: the promoted answer may now be wrong, and leaving a
        stale vector behind is exactly the drift Section 1 forbids."""

        ticket = await self._require_ticket(ticket_id)
        current = TicketStatus(ticket.status)
        assert_transition_allowed(ticket_id, current, TicketStatus.REOPENED)

        ticket = await self.ticket_model.apply_transition(
            ticket_id=ticket_id,
            from_status=current,
            to_status=TicketStatus.REOPENED,
            actor=actor,
            note=note,
            fields={"promoted_to_kb": False, "promoted_at": None},
        )

        await self._sync_quietly(ticket_id)
        return ticket

    async def release(self, ticket_id: int, actor: str, note: str | None = None) -> Ticket:
        """under_review -> pending. Put a claimed ticket back in the queue.

        The counterpart to claim, and the reason it has to exist: claiming is the only
        door into under_review, and resolving was the only door out. An advisor who
        claimed a ticket and then went home left it parked there permanently — absent
        from the pending queue, owned by nobody, and waiting on a stale-ticket scanner
        that is still phase 2. Every abandoned claim was a dead ticket and a student
        who never got an answer.

        No re-sync: nothing that decides indexing has changed.
        """

        ticket = await self._require_ticket(ticket_id)
        current = TicketStatus(ticket.status)
        assert_transition_allowed(ticket_id, current, TicketStatus.PENDING)

        return await self.ticket_model.apply_transition(
            ticket_id=ticket_id,
            from_status=current,
            to_status=TicketStatus.PENDING,
            actor=actor,
            note=note,
        )

    async def reject(self, ticket_id: int, actor: str, reason: str) -> Ticket:
        """-> rejected. This question will not be answered — out of scope, spam, or
        something no handbook should carry.

        The reason is required because the rejection history is the only record of the
        decision, and "why was this refused?" is the first question anyone reviewing
        the queue will ask.

        Re-syncs deliberately. A rejected ticket must own no vector, and normally it
        cannot: indexing needs RESOLVED-or-CLOSED plus the promoted flag, and the only
        route from there to rejected runs through reopen, which clears it. But reopen's
        sync is best-effort — it logs failures instead of raising — so a vector can
        outlive the flag that justified it. Rejection is the last state that can clean
        that up, and the call is idempotent, so doing it here costs one delete on the
        normal path and closes the hole on the abnormal one.
        """

        if not reason or not reason.strip():
            raise ValueError("a rejection needs a reason: it is the only record of why")

        ticket = await self._require_ticket(ticket_id)
        current = TicketStatus(ticket.status)
        assert_transition_allowed(ticket_id, current, TicketStatus.REJECTED)

        ticket = await self.ticket_model.apply_transition(
            ticket_id=ticket_id,
            from_status=current,
            to_status=TicketStatus.REJECTED,
            actor=actor,
            note=reason,
            fields={"promoted_to_kb": False, "promoted_at": None},
        )

        # UNCONDITIONAL, and a guard here was considered and rejected. Recording why,
        # because "skip the sync when the ticket cannot own a vector" looks like an
        # obvious free win and is not:
        #
        # 1. The obvious guard — `ticket.promoted_to_kb or ticket.status in INDEXABLE`
        #    — reads the ticket AFTER apply_transition has already set the flag False
        #    and the status to rejected, so it skips every single time.
        # 2. Reading the PRE-transition state instead still fails, because REJECTED is
        #    reachable only from `pending` and `under_review`. Neither is in INDEXABLE,
        #    so that half of the condition is dead code.
        # 3. Worse, the orphan this call exists to repair is characterised by exactly
        #    `promoted_to_kb == False` with a live vector: reopen() clears the flag and
        #    THEN syncs best-effort, so a failed sync leaves the flag false and the row
        #    behind. Guarding on the flag skips precisely the case the call was written
        #    for.
        #
        # What remains is the cost: one DELETE matching zero rows on a partial index.
        # The `_require_ticket` round trip above happens either way. That is a fair
        # price for closing the one hole nothing else can close.
        await self._sync_quietly(ticket_id)
        return ticket

    async def close(self, ticket_id: int, actor: str, note: str | None = None) -> Ticket:
        """resolved -> closed. The resolution stood; nothing further is needed.

        Does NOT re-sync, and must not: a closed ticket keeps its promoted answer in
        the knowledge base. See the CLOSED entry in ``sync_ticket_to_vectors``, without
        which this transition would delete the knowledge the loop just earned.
        """

        ticket = await self._require_ticket(ticket_id)
        current = TicketStatus(ticket.status)
        assert_transition_allowed(ticket_id, current, TicketStatus.CLOSED)

        return await self.ticket_model.apply_transition(
            ticket_id=ticket_id,
            from_status=current,
            to_status=TicketStatus.CLOSED,
            actor=actor,
            note=note,
        )

    # -------------------------------------------------- Section 3 delivery

    def _scrub(self, text: str | None, *, where: str, ticket_id: int) -> str | None:
        """Redact PII from advisor free text on its way OUT of the ticket.

        The ticket itself keeps the advisor's exact words: it is the source of truth
        and the audit record, and an advisor's original phrasing is what a later
        reviewer needs. What gets scrubbed are the two DERIVED copies — the thread the
        student reads and the vector the knowledge base retrieves — because those are
        the ones that reach someone other than the advisor who typed them.

        This also keeps the project's existing rule: redaction happens on the way out,
        never on the way in.

        One helper rather than two call sites, because two call sites redacting "the
        same" field is how one of them later stops.
        """
        if not text or not self.app_settings.PII_REDACTION_ENABLED:
            return text

        scrubbed, found = redact(text)
        if found:
            self.logger.warning(
                "pii_redacted",
                where=where,
                ticket_id=ticket_id,
                kinds=dict(found),
            )
        return scrubbed

    async def _deliver_to_thread(self, ticket: Ticket) -> None:
        """Write the advisor's answer into the persisted thread state.

        This is what makes delivery passive (Section 3): the resolution is a second,
        separate write over the same ``thread_id``, and the student reads it from
        ``GET /api/v1/chat/{thread_id}`` whenever they come back. Without it the thread
        keeps the "I've escalated this" placeholder forever and the advisor's answer
        only ever exists on the ticket.
        """

        if self.thread_writer is None:
            self.logger.error(
                "thread_writer_unset_answer_not_delivered",
                ticket_id=ticket.ticket_id,
                thread_id=ticket.thread_id,
            )
            return

        config = {"configurable": {"thread_id": ticket.thread_id}}

        # F7: refuse to CREATE a thread here.
        #
        # `aupdate_state` on a thread_id with no checkpoint does not fail — it writes
        # a new one. So if a checkpoint is ever gone (manual cleanup, a restored
        # database, a future retention policy on the checkpoints table), resolving its
        # ticket would produce a brand-new thread containing exactly one message: the
        # advisor's answer. The student then opens the conversation and reads a reply
        # to nothing, with no question and no holding message above it.
        #
        # This is a seam between two positions the system already holds: the messages
        # reducer states that checkpoints are working memory and may be trimmed, while
        # this write assumed the checkpoint exists. Nothing deletes checkpoints today,
        # so the seam is closed by an absence rather than by a guarantee.
        #
        # Nothing is lost by declining: the answer is on the ticket, which is the
        # source of truth, and still reachable from the advisor UI. The error line is
        # the only way anyone would learn a checkpoint had vanished.
        try:
            snapshot = await self.thread_writer.aget_state(config)
        except Exception as exc:
            self.logger.error(
                "thread_state_unreadable_answer_not_delivered",
                ticket_id=ticket.ticket_id,
                thread_id=ticket.thread_id,
                error=str(exc),
                exc_info=True,
            )
            return

        if snapshot is None or not snapshot.values:
            self.logger.error(
                "thread_missing_answer_not_delivered",
                ticket_id=ticket.ticket_id,
                thread_id=ticket.thread_id,
                detail=(
                    "no checkpoint for this thread; declining to create a one-message "
                    "conversation. The answer remains on the ticket."
                ),
            )
            return

        # Scrubbed once, used for both the transcript turn and the mirrored
        # `answer` field, so the student cannot read one redacted and one raw.
        delivered = self._scrub(
            ticket.advisor_answer, where="thread_delivery", ticket_id=ticket.ticket_id
        )

        try:
            await self.thread_writer.aupdate_state(
                config,
                {
                    # APPENDED, via the messages reducer. This is the fix the whole
                    # transcript refactor exists for: the advisor's answer is the next
                    # turn in the conversation, not a correction that erases whatever
                    # the student was told in between. Section 3 calls this appending
                    # the instructor's message, and a single `answer` slot could not.
                    "messages": [
                        ConversationMessage.advisor(
                            content=delivered,
                            ticket_id=ticket.ticket_id,
                            author=ticket.resolved_by,
                        )
                    ],
                    # Still mirrored onto the run-scoped fields so a client reading
                    # only "the latest answer" sees the advisor's, not the holding
                    # message. The transcript above is what actually preserves history.
                    "answer": delivered,
                    "resolved_by": ticket.resolved_by,
                    "ticket_id": ticket.ticket_id,
                },
                as_node=RESOLUTION_AS_NODE,
            )
        except Exception as exc:
            self.logger.error(
                "thread_delivery_failed",
                ticket_id=ticket.ticket_id,
                thread_id=ticket.thread_id,
                error=str(exc),
                exc_info=True,
            )
            return

        self.logger.info(
            "answer_delivered_to_thread",
            ticket_id=ticket.ticket_id,
            thread_id=ticket.thread_id,
        )

    async def _sync_quietly(self, ticket_id: int) -> None:
        """sync_ticket_to_vectors, with the failure logged instead of raised."""
        try:
            await self.sync_ticket_to_vectors(ticket_id)
        except Exception as exc:
            self.logger.error(
                "ticket_vector_sync_failed",
                ticket_id=ticket_id,
                error=str(exc),
                exc_info=True,
            )

    # ------------------------------------------------------- Section 1 sync

    async def sync_ticket_to_vectors(self, ticket_id: int) -> None:
        """Force pgvector to match the ticket's current state. Idempotent.

        It does not know whether this is a resolve, an edit, an unpromote or a reopen:
        it deletes by the stable key and then inserts the current state if there is one.
        Running it once or ten times leaves the same correct index — which is what makes
        a retry after a crash safe.
        """

        ticket = await self._require_ticket(ticket_id)

        # 1. delete by stable key — never "insert the new one and hope"
        deleted = await self.vectordb_client.delete_by_metadata(
            collection_name=self.collection_name,
            criteria={"ticket_id": str(ticket_id)},
        )

        # CLOSED counts as indexable, not just RESOLVED. Closing is the happy ending
        # of a resolution — the answer stood and nothing further is needed — so a
        # closed ticket keeps its contribution to the knowledge base.
        #
        # This is load-bearing rather than tidy. This method's contract is that it can
        # be re-run at any time and leave the index correct, which is what makes the
        # best-effort calls to it safe and what the phase-2 repair job will rely on. If
        # CLOSED were excluded, every closed ticket's answer would be deleted by the
        # next sync from anywhere — the learning loop would work perfectly right up to
        # the moment an advisor ticked the box that says "this went well".
        INDEXABLE = (TicketStatus.RESOLVED.value, TicketStatus.CLOSED.value)
        should_index = (
            ticket.status in INDEXABLE and ticket.promoted_to_kb and bool(ticket.advisor_answer)
        )

        if not should_index:
            self.logger.info(
                "ticket_vectors_removed",
                ticket_id=ticket_id,
                deleted_rows=deleted,
                status=ticket.status,
            )
            return

        # 2. re-embed the current state and insert exactly one row.
        # to_thread because the provider SDKs are synchronous: embedding inline would
        # block the event loop for the whole round trip.
        # Both halves are free text and both are redacted. The question is the
        # student's own words, so "my number is 010..., when are exams?" would
        # otherwise be embedded verbatim and retrievable by anyone.
        text = (
            f"Question: {self._scrub(ticket.question, where='kb_question', ticket_id=ticket_id)}\n"
            f"Answer: {self._scrub(ticket.advisor_answer, where='kb_answer', ticket_id=ticket_id)}"
        )
        vectors = await asyncio.to_thread(
            self.embedding_client.embed_text,
            text,
            DocumentTypeEnum.DOCUMENT.value,
        )
        if not vectors:
            raise RuntimeError(f"embedding failed while syncing ticket {ticket_id}")

        # record_id stays None: this row derives from a ticket, not a handbook chunk, so
        # it has no chunk_id to point at. The return value IS checked — a provider that
        # refuses the row would otherwise make promotion fail silently, and the ticket
        # would claim promoted_to_kb while the vector never existed.
        inserted = await self.vectordb_client.insert_one(
            collection_name=self.collection_name,
            text=text,
            vector=vectors[0],
            metadata={
                "ticket_id": str(ticket_id),
                "source": INSTRUCTOR_RESOLVED_SOURCE,
                "department": ticket.department,
                "promoted_at": datetime.now(UTC).isoformat(),
            },
            record_id=None,
        )
        if not inserted:
            raise RuntimeError(
                f"vector insert refused while promoting ticket {ticket_id} "
                f"into collection {self.collection_name}"
            )

        self.logger.info("ticket_promoted_to_kb", ticket_id=ticket_id, department=ticket.department)

    # -------------------------------------------------------------- helpers

    async def _require_ticket(self, ticket_id: int) -> Ticket:
        ticket = await self.ticket_model.get_ticket(ticket_id)
        if ticket is None:
            raise LookupError(f"ticket {ticket_id} not found")

        # Every lifecycle method starts here, and this is the first moment the advisor's
        # request knows which thread it belongs to. Binding it means one grep on the
        # thread id spans the student's question, the gate verdicts, the ticket and the
        # advisor's resolution days later — which is the whole point of the id.
        set_correlation_id(ticket.thread_id)
        return ticket

    @staticmethod
    def _serialise_chunks(chunks: list[RetrievedDocument]) -> list[dict]:
        """Store what the bot saw, trimmed. The advisor needs the evidence, not a
        megabyte of JSONB per ticket."""
        return json.loads(
            json.dumps(
                [
                    {
                        "text": c.text[:2000],
                        "score": round(c.score, 4),
                        "metadata": c.metadata,
                    }
                    for c in chunks
                ],
                ensure_ascii=False,
            )
        )
