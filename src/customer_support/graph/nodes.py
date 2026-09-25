"""Thin nodes (Section 4).

Every node: read state -> call ONE controller method -> write the result back.
If a node grows past a handful of lines, or gains a prompt or a query, the logic
belongs in a controller.

Each node is built by a factory that closes over GraphDeps, so nodes stay pure
functions of (state) from LangGraph's point of view.
"""

from customer_support.helpers.logging_config import get_logger
from customer_support.models.enums.GateEnum import GateEnum, GateFailureReason
from customer_support.models.graph.conversation import ConversationMessage
from customer_support.models.graph.graph_state import GraphState
from customer_support.models.llm_schemas.gate_result import GateResult

from .dependencies import GraphDeps

logger = get_logger(__name__)


def _record_gates(state: GraphState, results: list[GateResult]) -> dict:
    """Merge gate verdicts into the state shape. The FIRST failure wins, because the
    gates are a fail-fast sequence, not a weighted average."""

    scores = dict(state.get("gate_scores") or {})
    scores.update({r.gate: r.score for r in results})

    failed = next((r for r in results if not r.passed), None)

    return {
        "gate_scores": scores,
        "escalate": failed is not None,
        "failed_gate": failed.gate if failed else None,
        "gate_reason": failed.reason if failed else None,
    }


def make_retrieve_node(deps: GraphDeps):
    async def retrieve_node(state: GraphState) -> dict:
        chunks = await deps.retrieval.retrieve(
            question=state["question"],
            department=state.get("department"),
            limit=deps.settings.RETRIEVAL_TOP_K,
        )
        return {"retrieved_chunks": chunks}

    return retrieve_node


def make_grade_node(deps: GraphDeps):
    async def grade_node(state: GraphState) -> dict:
        result = await deps.grading.check_context_relevance(
            question=state["question"],
            chunks=state.get("retrieved_chunks") or [],
        )
        return _record_gates(state, [result])

    return grade_node


def make_generate_node(deps: GraphDeps):
    async def generate_node(state: GraphState) -> dict:
        answer = await deps.generation.generate_answer(
            question=state["question"],
            chunks=state["retrieved_chunks"],
            # All three are resolved or loaded by the router; the node just carries
            # them through. The controller decides how much history to spend.
            language=state.get("language"),
            profile=state.get("profile"),
            messages=state.get("messages"),
        )
        if answer is None:
            # Generation failed: treat it as a gate failure so the student gets a
            # human, not a 500 and a dead conversation. Recorded as GENERATION, not
            # FAITHFULNESS — nothing was judged, and mislabelling it would skew the
            # faithfulness scores that threshold tuning reads back out of the logs.
            return {
                "answer": None,
                "escalate": True,
                "failed_gate": GateEnum.GENERATION.value,
                "gate_reason": GateFailureReason.GENERATION_FAILED.value,
            }
        return {"answer": answer}

    return generate_node


def make_judge_node(deps: GraphDeps):
    async def judge_node(state: GraphState) -> dict:
        results = await deps.grading.check_post_generation(
            question=state["question"],
            answer=state["answer"],
            chunks=state.get("retrieved_chunks") or [],
        )
        verdict = _record_gates(state, results)

        if not verdict["escalate"]:
            # The answer joins the transcript HERE and nowhere earlier: this is
            # the first point it has survived every gate. Appending it in
            # generate_node would write an answer that later failed
            # faithfulness into the student's history.
            verdict["messages"] = [ConversationMessage.assistant(state["answer"])]

        return verdict

    return judge_node


def make_escalate_node(deps: GraphDeps):
    async def escalate_node(state: GraphState) -> dict:
        ticket = await deps.escalation.create_ticket(state)

        # The drafted answer is deliberately dropped: an answer that failed a gate
        # must never reach the student, not even as a "here's my best guess".
        holding_message = deps.settings.ESCALATION_STUDENT_MESSAGE
        return {
            "ticket_id": ticket.ticket_id,
            "answer": holding_message,
            # The holding message is a real turn: the student saw it, and when the
            # advisor's answer arrives later it must read as the NEXT turn rather
            # than as a correction that erased this one.
            "messages": [ConversationMessage.assistant(holding_message)],
        }

    return escalate_node
