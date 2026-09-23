"""The transcript must survive across runs, and an advisor's reply must not erase it.

These tests use a throwaway two-node graph over the real ``GraphState`` and an
in-memory checkpointer. No LLM, no database, no controllers — the thing under test is
the state's reducer semantics, which is precisely where the bug was: a single
``answer`` slot meant the advisor's resolution overwrote whatever the student had
last been told, and a ``ticket_id`` that was never reset reported a previous turn's
escalation as the current one's.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from customer_support.models.enums.MessageRoleEnum import MessageRole
from customer_support.models.graph.conversation import ConversationMessage
from customer_support.models.graph.graph_state import GraphState

RESOLUTION_NODE = "reply"


def build_graph():
    """question in -> assistant turn appended -> END."""

    async def reply(state: GraphState) -> dict:
        return {
            "answer": f"answered: {state['question']}",
            "messages": [ConversationMessage.assistant(f"answered: {state['question']}")],
        }

    builder = StateGraph(GraphState)
    builder.add_node(RESOLUTION_NODE, reply)
    builder.set_entry_point(RESOLUTION_NODE)
    builder.add_edge(RESOLUTION_NODE, END)
    return builder.compile(checkpointer=InMemorySaver())


def student_run(question: str) -> dict:
    """What the chat router sends: the new turn appended, run-scoped fields reset."""
    return {
        "question": question,
        "messages": [ConversationMessage.student(question)],
        "ticket_id": None,
        "escalate": False,
        "gate_reason": None,
        "resolved_by": None,
    }


def roles(state) -> list[str]:
    return [message.role.value for message in state["messages"]]


@pytest.mark.asyncio
async def test_messages_accumulate_across_separate_runs():
    """A thread is a conversation, not one exchange. Two invocations on the same
    thread_id must leave four turns, not two."""
    graph = build_graph()
    config = {"configurable": {"thread_id": "t-1"}}

    await graph.ainvoke(student_run("first question"), config=config)
    final = await graph.ainvoke(student_run("second question"), config=config)

    assert roles(final) == ["student", "assistant", "student", "assistant"]
    assert final["messages"][0].content == "first question"
    assert final["messages"][2].content == "second question"


@pytest.mark.asyncio
async def test_separate_threads_do_not_share_a_transcript():
    graph = build_graph()

    await graph.ainvoke(student_run("q for one"), config={"configurable": {"thread_id": "a"}})
    other = await graph.ainvoke(
        student_run("q for two"), config={"configurable": {"thread_id": "b"}}
    )

    assert len(other["messages"]) == 2
    assert other["messages"][0].content == "q for two"


@pytest.mark.asyncio
async def test_an_advisors_reply_appends_instead_of_overwriting():
    """The regression this refactor exists for.

    The advisor resolves hours later through aupdate_state — a separate write over
    the same thread, exactly as EscalationController does it. The student's earlier
    turns must still be there afterwards.
    """
    graph = build_graph()
    config = {"configurable": {"thread_id": "t-esc"}}

    await graph.ainvoke(student_run("what is the fee?"), config=config)

    await graph.aupdate_state(
        config,
        {
            "messages": [
                ConversationMessage.advisor(
                    content="The fee is 500 EGP.", ticket_id=7, author="advisor-nour"
                )
            ],
            "answer": "The fee is 500 EGP.",
            "resolved_by": "advisor-nour",
            "ticket_id": 7,
        },
        as_node=RESOLUTION_NODE,
    )

    snapshot = await graph.aget_state(config)
    assert roles(snapshot.values) == ["student", "assistant", "advisor"]

    advisor_turn = snapshot.values["messages"][-1]
    assert advisor_turn.ticket_id == 7
    assert advisor_turn.author == "advisor-nour"
    # The holding message the student originally received is still in the history.
    assert snapshot.values["messages"][1].content == "answered: what is the fee?"


@pytest.mark.asyncio
async def test_a_later_question_does_not_inherit_a_stale_ticket_id():
    """ticket_id is run-scoped. It used to persist, so a successful answer on a
    thread that had previously escalated still reported escalated=true."""
    graph = build_graph()
    config = {"configurable": {"thread_id": "t-stale"}}

    # A run that escalated: the node did not set it, so simulate it out of band.
    await graph.ainvoke(student_run("unanswerable"), config=config)
    await graph.aupdate_state(config, {"ticket_id": 99}, as_node=RESOLUTION_NODE)
    assert (await graph.aget_state(config)).values["ticket_id"] == 99

    # The next question resets it, the way the router now does.
    final = await graph.ainvoke(student_run("answerable"), config=config)

    assert final["ticket_id"] is None
    assert final["escalate"] is False
    # ...while the transcript is untouched by the reset.
    assert roles(final) == ["student", "assistant", "student", "assistant"]


def test_a_message_is_immutable():
    """Turns are records of what was said. Corrections are new turns."""
    message = ConversationMessage.student("hello")
    with pytest.raises(ValidationError):
        message.content = "something else"


def test_advisor_turns_carry_their_provenance():
    message = ConversationMessage.advisor(content="answer", ticket_id=3, author="advisor-sara")
    assert message.role is MessageRole.ADVISOR
    assert (message.ticket_id, message.author) == (3, "advisor-sara")
    # A student turn has no ticket to point at.
    assert ConversationMessage.student("q").ticket_id is None
