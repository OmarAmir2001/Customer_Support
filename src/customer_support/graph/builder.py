"""Graph assembly. Wiring only — no logic lives here.

Shape (Section 2's cheapest-first ordering):

    retrieve -> grade ──fail──> escalate -> END
                  │
                 pass
                  v
              generate -> judge ──fail──> escalate -> END
                              │
                             pass
                              v
                             END

``escalate_node`` ends the run (Section 3). There is no ``interrupt()``: the advisor
answers hours later, and a paused function cannot survive a restart. Resolution is a
separate short run over the same ``thread_id``.
"""

from langgraph.graph import END, StateGraph

from customer_support.helpers.logging_config import get_logger
from customer_support.models.graph.graph_state import GraphState

from .dependencies import GraphDeps
from .edges import route_after_generate, route_after_grade, route_after_judge
from .nodes import (
    make_escalate_node,
    make_generate_node,
    make_grade_node,
    make_judge_node,
    make_retrieve_node,
)

logger = get_logger(__name__)


def build_graph(deps: GraphDeps, checkpointer=None):
    builder = StateGraph(GraphState)

    builder.add_node("retrieve_node", make_retrieve_node(deps))
    builder.add_node("grade_node", make_grade_node(deps))
    builder.add_node("generate_node", make_generate_node(deps))
    builder.add_node("judge_node", make_judge_node(deps))
    builder.add_node("escalate_node", make_escalate_node(deps))

    builder.set_entry_point("retrieve_node")
    builder.add_edge("retrieve_node", "grade_node")

    builder.add_conditional_edges(
        "grade_node",
        route_after_grade,
        {"escalate_node": "escalate_node", "generate_node": "generate_node"},
    )
    builder.add_conditional_edges(
        "generate_node",
        route_after_generate,
        {"escalate_node": "escalate_node", "judge_node": "judge_node"},
    )
    builder.add_conditional_edges(
        "judge_node",
        route_after_judge,
        {"escalate_node": "escalate_node", END: END},
    )

    builder.add_edge("escalate_node", END)

    compiled = builder.compile(checkpointer=checkpointer)
    logger.info("graph_compiled", persistent=checkpointer is not None)
    return compiled
