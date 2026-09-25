"""Conditional edges (Section 4, step 5).

An edge READS a verdict that a controller already reached and points at the next
node. It never decides anything: "should we escalate?" was answered in
GradingController and stored in state.
"""

from langgraph.graph import END

from customer_support.models.graph.graph_state import GraphState

ESCALATE_NODE = "escalate_node"
GENERATE_NODE = "generate_node"


def route_after_grade(state: GraphState) -> str:
    return ESCALATE_NODE if state.get("escalate") else GENERATE_NODE


def route_after_generate(state: GraphState) -> str:
    # Generation itself can fail; no point judging an answer that does not exist.
    return ESCALATE_NODE if state.get("escalate") else "judge_node"


def route_after_judge(state: GraphState) -> str:
    return ESCALATE_NODE if state.get("escalate") else END
