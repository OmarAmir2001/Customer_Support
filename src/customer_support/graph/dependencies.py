"""What the nodes are allowed to reach.

Built once in the lifespan and handed to ``build_graph``. Nodes close over this
object, so the graph never imports FastAPI and never touches ``app.state`` — which
is what lets a test build a graph with fake controllers.
"""

from dataclasses import dataclass

from customer_support.controllers.EscalationController import EscalationController
from customer_support.controllers.GenerationController import GenerationController
from customer_support.controllers.GradingController import GradingController
from customer_support.controllers.RetrievalController import RetrievalController
from customer_support.helpers.config import Settings


@dataclass(frozen=True)
class GraphDeps:
    retrieval: RetrievalController
    grading: GradingController
    generation: GenerationController
    escalation: EscalationController
    settings: Settings