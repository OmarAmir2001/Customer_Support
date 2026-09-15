from typing_extensions import TypedDict

class GraphState(TypedDict):
    question: str
    student_id: str
    department: str | None
    retrieved_chunks: list
    answer: str | None
    escalate: bool
    grade_reason: str | None

